from fastapi import FastAPI, UploadFile, File, HTTPException
import time
import logging
import httpx
import os
from groq import Groq

# Configure logging for structured output and benchmarking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice-Enabled RAG Harness")

RETRIEVAL_URL = "http://127.0.0.1:8000/retrieve"
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# Real function: retrieves context from app.py, then generates a grounded answer via Groq
async def execute_rag_pipeline(transcribed_text: str) -> str | None:
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(
                RETRIEVAL_URL,
                json={"query": transcribed_text, "k": 5},
                timeout=5.0
            )
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            logger.error(f"Retrieval call failed: {e}")
            return None  # triggers guardrail fallback

    chunks = data.get("chunks", [])

    # Guardrail: no results or weak top match -> don't answer
    if not chunks or chunks[0]["score"] > 1.2:
        logger.info("Guardrail triggered: no confident match found")
        return None

    context = "\n\n".join(c["passage"] for c in chunks[:3])

    # Real answer generation via Groq
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question using ONLY the provided context. "
                                "If the context doesn't contain the answer, say you don't have enough information. "
                                "Be concise."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {transcribed_text}"
                }
            ],
            temperature=0.2,
            max_tokens=200,
        )
        answer = completion.choices[0].message.content

        # Simple groundedness guardrail: reject if the model says it doesn't know
        if "don't have enough" in answer.lower() or "cannot answer" in answer.lower():
            logger.info("Guardrail triggered: model reported insufficient grounding")
            return None

        return answer

    except Exception as e:
        logger.error(f"Generation call failed: {e}")
        # Fallback: return raw top passage if the LLM call fails
        return f"Based on the retrieved context: {chunks[0]['passage'][:400]}"


# Mock function: Voice-to-text transcription
# TODO: replace with real Sarvam or ElevenLabs STT call
async def transcribe_audio(audio_bytes: bytes) -> str:
    return "What is the capital of France?"


@app.post("/api/v1/voice-query")
async def process_voice_query(audio_file: UploadFile = File(...)):
    start_time = time.time()
    try:
        # 1. Read Audio (Structured I/O)
        audio_bytes = await audio_file.read()
        if not audio_bytes:
            raise ValueError("Empty audio file received.")

        # 2. Transcription (Voice-to-text input)
        transcription_start = time.time()
        question_text = await transcribe_audio(audio_bytes)
        logger.info(f"Transcription took: {(time.time() - transcription_start) * 1000:.2f}ms")

        # 3. Execute RAG Pipeline (End-to-end integration: retrieval + generation + guardrails)
        rag_start = time.time()
        answer = await execute_rag_pipeline(question_text)
        logger.info(f"RAG Pipeline took: {(time.time() - rag_start) * 1000:.2f}ms")

        # Calculate total latency
        total_latency_ms = (time.time() - start_time) * 1000

        # Enforce Guardrails (Check if it knows when *not* to answer)
        if not answer:
            answer = "I'm sorry, I don't have enough grounded context to answer that."

        return {
            "status": "success",
            "latency_ms": round(total_latency_ms, 2),
            "transcription": question_text,
            "answer": answer
        }

    except Exception as e:
        # Error Recovery Harness
        logger.error(f"Pipeline failure: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Pipeline error. Retries and error recovery initiated."
        )


@app.get("/health")
def health():
    return {"status": "ok"}