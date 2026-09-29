"""FastAPI service exposing CopilotBrief over a WebSocket (for the live
streaming demo UI) plus a couple of small REST endpoints for telemetry and
scenario listing. This is the "Output: Streamed Answer + Grounded
Citations + Observability Telemetry" edge of the architecture diagram.

Run with:  uvicorn server.main:app --reload --port 8000
or simply: python server/main.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python server/main.py`

from copilotbrief.models import TranscriptChunk  # noqa: E402
from copilotbrief.pipeline import CopilotBriefEngine  # noqa: E402
from copilotbrief.telemetry import TelemetryLogger  # noqa: E402
from simulate.scenarios import ALL_SCENARIOS, get_scenario  # noqa: E402

logging.basicConfig(level=logging.WARNING)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="CopilotBrief", description="Streaming Live RAG for aviation pre-flight briefings")

# One shared engine (one shared corpus index + retriever) for the process;
# session state itself is still fully isolated per session_id inside it.
engine = CopilotBriefEngine()


def _answer_payload(answer) -> dict | None:
    if answer is None:
        return None
    return {
        "version": answer.version,
        "text": answer.text,
        "citations": answer.citations,
        "uncertainty": answer.uncertainty,
        "changed_claim_ids": answer.changed_claim_ids,
        "refinement_ambiguity": answer.refinement_ambiguity,
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/scenarios")
def list_scenarios():
    return [
        {"key": s.key, "title": s.title, "description": s.description}
        for s in ALL_SCENARIOS
    ]


@app.get("/api/corpus")
def get_corpus_catalog():
    """Doc/section title catalog — lets the UI resolve a bare "[Doc_03 §2]"
    citation into a readable title without shipping full chunk text."""
    return engine.corpus_catalog


@app.get("/api/telemetry/{session_id}")
def get_telemetry(session_id: str):
    session = engine.get_session(session_id)
    return JSONResponse(TelemetryLogger.structured_output_record(session))


@app.get("/api/health")
def health():
    return {"status": "ok", "dense_backend": engine.retriever.dense_backend_name}


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    async def send_step(event_type: str, payload: dict):
        await websocket.send_json({"event": event_type, **payload})

    try:
        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")

            if action == "reset":
                engine.reset_session(session_id)
                await send_step("reset_ack", {"session_id": session_id})

            elif action == "chunk":
                chunk = TranscriptChunk(
                    session_id=session_id,
                    text=msg.get("text", ""),
                    timestamp_s=float(msg.get("timestamp_s", 0.0)),
                    is_utterance_end=bool(msg.get("is_utterance_end", False)),
                )
                _t0 = time.perf_counter()
                result = engine.process_chunk(chunk)
                processing_ms = round((time.perf_counter() - _t0) * 1000, 2)
                session = engine.get_session(session_id)
                await send_step("step", {
                    "timestamp_s": chunk.timestamp_s,
                    "chunk_text": chunk.text,
                    "controller_decision": result.controller_decision.value,
                    "controller_reason": result.controller_reason,
                    "is_utterance_end": result.is_utterance_end,
                    "synthesis_mode": result.synthesis_mode,
                    "answer": _answer_payload(result.answer),
                    "claims": session.claims_snapshot(),
                    "processing_ms": processing_ms,
                })
                if chunk.is_utterance_end:
                    await send_step("telemetry", TelemetryLogger.structured_output_record(session))

            elif action == "run_scenario":
                key = msg.get("scenario_key")
                speed = float(msg.get("speed", 0.4))
                scenario = get_scenario(key)
                engine.reset_session(scenario.session_id)
                await send_step("scenario_start", {
                    "key": scenario.key, "title": scenario.title, "description": scenario.description,
                    "session_id": scenario.session_id,
                })
                prev_ts = None
                for turn in scenario.turns:
                    for i, (ts, text) in enumerate(turn.chunks):
                        if speed > 0 and prev_ts is not None:
                            await asyncio.sleep(max(0.0, ts - prev_ts) * speed)
                        prev_ts = ts
                        is_end = i == len(turn.chunks) - 1
                        chunk = TranscriptChunk(
                            session_id=scenario.session_id, text=text, timestamp_s=ts, is_utterance_end=is_end,
                        )
                        _t0 = time.perf_counter()
                        result = engine.process_chunk(chunk)
                        processing_ms = round((time.perf_counter() - _t0) * 1000, 2)
                        session = engine.get_session(scenario.session_id)
                        await send_step("step", {
                            "timestamp_s": ts,
                            "chunk_text": text,
                            "controller_decision": result.controller_decision.value,
                            "controller_reason": result.controller_reason,
                            "is_utterance_end": is_end,
                            "synthesis_mode": result.synthesis_mode,
                            "answer": _answer_payload(result.answer),
                            "claims": session.claims_snapshot(),
                            "processing_ms": processing_ms,
                        })
                session = engine.get_session(scenario.session_id)
                await send_step("telemetry", TelemetryLogger.structured_output_record(session))
                await send_step("scenario_end", {"key": scenario.key})

            else:
                await send_step("error", {"message": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        pass


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=False)
