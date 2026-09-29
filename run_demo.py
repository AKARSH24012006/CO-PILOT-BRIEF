#!/usr/bin/env python3
"""CopilotBrief demo runner — the single-command entry point for Gate G1
(Reproducibility): `python run_demo.py` launches on a clean machine with no
API key and no network access required, replays the three canonical
scenarios from the Theme 4 spec (ported into the aviation-briefing
domain) plus one bonus scenario demonstrating refinement-target
ambiguity handling, and prints the streaming trace, the final grounded
answer, and the structured telemetry record for each.

Usage:
    python run_demo.py                      # all 4 scenarios, instant replay
    python run_demo.py --scenario multi_intent
    python run_demo.py --speed 1.0           # real-time playback (for recording)
    python run_demo.py --json                # machine-readable telemetry dump
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from copilotbrief.pipeline import CopilotBriefEngine
from copilotbrief.telemetry import TelemetryLogger
from simulate.scenarios import ALL_SCENARIOS, get_scenario
from simulate.stream_simulator import replay_scenario

logging.basicConfig(level=logging.ERROR)  # keep stdout clean; dense-encoder
# fallback warnings are informative but noisy for a demo transcript


def _print_rule(char="-", width=78):
    print(char * width)


def run_scenario(engine: CopilotBriefEngine, scenario, as_json: bool, speed: float) -> dict:
    if not as_json:
        print()
        _print_rule("=")
        print(scenario.title)
        print(scenario.description)
        _print_rule("=")

    last_answer = None
    for event in replay_scenario(engine, scenario, speed=speed):
        if not as_json:
            marker = " [utterance end]" if event["is_utterance_end"] else ""
            print(f"  t={event['timestamp_s']:>5.1f}s  \"{event['chunk_text']}\"{marker}")
            print(f"            -> {event['controller_decision']:<24} ({event['controller_reason']})")
            if event["answer"] is not None:
                a = event["answer"]
                print(f"            -> answer v{a['version']}: {a['text']}")
                print(f"               citations: {a['citations']}")
                if a["uncertainty"]:
                    print(f"               uncertainty: {a['uncertainty']}")
                last_answer = a

    session = engine.get_session(scenario.session_id)
    record = TelemetryLogger.structured_output_record(session)

    if not as_json:
        print()
        print("  Structured Output Event Record:")
        print("  " + json.dumps(record, indent=2).replace("\n", "\n  "))

    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CopilotBrief demo scenarios.")
    parser.add_argument("--scenario", choices=[s.key for s in ALL_SCENARIOS], default=None,
                         help="Run only this scenario (default: all three).")
    parser.add_argument("--speed", type=float, default=0.0,
                         help="Playback speed multiplier for inter-chunk delay (0 = instant, 1 = real time).")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON telemetry only.")
    parser.add_argument("--corpus-dir", default=None, help="Override the corpus directory.")
    args = parser.parse_args()

    if not args.json:
        print("CopilotBrief — Streaming Live RAG for aviation pre-flight briefings")
        print("(Samsung PRISM GenAI Hackathon 3.0 — Theme 4)")

    engine = CopilotBriefEngine(corpus_dir=args.corpus_dir)
    if not args.json:
        print(f"Corpus loaded. Dense retrieval backend: {engine.retriever.dense_backend_name}")

    scenarios = [get_scenario(args.scenario)] if args.scenario else ALL_SCENARIOS

    records = {}
    for scenario in scenarios:
        records[scenario.key] = run_scenario(engine, scenario, as_json=args.json, speed=args.speed)

    if args.json:
        print(json.dumps(records, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
