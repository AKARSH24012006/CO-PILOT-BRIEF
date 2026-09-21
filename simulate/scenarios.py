"""Canonical demo scenarios.

Pure data — timestamped transcript fragments — reproducing the three
Input & Output Event Specifications from the Theme 4 guide, ported into the
CopilotBrief aviation-briefing domain. Nothing here is read by the pipeline
as a query or a canned answer; it is streamed in exactly the way a live
transcript would be, one fragment at a time, through the same
controller -> decomposer -> retrieval -> synthesis path any other input
takes. This keeps the "No Hardcoding / No Precomputation" rule intact:
these are demo *inputs*, not shortcuts baked into the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Turn:
    """One user utterance, streamed as timestamped fragments."""
    chunks: list[tuple[float, str]]  # (timestamp_s, fragment_text)


@dataclass
class Scenario:
    key: str
    title: str
    description: str
    session_id: str
    turns: list[Turn]


SCENARIO_1_MULTI_INTENT = Scenario(
    key="multi_intent",
    title="Example 1 — Incremental Multi-Intent Utterance",
    description=(
        "A dispatcher packs three questions into one streamed utterance. "
        "The engine begins retrieval before the utterance ends, decomposes "
        "into three sub-queries once the 'and' boundaries appear, and "
        "synthesizes a single grounded answer covering all three."
    ),
    session_id="demo-multi-intent",
    turns=[
        Turn(chunks=[
            (0.0, "Before we release this flight, I need to know"),
            (0.8, "the crosswind limit for the A320 on a wet runway,"),
            (1.6, "whether we need de-icing at minus two degrees,"),
            (2.4, "and the diversion fuel reserve for an international routing."),
        ]),
    ],
)


SCENARIO_2_LATE_REFINEMENT = Scenario(
    key="late_refinement",
    title="Example 2 — Late-Arriving Detail (Refine, Do Not Restart)",
    description=(
        "An initial question is answered from the domestic fuel-reserve "
        "rule. A follow-up utterance then introduces a late constraint "
        "('the flight is now international'); the engine narrows the "
        "existing answer in place rather than clearing session context "
        "and re-running a full search."
    ),
    session_id="demo-late-refinement",
    turns=[
        Turn(chunks=[
            (0.0, "What is the diversion fuel reserve requirement for a domestic flight?"),
        ]),
        Turn(chunks=[
            (6.0, "Actually, the flight is now international."),
        ]),
    ],
)


SCENARIO_3_QUERY_SUPPRESSION = Scenario(
    key="query_suppression",
    title="Example 3 — Query Suppression (No Retrieval Required)",
    description=(
        "After an answer is given, the user asks only for a presentation "
        "change. The controller recognizes this needs no corpus search at "
        "all and reformats the existing session context instead."
    ),
    session_id="demo-query-suppression",
    turns=[
        Turn(chunks=[
            (0.0, "What is the crosswind limit for the A320 on a wet runway?"),
        ]),
        Turn(chunks=[
            (5.0, "Please repeat your last answer in two bullets."),
        ]),
    ],
)


ALL_SCENARIOS: list[Scenario] = [
    SCENARIO_1_MULTI_INTENT,
    SCENARIO_2_LATE_REFINEMENT,
    SCENARIO_3_QUERY_SUPPRESSION,
]


def get_scenario(key: str) -> Scenario:
    for s in ALL_SCENARIOS:
        if s.key == key:
            return s
    raise KeyError(f"Unknown scenario: {key}")
