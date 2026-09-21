"""Held-out benchmark prompts — separate from the three demo scenarios in
simulate/scenarios.py. These exercise a broader slice of the corpus and
carry lightweight ground-truth labels (expected source doc, expected
sub-intent count) so evaluate.py can score the pipeline objectively rather
than just eyeballing output.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SingleIntentCase:
    text: str
    expected_doc: str  # e.g. "Doc_01" — the section this should ground in
    streamed: list[tuple[float, str]] | None = None  # None = single-chunk


@dataclass
class MultiIntentCase:
    text_chunks: list[tuple[float, str]]
    expected_min_subintents: int
    expected_docs: set[str] = field(default_factory=set)


@dataclass
class OutOfDomainCase:
    text: str


SINGLE_INTENT_CASES: list[SingleIntentCase] = [
    SingleIntentCase("What is the crosswind limit for the A320 on a dry runway?", "Doc_01"),
    SingleIntentCase("How long does Type IV de-icing fluid hold over in light snow?", "Doc_02"),
    SingleIntentCase("What final reserve fuel is required for a domestic sector?", "Doc_03"),
    SingleIntentCase("What is the CAT I landing decision height and visibility minimum?", "Doc_04"),
    SingleIntentCase("What is the maximum holding speed above 14000 feet?", "Doc_05"),
    SingleIntentCase("What NOTAM classes does dispatch have to review before release?", "Doc_06"),
    SingleIntentCase("What documents must be carried onboard for an international sector?", "Doc_07"),
    SingleIntentCase("What does a runway condition code of 3 or below require?", "Doc_08"),
    SingleIntentCase("When is a destination alternate not required?", "Doc_09"),
    SingleIntentCase("What must every dispatch release contain?", "Doc_10"),
    SingleIntentCase("What is the maximum unaugmented flight duty period for two pilots?", "Doc_11"),
    SingleIntentCase("What are the MEL rectification interval categories?", "Doc_12"),
    SingleIntentCase("How far should crews avoid a thunderstorm cell when practical?", "Doc_13"),
    SingleIntentCase("What RVR is required for a CAT IIIB autoland approach?", "Doc_14"),
    SingleIntentCase("When must a crew declare MAYDAY FUEL?", "Doc_15"),
    # streamed (multi-chunk) versions, to feed the "early retrieval" gate
    SingleIntentCase(
        "What is the crosswind limit for the A320 on a wet runway with gusts?", "Doc_01",
        streamed=[(0.0, "Before departure I need to check the crosswind limit for the A320,"), (0.9, "on a wet runway with gusts.")],
    ),
    SingleIntentCase(
        "What is the diversion fuel reserve for an international sector?", "Doc_03",
        streamed=[(0.0, "For dispatch planning, what's the diversion fuel reserve,"), (0.9, "for an international sector?")],
    ),
    SingleIntentCase(
        "What braking action requires a joint risk assessment?", "Doc_08",
        streamed=[(0.0, "Quick question about runway conditions —"), (0.9, "what braking action requires a joint risk assessment?")],
    ),
]

MULTI_INTENT_CASES: list[MultiIntentCase] = [
    MultiIntentCase(
        text_chunks=[
            (0.0, "I need the CAT II landing minima,"),
            (0.9, "the runway condition code threshold for recompute,"),
            (1.8, "and the MEL rectification interval for category B items."),
        ],
        expected_min_subintents=3,
        expected_docs={"Doc_14", "Doc_08", "Doc_12"},
    ),
    MultiIntentCase(
        text_chunks=[
            (0.0, "What's the maximum FDP for an augmented crew,"),
            (1.0, "and what's the minimum rest period before the next duty?"),
        ],
        expected_min_subintents=2,
        expected_docs={"Doc_11"},
    ),
    MultiIntentCase(
        text_chunks=[
            (0.0, "Tell me the thunderstorm avoidance distance,"),
            (1.0, "whether ground handling must stop for lightning,"),
            (2.0, "and the seatbelt sign policy for turbulence."),
        ],
        expected_min_subintents=3,
        expected_docs={"Doc_13"},
    ),
    MultiIntentCase(
        text_chunks=[
            (0.0, "What documents does international crew need,"),
            (1.0, "and how far ahead must dispatch notify customs?"),
        ],
        expected_min_subintents=2,
        expected_docs={"Doc_07"},
    ),
    MultiIntentCase(
        text_chunks=[
            (0.0, "I need the CAT III RVR minimum,"),
            (1.0, "the number of alternates required near minima,"),
            (2.0, "and the mandatory rest facility class for a 4-pilot augmented crew."),
        ],
        expected_min_subintents=3,
        expected_docs={"Doc_14", "Doc_09", "Doc_11"},
    ),
]

OUT_OF_DOMAIN_CASES: list[OutOfDomainCase] = [
    OutOfDomainCase("What's the best pizza topping in Naples?"),
    OutOfDomainCase("Recommend a good science fiction novel about space travel."),
    OutOfDomainCase("What's the capital of Australia?"),
]
