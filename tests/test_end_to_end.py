"""End-to-end pipeline tests replaying the three canonical scenarios.
This is the automated replay suite Gate G1 asks for: it must complete with
no manual intervention and no network/API key."""

from copilotbrief.models import ControllerDecision
from simulate.scenarios import get_scenario
from simulate.stream_simulator import replay_scenario


def test_example1_multi_intent_end_to_end(engine):
    scenario = get_scenario("multi_intent")
    events = list(replay_scenario(engine, scenario, speed=0))

    decisions = [e["controller_decision"] for e in events]
    assert decisions[0] == ControllerDecision.WAIT.value
    assert ControllerDecision.PROVISIONAL_RETRIEVE.value in decisions
    assert ControllerDecision.MULTI_INTENT_RETRIEVE.value in decisions

    final_answer = events[-1]["answer"]
    assert final_answer is not None
    # all three sub-topics should be represented in the grounded citations
    doc_ids = {c.split(" §")[0] for c in final_answer["citations"]}
    assert "Doc_01" in doc_ids  # crosswind
    assert "Doc_02" in doc_ids  # de-icing
    assert "Doc_03" in doc_ids  # fuel reserve


def test_example2_refinement_narrows_without_losing_thread(engine):
    scenario = get_scenario("late_refinement")
    events = list(replay_scenario(engine, scenario, speed=0))

    answers = [e["answer"] for e in events if e["answer"] is not None]
    assert len(answers) == 2
    v1, v2 = answers
    assert v2["version"] == v1["version"] + 1
    # the refined answer should now cite the international rule, not just
    # the original domestic one
    assert any(c == "Doc_03 §3" for c in v2["citations"])


def test_example3_query_suppression_makes_no_new_retrieval(engine):
    scenario = get_scenario("query_suppression")
    events = list(replay_scenario(engine, scenario, speed=0))

    suppression_events = [e for e in events if e["controller_decision"] == ControllerDecision.NO_RETRIEVAL.value]
    assert len(suppression_events) == 1

    answers = [e["answer"] for e in events if e["answer"] is not None]
    v1, v2 = answers
    assert v2["citations"] == v1["citations"]  # no new evidence introduced
    assert v2["text"] != v1["text"]  # but presentation changed (bulleted)


def test_full_replay_suite_runs_without_manual_intervention(engine):
    """Smoke test: all three scenarios run back-to-back with no exceptions,
    matching Gate G1's automated replay requirement."""
    from simulate.scenarios import ALL_SCENARIOS

    for scenario in ALL_SCENARIOS:
        events = list(replay_scenario(engine, scenario, speed=0))
        assert len(events) > 0
        assert events[-1]["answer"] is not None
