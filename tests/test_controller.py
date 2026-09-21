from copilotbrief.controller import RetrievalController
from copilotbrief.models import ControllerDecision, TranscriptChunk


def _evaluate_stream(controller, session_id, fragments):
    verdicts = []
    for i, (ts, text) in enumerate(fragments):
        is_end = i == len(fragments) - 1
        chunk = TranscriptChunk(session_id=session_id, text=text, timestamp_s=ts, is_utterance_end=is_end)
        verdicts.append(controller.evaluate(chunk))
    return verdicts


def test_waits_on_incomplete_utterance():
    c = RetrievalController()
    v = c.evaluate(TranscriptChunk(session_id="s", text="I need to plan a customer workshop in", timestamp_s=0.0))
    assert v.decision == ControllerDecision.WAIT


def test_provisional_retrieve_on_stable_entity():
    c = RetrievalController()
    verdicts = _evaluate_stream(c, "s", [
        (0.0, "I need to plan a customer workshop in"),
        (0.8, "Pune for 30 people, and I need"),
    ])
    assert verdicts[0].decision == ControllerDecision.WAIT
    assert verdicts[1].decision == ControllerDecision.PROVISIONAL_RETRIEVE


def test_multi_intent_retrieve_on_conjunction():
    c = RetrievalController()
    verdicts = _evaluate_stream(c, "s", [
        (0.0, "I need to plan a customer workshop in"),
        (0.8, "Pune for 30 people, and I need"),
        (1.6, "the cancellation policy and the catering options."),
    ])
    assert verdicts[2].decision == ControllerDecision.MULTI_INTENT_RETRIEVE
    assert "3" in verdicts[2].reason


def test_no_retrieval_on_presentation_only_turn_with_prior_answer():
    c = RetrievalController()
    v = c.evaluate(
        TranscriptChunk(session_id="s", text="Please repeat your last answer in two bullets.", timestamp_s=0.0),
        has_prior_answer=True,
    )
    assert v.decision == ControllerDecision.NO_RETRIEVAL
    assert v.reason == "presentation_restructure"


def test_presentation_phrase_ignored_without_prior_answer():
    """The same phrasing without any prior answer in the session shouldn't
    be misread as reformatting — there's nothing to reformat."""
    c = RetrievalController()
    v = c.evaluate(
        TranscriptChunk(session_id="s", text="Please repeat your last answer in two bullets.", timestamp_s=0.0),
        has_prior_answer=False,
    )
    assert v.decision != ControllerDecision.NO_RETRIEVAL


def test_provisional_fires_only_once_per_utterance():
    c = RetrievalController()
    verdicts = _evaluate_stream(c, "s", [
        (0.0, "What is the crosswind limit for the A320"),
        (0.5, "on a wet runway at Boston?"),
    ])
    assert verdicts[0].decision == ControllerDecision.PROVISIONAL_RETRIEVE
    # second chunk shouldn't re-fire provisional (already fired), and has no
    # new "and"-joined clause, so it should wait rather than retrieve again
    assert verdicts[1].decision == ControllerDecision.WAIT


def test_reset_utterance_clears_state_for_next_turn():
    c = RetrievalController()
    _evaluate_stream(c, "s", [(0.0, "What is the crosswind limit for the A320?")])
    c.reset_utterance("s")
    v = c.evaluate(TranscriptChunk(session_id="s", text="What is the fuel reserve rule?", timestamp_s=0.0))
    assert v.decision == ControllerDecision.PROVISIONAL_RETRIEVE
