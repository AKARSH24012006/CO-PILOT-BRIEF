from copilotbrief.models import SubQuery
from copilotbrief.session import SessionStore
from copilotbrief.synthesis import SynthesisEngine
from copilotbrief.telemetry import TelemetryLogger


def test_fresh_synthesis_grounds_and_cites(retriever):
    tel = TelemetryLogger()
    syn = SynthesisEngine(retriever, tel)
    store = SessionStore()
    sess = store.get_or_create("t1")

    v = syn.synthesize_fresh(
        sess, [SubQuery(text="crosswind limit for the A320 on a wet runway")],
        trigger="provisional", timestamp_s=0.0,
    )
    assert v.version == 1
    assert v.citations
    assert all(c.startswith("Doc_") for c in v.citations)


def test_out_of_domain_query_flags_uncertainty_not_fabrication(retriever):
    tel = TelemetryLogger()
    syn = SynthesisEngine(retriever, tel)
    store = SessionStore()
    sess = store.get_or_create("t2")

    v = syn.synthesize_fresh(
        sess, [SubQuery(text="what is the best pizza topping in Naples")],
        trigger="provisional", timestamp_s=0.0,
    )
    assert v.citations == []
    assert v.uncertainty is not None


def test_refinement_mutates_only_the_affected_claim(retriever):
    tel = TelemetryLogger()
    syn = SynthesisEngine(retriever, tel)
    store = SessionStore()
    sess = store.get_or_create("t3")

    v1 = syn.synthesize_fresh(
        sess, [SubQuery(text="diversion fuel reserve requirement for a domestic flight")],
        trigger="provisional", timestamp_s=0.0,
    )
    v2 = syn.synthesize_refinement(
        sess, [SubQuery(text="the flight is now international")], timestamp_s=5.0,
    )
    assert v2.version == 2
    assert v2.text != v1.text
    # session-bound state: still exactly one active claim (mutated in place,
    # not appended as a second parallel claim)
    assert len(sess.active_claim_order) == 1


def test_presentation_only_does_not_touch_citations(retriever):
    tel = TelemetryLogger()
    syn = SynthesisEngine(retriever, tel)
    store = SessionStore()
    sess = store.get_or_create("t4")

    v1 = syn.synthesize_fresh(
        sess, [SubQuery(text="crosswind limit for the A320 on a wet runway")],
        trigger="provisional", timestamp_s=0.0,
    )
    v2 = syn.synthesize_presentation_only(sess, "please repeat that in two bullets", timestamp_s=5.0)
    assert v2.citations == v1.citations
    assert v2.text.startswith("-")


def test_session_state_is_isolated_between_sessions(retriever):
    tel = TelemetryLogger()
    syn = SynthesisEngine(retriever, tel)
    store = SessionStore()
    sess_a = store.get_or_create("a")
    sess_b = store.get_or_create("b")

    syn.synthesize_fresh(sess_a, [SubQuery(text="crosswind limit for the A320")], "provisional", 0.0)
    assert sess_b.claims == {}
    assert sess_b.active_claim_order == []
