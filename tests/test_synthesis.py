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


def test_negated_sentence_does_not_outrank_the_actual_answer(retriever):
    """Regression test for benchmark/report.md edge case #1: a short,
    lexically-dense NEGATIVE sentence elsewhere in the corpus ("Extra fuel
    does not reduce the mandatory Final Reserve...") used to out-score the
    longer sentence that actually answers the question, because Dice
    rewards small denominators and has no notion of polarity. The query
    itself is NOT negated, so the correct (positive) §3 international-fuel
    rule should win over the off-topic negated §5 sentence."""
    tel = TelemetryLogger()
    syn = SynthesisEngine(retriever, tel)
    store = SessionStore()
    sess = store.get_or_create("t5")

    v = syn.synthesize_fresh(
        sess,
        [SubQuery(text="does an international flight need additional reserve fuel beyond the standard reserve")],
        trigger="provisional", timestamp_s=0.0,
    )
    assert "Doc_03 §3" in v.citations, (
        f"expected the international 45-minute reserve rule (Doc_03 §3) to be cited, got {v.citations}"
    )


def test_regulatory_no_less_than_phrasing_is_not_treated_as_negation(retriever):
    """The negation-penalty heuristic above must not fire on "no less
    than"/"no lower than" style regulatory phrasing, which states a
    positive numeric floor, not a negation — this exact corpus sentence
    ("...DH of no lower than 200 feet and RVR/visibility no less than 550
    meters...") was the actual answer for this query and must still win."""
    tel = TelemetryLogger()
    syn = SynthesisEngine(retriever, tel)
    store = SessionStore()
    sess = store.get_or_create("t6")

    v = syn.synthesize_fresh(
        sess,
        [SubQuery(text="What is the CAT I landing decision height and visibility minimum?")],
        trigger="provisional", timestamp_s=0.0,
    )
    assert "Doc_04 §2" in v.citations, f"expected Doc_04 §2, got {v.citations}"


def test_refinement_target_ambiguity_is_surfaced_not_silent(retriever):
    """Regression test for benchmark/report.md edge case #2: when 2+ open
    topics exist and a late constraint carries no topic/citation signal
    tying it to one of them, the system still has to pick a target (most
    recently active claim), but should now say the pick was a guess
    instead of staying silent about it."""
    tel = TelemetryLogger()
    syn = SynthesisEngine(retriever, tel)
    store = SessionStore()
    sess = store.get_or_create("t7")

    syn.synthesize_fresh(
        sess,
        [
            SubQuery(text="crosswind limit for the A320 on a dry runway"),
            SubQuery(text="final reserve fuel required for a domestic sector"),
        ],
        trigger="multi_intent", timestamp_s=0.0,
    )
    assert len(sess.active_claim_order) == 2

    v2 = syn.synthesize_refinement(
        sess, [SubQuery(text="actually use the updated numbers")], timestamp_s=5.0,
    )
    assert v2.refinement_ambiguity is not None
    assert "open topics" in v2.refinement_ambiguity


def test_refinement_ambiguity_not_flagged_with_a_single_open_topic(retriever):
    """The ordinary single-thread case (spec Example 2) must NOT be
    flagged as ambiguous — there's only one plausible target, so applying
    the fallback there is routine, not a guess worth calling out."""
    tel = TelemetryLogger()
    syn = SynthesisEngine(retriever, tel)
    store = SessionStore()
    sess = store.get_or_create("t8")

    syn.synthesize_fresh(
        sess, [SubQuery(text="diversion fuel reserve requirement for a domestic flight")],
        trigger="provisional", timestamp_s=0.0,
    )
    v2 = syn.synthesize_refinement(
        sess, [SubQuery(text="the flight is now international")], timestamp_s=5.0,
    )
    assert v2.refinement_ambiguity is None


def test_session_state_is_isolated_between_sessions(retriever):
    tel = TelemetryLogger()
    syn = SynthesisEngine(retriever, tel)
    store = SessionStore()
    sess_a = store.get_or_create("a")
    sess_b = store.get_or_create("b")

    syn.synthesize_fresh(sess_a, [SubQuery(text="crosswind limit for the A320")], "provisional", 0.0)
    assert sess_b.claims == {}
    assert sess_b.active_claim_order == []
