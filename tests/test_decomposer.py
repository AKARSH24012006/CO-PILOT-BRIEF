from copilotbrief.decomposer import MultiIntentDecomposer


def test_single_intent_not_fragmented():
    d = MultiIntentDecomposer()
    result = d.decompose("What is the crosswind limit for the A320 on a wet runway?")
    assert len(result) == 1


def test_multi_intent_splits_into_three():
    d = MultiIntentDecomposer()
    text = (
        "Before we release this flight, I need to know the crosswind limit "
        "for the A320 on a wet runway, whether we need de-icing at minus "
        "two degrees, and the diversion fuel reserve for an international routing."
    )
    result = d.decompose(text)
    assert len(result) == 3
    topics = {sq.topic_hint for sq in result}
    assert topics == {"crosswind", "deicing", "fuel_reserve"}


def test_shared_entity_carried_into_later_clauses():
    d = MultiIntentDecomposer()
    text = "What is the crosswind limit for the A320, and what is the fuel reserve rule?"
    result = d.decompose(text)
    assert len(result) == 2
    # the aircraft type mentioned only in the first clause should be carried
    # into the second so retrieval for it isn't context-free
    assert "A320" in result[1].text


def test_trailing_dangling_fragment_is_dropped_not_fragmented():
    d = MultiIntentDecomposer()
    result = d.decompose("What is the crosswind limit and also")
    assert len(result) == 1
    assert "crosswind" in result[0].text.lower()


def test_wh_words_not_mistaken_for_proper_noun_entities():
    d = MultiIntentDecomposer()
    text = "What is the crosswind limit, and do we need de-icing?"
    result = d.decompose(text)
    for sq in result:
        assert "(What)" not in sq.text
        assert "(Do)" not in sq.text
