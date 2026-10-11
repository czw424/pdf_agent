from app.rag.memory import SessionMemory


def test_add_turn_and_get_history():
    mem = SessionMemory(max_turns=5)
    mem.add_turn("s1", "问题1", "回答1")

    history = mem.get_history("s1")

    assert len(history) == 2
    assert history[0].content == "问题1"
    assert history[1].content == "回答1"


def test_history_isolated_by_session():
    mem = SessionMemory(max_turns=5)
    mem.add_turn("s1", "q1", "a1")
    mem.add_turn("s2", "q2", "a2")

    assert len(mem.get_history("s1")) == 2
    assert len(mem.get_history("s2")) == 2
    assert mem.get_history("s1")[0].content == "q1"


def test_history_truncated_to_max_turns():
    mem = SessionMemory(max_turns=2)
    for i in range(5):
        mem.add_turn("s1", f"q{i}", f"a{i}")

    history = mem.get_history("s1")

    assert len(history) == 4  # 2轮 * 2条消息
    assert history[0].content == "q3"


def test_clear_session():
    mem = SessionMemory(max_turns=5)
    mem.add_turn("s1", "q", "a")
    mem.clear("s1")

    assert mem.get_history("s1") == []
