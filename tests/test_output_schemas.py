from src.llm import AgentAnswer
import pytest

def test_agent_answer_normalises_whitespace() -> None:
    result = AgentAnswer(
        answer=(
            "  Alexander McQueen,\n"
            "Coco Chanel  "
        ),
        reasoning="Supported by accessible memory.",
        confidence=0.9,
        used_memory_ids=[],
        supporting_source_ids=[],
        contributing_agent_ids=[],
    )

    assert (
        result.answer
        == "Alexander McQueen, Coco Chanel"
    )

    def test_agent_answer_rejects_empty_answer() -> None:
        with pytest.raises(
                ValueError,
                match="answer cannot be empty",
        ):
            AgentAnswer(
                answer="   ",
                reasoning="No answer.",
                confidence=0.0,
                used_memory_ids=[],
                supporting_source_ids=[],
                contributing_agent_ids=[],
            )


def test_agent_answer_rejects_empty_answer() -> None:
    with pytest.raises(
        ValueError,
        match="answer cannot be empty",
    ):
        AgentAnswer(
            answer="   ",
            reasoning="No answer.",
            confidence=0.0,
            used_memory_ids=[],
            supporting_source_ids=[],
            contributing_agent_ids=[],
        )