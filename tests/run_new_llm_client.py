from src.llm.llm_client import LLMClient
from src.llm.output_schemas import AgentAnswer
from src.llm.prompt_templates import PromptTemplates


client = LLMClient()

alice_prompts = PromptTemplates.worker(
    agent_id="alice_agent",
    role_description="Represent Alice and use only Alice's authorised information.",
)

prompt = alice_prompts.build_answer_prompt(
    task="What is Alice's favourite book?",
    accessible_memory_context="Alice's favourite book is Dune.",
)

result = client.invoke_structured(
    prompt,
    AgentAnswer,
)

print(result)

def test_agent_answer_clears_missing_information_for_answered_status(
) -> None:
    answer = AgentAnswer.model_validate(
        {
            "schema_version": "2.0",
            "status": "answered",
            "answer": (
                "The Northstar petition "
                "was approved."
            ),
            "reasoning": (
                "The authorised memory "
                "supports the answer."
            ),
            "confidence": 0.9,
            "missing_information": [
                "Additional background details"
            ],
            "used_memory_ids": [
                "mem_001"
            ],
            "supporting_source_ids": [],
            "contributing_agent_ids": [
                "it_miles_khan"
            ],
        }
    )

    assert answer.status == "answered"
    assert answer.answer == (
        "The Northstar petition was approved."
    )
    assert answer.missing_information == []

