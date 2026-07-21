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

