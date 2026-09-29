from agents import Agent, Model
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

def create_router(model: Model, tutor: Agent) -> Agent:
    """
    Create a router agent that directs user queries to the appropriate agent.

    Args:
        model (Model): The model to be used for the router agent.
        tutor (Agent): The tutor agent to which queries may be routed.

    Returns:
        Agent: An instance of the router agent.
    """
    return Agent(
        name="Course Router",
        model=model,
        instructions=prompt_with_handoff_instructions(
        "You route students of Abi's inference engineering course. "
        "For any question about course concepts, immediately call "
        "transfer_to_course_tutor. Do not answer it yourself and do not "
        "ask the student for permission to transfer. "
        "For requests outside the course, explain your scope briefly."
        ),
        handoffs=[tutor],
    )