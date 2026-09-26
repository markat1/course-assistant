from agents import Agent, Model

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
        instructions=(
         "Identify what help the student needs with Abi's course. "
         "For explanations of course concepts, hand off to the Tutor. "
         "For requests outside the course, explain your scope briefly."
        ),
        handoffs=[tutor],
    )