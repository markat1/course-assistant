from agents import Agent, Model

def create_tutor(model: Model) -> Agent:
    """
    Create a tutor agent based on the specified mode.

    Args:
        mode (Model): The model to be used for the tutor agent.

    Returns:
        Agent: An instance of the tutor agent.
    """
    return Agent(
        name="Course Tutor",
        model=model,
        instructions=(
            "Help students understand Abi's inference engineering course. "
            "Explain one concept at a time with a concrete example. "
            "Use the supplied course material to support your explanation. "
            "If evidence is missing, say so. "
            "Never invent sources or video timestamps. "
            "End with one short question to check understanding."
        ),

    )