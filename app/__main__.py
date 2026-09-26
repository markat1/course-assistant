from app.agents.tutor import create_tutor
from app.llm import create_model
from app.agents.router import create_router

def main() -> None:
    """
    Main function to create a tutor agent and an OpenAI chat completions model.
    """

    model = create_model(model_name="connection-test", base_url="http://127.0.0.1:8780/v1")
    tutor_agent = create_tutor(model=model)
    router_agent = create_router(model=model, tutor=tutor_agent)

    print(f"Tutor Agent created: {tutor_agent.name}")
    print(f"Router Agent created: {router_agent.name}")

if __name__ == "__main__":
    main()