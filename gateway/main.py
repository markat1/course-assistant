from fastapi import FastAPI, HTTPException


from gateway.lifespan import lifespan
from gateway.models.chat_request import ChatRequest
    
app = FastAPI(title="Course Assistant Gateway", lifespan=lifespan)

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify that the gateway is running.

    Returns:
        dict: A dictionary indicating the health status of the gateway.
    """
    return {"status": "ok", "service": "gateway"}

@app.post("/v1/chat/completions")
async def chat_completions(chat_request: ChatRequest):
    """
    Endpoint for handling chat completions.

    Args:
        chat_request (ChatRequest): The request containing the model and messages.

    Returns:
        dict: A dictionary containing the chat completion response.
    """
    # Placeholder implementation - replace with actual chat completion logic
    raise HTTPException(status_code=501, detail="Chat completion logic not implemented yet.")