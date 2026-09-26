from fastapi import FastAPI

app = FastAPI(title="Course Assistant Gateway")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify that the gateway is running.

    Returns:
        dict: A dictionary indicating the health status of the gateway.
    """
    return {"status": "ok", "service": "gateway"}