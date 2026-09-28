import uvicorn

def main() -> None:
    uvicorn.run(
        "app.main:create_runtime_app",
        factory=True,
        host="127.0.0.1",
        port=8781,
    )

if __name__=="__main__":
    main()