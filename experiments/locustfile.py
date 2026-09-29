import random

from locust import HttpUser, between, task

MODEL = "Qwen/Qwen3-8B"

TUTOR_PREFIX = "You are the Course Tutor. Use the course passages.\n" + (
    "Course passage: A KV cache stores attention keys and values for every "
    "previous token so decoding does not recompute them. "
) * 20
AGENT_PREFIX = "You are the Course Router with tools lookup_course and transfer_to_course_tutor.\n" + (
    "Tool schema: lookup_course(query: string) returns course passages with sources. "
) * 200
DOCUMENT = (
    "Lecture transcript: prefill processes the prompt in parallel while decode "
    "generates one token at a time and is limited by memory bandwidth. "
)
QUESTIONS = [
    "Why does KV cache memory limit concurrency?",
    "What does prefix caching reuse?",
    "When is prefill the bottleneck?",
    "Why must a replica be warmed before traffic?",
]


def chat(user: HttpUser, name: str, system: str, question: str, max_tokens: int) -> None:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
    }
    with user.client.post("/v1/chat/completions", json=payload, name=name, catch_response=True) as response:
        if response.status_code == 200:
            response.success()
        else:
            response.failure(f"HTTP {response.status_code}")


class InteractiveUser(HttpUser):
    weight = 7
    wait_time = between(1, 3)

    @task
    def ask_tutor(self) -> None:
        chat(self, "interactive", TUTOR_PREFIX, random.choice(QUESTIONS), random.randint(64, 256))


class BatchUser(HttpUser):
    weight = 2
    wait_time = between(2, 5)

    @task
    def summarise_document(self) -> None:
        document = f"Document {random.randint(0, 10**9)}\n" + DOCUMENT * random.randint(80, 160)
        chat(self, "batch", "Summarise the document for revision notes.\n" + document, "Summarise it.", random.randint(256, 512))


class AgentUser(HttpUser):
    weight = 1
    wait_time = between(0.5, 1.5)

    @task
    def agent_step(self) -> None:
        chat(self, "agent", AGENT_PREFIX, "Plan the next tool call for: " + random.choice(QUESTIONS), 128)