from app.models.course_passage import CoursePassage

COURSE_PASSAGES: tuple[CoursePassage, ...] = (
    CoursePassage(
        id="kv-capacity",
        title="KV memory and concurrency capacity",
        content=(
            "Estimate concurrent sequences by dividing available KV memory "
            "by KV bytes per token times sequence length. Available memory "
            "must account for model weights and activations.  Compare the "
            "configured maximum length with actual application lengths, "
            "then check the prediction against measurements."
        ),
        source="Final project brief - Part 1: Capacity on paper",
    ),
    CoursePassage(
        id="queue-boundaries",
        title="Gateway and engine queue ownership",
        content=(
            "The gateway owns admission, worker placement and its request "
            "queue. After dispatch, the engine owns waiting, running and "
            "preemption. Observe both queues separately and leave continuous "
            "batching and token scheduling to the engine."
        ),
        source="Final project brief - Part 5: Queue and engine boundaries"
    ),
    CoursePassage(
        id="warmup",
        title="Worker warmup and KV transfer",
        content=(
            "Loaded model weights alone do not establish readiness. "
            "Measure the first agent step before and after warmup. "
            "When prefill and decode use different workers, the destination "
            "needs the source KV state or must recompute the prompt. "
            "Using the same worker requires no cross-worker transfer."
        ),
        source="Final project brief - Part 6: Hop and warmup",
    ),
)