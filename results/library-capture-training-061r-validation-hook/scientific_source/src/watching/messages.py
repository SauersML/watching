"""Explicit chat histories with an unchanged legacy single-user rendering path."""

from collections import defaultdict

PREFIX_GROUPS = {
    "single_user": ("user",),
    "system_user": ("system", "user"),
    "multiturn": ("user", "assistant", "user"),
    "system_multiturn": ("system", "user", "assistant", "user"),
}


def record_messages(record):
    if "messages" not in record:
        return [{"role": "user", "content": record["text"]}]
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty explicit list")
    expected = "user"
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Each message must contain only role and content")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError("Message content must be a nonempty string")
        role = message["role"]
        if index == 0 and role == "system":
            continue
        if role != expected:
            raise ValueError("Messages must alternate user/assistant after an optional system")
        expected = "assistant" if role == "user" else "user"
    if messages[-1]["role"] != "user" or messages[-1]["content"] != record["text"]:
        raise ValueError("The final user message must equal the record's text")
    return [dict(message) for message in messages]


def render_record(tokenizer, record):
    return tokenizer.apply_chat_template(
        record_messages(record),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def validate_prefix_groups(records):
    """Opt in through base_context_id; never accept incomplete repeated-measure sets."""
    grouped = any("base_context_id" in row for row in records)
    for row in records:
        if "messages" in row:
            record_messages(row)
    if not grouped:
        return False
    if any("base_context_id" not in row or "messages" not in row for row in records):
        raise ValueError("Grouped message data cannot mix grouped and ungrouped records")
    variants = set()
    bases = defaultdict(list)
    for row in records:
        identity = row.get("context_id")
        if not identity or identity in variants:
            raise ValueError("Message variants require unique context_id values")
        variants.add(identity)
        bases[row["base_context_id"]].append(row)
    contexts = set()
    for base_rows in bases.values():
        if len(base_rows) != 4 or {r.get("group") for r in base_rows} != set(PREFIX_GROUPS):
            raise ValueError("Each base context requires all four prefix groups exactly once")
        first = base_rows[0]
        shared = (
            "context",
            "text",
            "task",
            "answer_negative",
            "answer_positive",
            "question_family",
            "prefix_family",
        )
        if first["context"] in contexts:
            raise ValueError("One context cannot masquerade as multiple independent base cases")
        contexts.add(first["context"])
        by_group = {r["group"]: r for r in base_rows}
        for row in base_rows:
            if any(row.get(key) != first.get(key) for key in shared):
                raise ValueError("Matched layouts must share context, question, task and labels")
            if tuple(m["role"] for m in record_messages(row)) != PREFIX_GROUPS[row["group"]]:
                raise ValueError("Message roles do not match the declared prefix group")
        if by_group["system_user"]["messages"][0] != by_group["system_multiturn"]["messages"][0]:
            raise ValueError("System text must match across the two system layouts")
        if by_group["multiturn"]["messages"] != by_group["system_multiturn"]["messages"][1:]:
            raise ValueError("History must match across the two multiturn layouts")
    return True


def validate_message_splits(data):
    """Check actual base identities/text, rather than only variant IDs, across splits."""
    identities, contexts = set(), set()
    for records in data.values():
        if not validate_prefix_groups(records):
            continue
        current_ids = {r["base_context_id"] for r in records}
        current_contexts = {r["context"] for r in records}
        if identities & current_ids or contexts & current_contexts:
            raise ValueError("Base contexts overlap across message-data splits")
        identities.update(current_ids)
        contexts.update(current_contexts)
