"""The query prompt: context passages, then the question, in the backbone's chat format."""

NUM_CONTEXT_PASSAGES = 5
CONTEXT_TOKENS_PER_PASSAGE = 200
MAX_QUERY_TOKENS = 256 + NUM_CONTEXT_PASSAGES * CONTEXT_TOKENS_PER_PASSAGE


def context_block(tokenizer, passages: list[str]) -> str:
    """Renders up to 5 passages as "Document i:" sections, within a budget of 1000 tokens.

    Each passage is cut to 200 tokens; the headers and separators count against the budget too.
    """
    budget = NUM_CONTEXT_PASSAGES * CONTEXT_TOKENS_PER_PASSAGE
    separator_tokens = len(tokenizer.encode("\n\n", add_special_tokens=False))
    texts = []
    used = 0
    for passage in passages:
        tokens = tokenizer.encode(passage, add_special_tokens=False)
        if not tokens:
            continue
        header_tokens = len(tokenizer.encode(f"Document {len(texts) + 1}:\n", add_special_tokens=False))
        frame = header_tokens + (separator_tokens if texts else 0)
        remaining = budget - used - frame
        if remaining <= 0:
            break
        tokens = tokens[: min(CONTEXT_TOKENS_PER_PASSAGE, remaining)]
        texts.append(tokenizer.decode(tokens, skip_special_tokens=True))
        used += frame + len(tokens)
        if len(texts) == NUM_CONTEXT_PASSAGES or used >= budget:
            break
    return "\n\n".join(f"Document {i + 1}:\n{text}" for i, text in enumerate(texts))


def query_token_ids(tokenizer, question: str, passages: list[str]) -> list[int]:
    """Token ids of the query prompt, at most 1256 tokens. A question that does not fit is cut from its end."""
    block = context_block(tokenizer, passages)

    def encode(text: str) -> list[int]:
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": f"{block}\n\n{text}"}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return tokenizer.encode(prompt, add_special_tokens=False)

    ids = encode(question)
    if len(ids) <= MAX_QUERY_TOKENS:
        return ids
    low, high = 0, len(question)
    while low < high:
        middle = (low + high + 1) // 2
        if len(encode(question[:middle])) <= MAX_QUERY_TOKENS:
            low = middle
        else:
            high = middle - 1
    return encode(question[:low])[:MAX_QUERY_TOKENS]
