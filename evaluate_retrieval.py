test_cases = [
    {
        "question": "When did Lionel Messi join Inter Miami?",
        "expected_text": "2023",
    },
    {
        "question": "What position did Lionel Messi primarily play?",
        "expected_text": "forward",
    },
]


def evaluate_retrieval(retriever):
    hits = 0
    for case in test_cases:
        question = case["question"]
        expected = case["expected_text"]

        docs = retriever.invoke(question)
        found = any(expected in doc.page_content.lower() for doc in docs)

        hits += int(found)

        print(f"\nQuestion: {question}")
        print(f"Hit: {found}")

        for i, doc in enumerate(docs, start=1):
            print(f"Chunk {i}: {doc.page_content}")

    hit_rate = hits / len(test_cases) if test_cases else 0
    print(f"\nHit Rate@3: {hit_rate:.2%}")
