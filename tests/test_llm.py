from django_minimal_rag.llm import FakeLLM


def test_fake_llm_answers_with_the_prompt_it_was_given() -> None:
    llm = FakeLLM()

    first = llm.complete("What is a vector?", system="Answer briefly.")
    second = llm.complete("Who wrote the docs?", system="Cite your sources.")

    assert first == "What is a vector?"
    assert second == "Who wrote the docs?"
