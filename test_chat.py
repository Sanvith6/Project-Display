import asyncio
from cricket_server import get_match_context_at_time
from pipeline10 import call_llm

async def test_chat_logic():
    print("--- Testing Context Retrieval ---")
    # Simulate time=10s
    ctx = get_match_context_at_time(10.0)
    print(f"Context at 10s: {ctx[:100]}...") # Print first 100 chars

    print("\n--- Testing LLM Call ---")
    prompt = (
        f"CONTEXT: {ctx}\n"
        "USER QUESTION: What is the current score?\n"
        "Answer briefly."
    )
    # We mock the call if API key is missing, but here we try real call
    # Note: If JINA_API_KEY is not set in env, this prints an error but won't crash
    try:
        ans = call_llm(prompt)
        print(f"LLM Answer: {ans}")
    except Exception as e:
        print(f"LLM Call failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_chat_logic())
