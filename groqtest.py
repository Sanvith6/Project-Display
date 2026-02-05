import os
from groq import Groq

def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in the environment")

    print("GROQ_API_KEY is set ✅")

    # Create Groq client
    client = Groq(api_key=api_key)

    # Choose a Groq model (common ones below)
    # You can change this if you know a specific one:
    #   - "llama-3.1-8b-instant"
    #   - "llama-3.1-70b-versatile"
    model_name = "llama-3.1-8b-instant"
    print("Using model:", model_name)

    # Simple test prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert, lively cricket commentator. "
                "Answer briefly."
            ),
        },
        {
            "role": "user",
            "content": "Give one exciting line of commentary for a bowler taking a wicket.",
        },
    ]

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.8,
            max_tokens=80,
        )
    except Exception as e:
        print("[ERROR] Groq API call failed:", e)
        return

    # Extract the text safely
    try:
        text = resp.choices[0].message.content
    except Exception:
        text = str(resp)

    print("\n=== GROQ LLM RESPONSE ===\n")
    print(text)


if __name__ == "__main__":
    main()
