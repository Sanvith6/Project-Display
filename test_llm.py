import json
from pathlib import Path
from llm import build_commentary_prompt_from_timeline, call_llm

def main():
    timeline_path = Path("timeline_for_llm.json")
    if not timeline_path.exists():
        print("No timeline json found.")
        return

    print("Loading timeline...")
    with open(timeline_path, "r") as f:
        timeline = json.load(f)

    # Slice strictly to first 50 events to save time/tokens for this test
    # (The user wants to see the STYLE, not the whole match right now)
    print(f"Building prompt from first 50 events (total {len(timeline)})...")
    mini_timeline = timeline[:50]
    
    prompt = build_commentary_prompt_from_timeline(mini_timeline)
    
    print("\n--- GENERATING COMMENTARY WITH NEW PROMPT (Danny Morrison Style) ---\n")
    commentary = call_llm(prompt)
    
    print("\n\n===== RESULT =====\n")
    print(commentary)
    
    # Save to a new test file
    with open("commentary_test_new.txt", "w", encoding="utf-8") as f:
        f.write(commentary)

if __name__ == "__main__":
    main()
