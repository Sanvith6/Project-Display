import os
from openai import OpenAI
from config import JINA_MODEL_ID, JINA_BASE_URL, JINA_API_KEY

def _format_parsed_score(parsed):
    if not parsed:
        return "Unknown score"
    t1 = parsed.get("team1_name")
    s1 = parsed.get("team1_score", {}) or {}
    t2 = parsed.get("team2_name")
    s2 = parsed.get("team2_score", {}) or {}

    def fmt(team, score):
        if not team:
            return None
        runs = score.get("runs")
        w    = score.get("wickets")
        o    = score.get("overs")
        if runs is None and w is None and o is None:
            return team
        s = f"{team}: "
        if runs is not None:
            s += str(runs)
            if w is not None:
                s += f"/{w}"
            elif w is not None:
                s += f"-{w}"
        if o:
            s += f" in {o} overs"
        return s

    parts = []
    p1 = fmt(t1, s1)
    p2 = fmt(t2, s2)
    if p1:
        parts.append(p1)
    if p2:
        parts.append(p2)
    return " | ".join(parts) if parts else "Unknown score"


def build_commentary_prompt_from_timeline(timeline_events):
    """
    Build a prompt describing the entire innings as a time-ordered series of events.
    Continuous commentary – no BALL 1 / BALL 2 labels.
    """
    lines = []

    lines.append(
        "You are given a time-ordered sequence of events from a cricket innings.\n"
        "Each event corresponds to a sampled frame and includes:\n"
        "- SCOREBOARD OCR (parsed runs/wickets/overs for the batting team).\n"
        "- YOLO detections describing the visual scene on the field.\n"
        "- An optional video-model label that gives a vague flavour of the shot type.\n\n"
        "HARD RULES:\n"
        "1. Treat the SCOREBOARD OCR as the ONLY source of truth for runs, wickets, and overs.\n"
        "2. Completely IGNORE all classifier outputs such as shot predictions, umpire gestures, and runout predictions. "
        "   Assume they are 'NOT AVAILABLE' and never use them to decide outcomes.\n"
        "3. PROCESSED VISUALS: High-confidence YOLO detections (>80%) are provided. You MAY use these to colour the visual scene "
        "   (e.g., 'Player detected', 'Bat detected'). Use them to add flavour but do NOT over-trust them for subtle events.\n"
        "4. When scorecard information is missing or unreliable at a time, do NOT invent exact scores. "
        "   Use safe, neutral commentary (for example, a dot ball or a generic defensive shot).\n"
        "5. When YOLO detections are empty, still write realistic but conservative commentary with no dramatic events.\n"
        "6. Do NOT mention OCR, detectors, models, probabilities, JSON, or any technical details.\n\n"
        "Your goal is to write continuous, live-style commentary over the innings, in chronological order, "
        "without labelling commentary as 'Ball 1', 'Ball 2', etc.\n"
    )

    for i in range(len(timeline_events)):
        e = timeline_events[i]
        
        t = e["time_sec"]
        # Calculate duration
        if i < len(timeline_events) - 1:
            next_t = timeline_events[i+1]["time_sec"]
            duration = next_t - t
        else:
            duration = 5.0 # Default for last event

        # Estimate words (approx 2.5 words/sec -> 150wpm)
        # We give a range to allow creativity but prevent rambling
        max_words = int(duration * 2.5) 
        if max_words < 5: max_words = 5

        score_parsed = e.get("score_parsed")
        score_str = _format_parsed_score(score_parsed) if score_parsed else "None"
        yolo_objs = e.get("models", {}).get("yolo_detections", []) or []
        clip_ctx = e.get("clip_context") or {}
        vc = clip_ctx.get("video_class") or {}
        vc_label = vc.get("label", "Unknown")
        vc_conf = vc.get("confidence", 0.0)

        # Filter high-confidence YOLO
        valid_yolo = [y for y in yolo_objs if y.get("conf", 0) > 0.8]
        
        yolo_desc = "None"
        if valid_yolo:
            items = [f"{obj.get('class_name')} ({obj.get('conf'):.2f})" for obj in valid_yolo]
            yolo_desc = ", ".join(items)

        lines.append(f"\nEVENT {i+1}:")
        lines.append(f" - Time (s): {t:.1f} (Next event in {duration:.1f}s -> Aim for approx {max_words} words)")
        lines.append(f" - Scoreboard snapshot: {score_str}")
        lines.append(f" - Video model label (flavour only): {vc_label} (confidence={vc_conf:.2f})")
        lines.append(f" - High-Confidence Visuals (>0.8): {yolo_desc}")

    lines.append(
        "\nTASK:\n"
        "1. Analyze the SCOREBOARD changes to track the flow of the game.\n"
        "2. Write a PASSIONATE, CONTINUOUS commentary trace. Don't just list events.\n"
        "3. **FILL THE SILENCE**: If the score doesn't change for a while, TALK about the tenseness, the fielding, or the bowler's strategy. "
        "   Do not let the commentary die down.\n"
        "4. **REACT**: If a wicket falls or a boundary is hit, I want to see capital letters and exclamation marks! "
        "   'AND HE'S BOWLED HIM! UNBELIEVABLE SCENES!'\n"
        "5. **PACING (CRITICAL):** Pay close attention to the 'Aim for X words' hint. "
        "   - If the duration is short (2s), write a quick burst. "
        "   - If the duration is long (10s), elaborate on the atmosphere.\n"
        "6. Keep it chronological but fluid. No 'Event 1', 'Event 2' labels."
    )

    return "\n".join(lines)


def call_llm(prompt: str) -> str:
    api_key = JINA_API_KEY
    if not api_key:
        print("[ERROR] JINA_API_KEY environment variable is not set.")
        return "[LLM ERROR] No Jina API key found. Skipping commentary generation."

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=JINA_BASE_URL,
        )
    except Exception as e:
        print(f"[ERROR] Failed to create Jina OpenAI client: {e}")
        return "[LLM ERROR] Failed to create Jina client."

    system_msg = (
        "You are a HIGH-ENERGY, EMOTIONAL professional live cricket commentator (like Danny Morrison or Ravi Shastri).\n"
        "You receive time-ordered structured data from a cricket innings. "
        "The only reliable source of runs, wickets, and overs is the SCOREBOARD OCR. "
        "YOLO detections and any video-model labels are used only for visual description.\n\n"
        "HARD CONSTRAINTS:\n"
        "- Completely IGNORE all shot-classification, umpire-gesture, and runout classifier outputs. "
        "  Act as if those predictions are not available.\n"
        "- Use SCOREBOARD OCR to determine how the score changes over time. "
        "  If the score is missing or unchanged for a period, do not invent exact numbers; "
        "  instead, use safe, neutral commentary.\n"
        "- PROCESSED VISUALS: Use them ONLY for flavor. \n"
        "  BAD: 'The umpire is visible.' (BORING!)\n"
        "  GOOD: 'The umpire steps in to calm things down.' or 'A nervous look from the umpire.'\n"
        "  If a visual doesn't add drama, IGNORE IT.\n"
        "- Never mention OCR, detectors, models, probabilities, JSON, or any technical terms.\n\n"
        "OUTPUT STYLE & EMOTION:\n"
        "- **HIGH ENERGY**: You must sound excited! The match is alive!\n"
        "- **SHOUTING**: Use UPPERCASE for big moments! 'THAT IS HUGE!', 'WHAT A SHOT!', 'OUT! HE IS GONE!'\n"
        "- **NO ROBOTIC LISTS**: Do not say 'I see a player'. Say 'Smith takes his stance' or 'The fielder is sprinting!'.\n"
        "- **FILL GAPS**: If the score is stuck, talk about the tension, the crowd, the weather, the strategy. \n"
        "  'The tension is palpable here at the stadium...'\n"
        "- **CONTINUOUS FLOW**: Do NOT label 'Ball 1', 'Ball 2'. Write a flowing narrative stream. "
        "  Connect events smoothly.\n\n"
        "CRITICAL: YOUR OUTPUT MUST BE UNDER 9000 CHARACTERS. "
        "Summarize if needed, but KEEP THE ENERGY HIGH."
    )

    try:
        print(f"[INFO] Calling Jina model: {JINA_MODEL_ID}")
        chat = client.chat.completions.create(
            model=JINA_MODEL_ID,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=4000,
        )

        if chat.choices and chat.choices[0].message and chat.choices[0].message.content:
            return chat.choices[0].message.content

        return "[LLM WARNING] Jina call succeeded but no text was returned."
    except Exception as e:
        msg = str(e)
        if "524" in msg or "A timeout occurred" in msg or "timed out" in msg.lower():
            print("[ERROR] Jina DeepSearch request timed out (Cloudflare 524).")
            return "[LLM ERROR] Jina DeepSearch timed out while generating commentary. Try again or shorten the input."
        print(f"[ERROR] Jina LLM call failed: {e}")
        return "[LLM ERROR] Jina call failed. See console for details."

def summarize_text(text: str, max_chars: int = 9500) -> str:
    """
    Compresses text to fit within a character limit using the LLM.
    """
    if len(text) <= max_chars:
        return text
        
    print(f"[LLM] Summarizing text (len={len(text)}) to under {max_chars} chars...")
    
    prompt = (
        f"The following cricket commentary is too long ({len(text)} chars). "
        f"Compress it to under {max_chars} characters while preserving the flow and key events. "
        "Keep the tone exciting and professional. Do not remove the end."
        "\n\nTEXT:\n" + text[:60000] # Safety cap on input
    )
    
    api_key = JINA_API_KEY
    if not api_key: 
        return text[:max_chars] # Hard chop fallback
        
    try:
        client = OpenAI(api_key=api_key, base_url=JINA_BASE_URL)
        chat = client.chat.completions.create(
            model=JINA_MODEL_ID,
            messages=[
                {"role": "system", "content": "You are an expert editor. Compress the text as requested."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=4000, # Large output window for the summary
        )
        if chat.choices and chat.choices[0].message and chat.choices[0].message.content:
            return chat.choices[0].message.content
        return text # Return original if empty
    except Exception as e:
        print(f"[LLM ERROR] Summarization failed: {e}")
        return text # Return original on error
