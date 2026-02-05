from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Cricket Analysis Server")

# Define Base Directory
BASE_DIR = Path(__file__).parent.resolve()

# File Paths
TIMELINE_FILE = BASE_DIR / "timeline_for_llm.json"
SCORE_FILE = BASE_DIR / "score_data.json"
LOG_FILE = BASE_DIR / "error_trace.log"

@mcp.resource("cricket://timeline")
def get_timeline() -> str:
    """Returns the full match timeline JSON used for LLM commentary generation."""
    if TIMELINE_FILE.exists():
        return TIMELINE_FILE.read_text(encoding="utf-8")
    return "Error: timeline_for_llm.json not found. Run the pipeline first."

@mcp.resource("cricket://scorecard")
def get_scorecard() -> str:
    """Returns the latest scorecard data extracted via OCR."""
    if SCORE_FILE.exists():
        return SCORE_FILE.read_text(encoding="utf-8")
    return "Error: score_data.json not found."

@mcp.resource("cricket://logs")
def get_logs() -> str:
    """Returns the latest error logs for debugging."""
    if LOG_FILE.exists():
        # Return last 2000 chars to avoid overwhelming output
        content = LOG_FILE.read_text(encoding="utf-8")
        return content[-2000:] if len(content) > 2000 else content
    return "No logs found."

@mcp.tool()
def get_match_context_at_time(seconds: float) -> str:
    """
    Returns the match status (score, events, visual description) for a specific video timestamp.
    Use this when a user asks a question about 'what is happening now' or 'at X:XX time'.
    """
    import json
    if not TIMELINE_FILE.exists():
        return "Error: No match data found."
    
    try:
        data = json.loads(TIMELINE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return "Error: Could not parse match data."

    # Find the event closest to the given time (or the most recent one)
    # We assume 'data' is a list of events sorted by 'time_sec'
    closest_event = None
    for event in data:
        t = event.get("time_sec", 0)
        # We look for the event that just happened or is happening (within 5 seconds)
        if t <= seconds:
            closest_event = event
        else:
            # We've gone past the time, so the previous one was the closest "past" event
            break
            
    if closest_event:
        return json.dumps(closest_event, indent=2)
    else:
        return f"No event found before time {seconds}."


@mcp.prompt()
def commentary_system_prompt() -> str:
    """Returns the standard System Prompt for the Cricket Commentator persona."""
    return """You are a professional live cricket commentator.
You receive time-ordered structured data from a cricket innings. 
The only reliable source of runs, wickets, and overs is the SCOREBOARD OCR. 
YOLO detections and any video-model labels are used only for visual description.

HARD CONSTRAINTS:
- Completely IGNORE all shot-classification, umpire-gesture, and runout classifier outputs. 
  Act as if those predictions are not available.
- Use SCOREBOARD OCR to determine how the score changes over time. 
  If the score is missing or unchanged for a period, do not invent exact numbers; 
  instead, use safe, neutral commentary.
- PROCESSED VISUALS: You are provided with High-Confidence YOLO detections (>80%). 
  You MAY use these to enrich the visual scene (e.g. 'The batter is visible', 'Fielders chasing'). 
  Do not use them to decide runs/wickets, but you can trust them for existence of objects.
- Never mention OCR, detectors, models, probabilities, JSON, or any technical terms.
"""

if __name__ == "__main__":
    # Run the server
    mcp.run()
