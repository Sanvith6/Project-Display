import re

text_lines = [
    "ENG",
    "0-0",
    "0/20",
    "Toss ENG",
    "/ Smith 0(0)",
    "Duckett 0(0)",
    "WI",
    "Holdor"
]
text_flattened = " ".join(text_lines) # simple join
# emulate ocr.py flattening:
text_ocr_py = re.sub(r"\s+", " ", "\n".join(text_lines)).strip()

print(f"Flattened Text: '{text_ocr_py}'")

# Original Regex from ocr.py
original_pattern = re.compile(
    r"\b(?P<bat_team>[A-Za-z]{2,4})\s+"
    r"(?P<runs>\d+)\s*[-/]\s*(?P<wickets>\d+)\s+"
    r"(?P<phase>[A-Za-z])?\s*"
    r"(?P<overs>\d+\.\d|\d+)\s*/\s*(?P<max_overs>\d+)\s+"
    r"Toss\s+(?P<toss_team>[A-Za-z]{2,4})\s+"
    r"(?P<striker>[A-Za-z]+)\s+(?P<striker_runs>\d+)\*?\s*\((?P<striker_balls>\d+)\)\s+"
    r"(?P<nonstriker>[A-Za-z]+)\s+(?P<nonstriker_runs>\d+)\s*\((?P<nonstriker_balls>\d+)\)\s+"
    r"(?P<bowl_team>[A-Za-z]{2,4})\s+"
    r"(?P<speed_mph>\d+)\s*mph\s*/\s*(?P<speed_kph>\d+)\s*kph",
    re.IGNORECASE
)

match = original_pattern.search(text_ocr_py)
print("Original Match:", "YES" if match else "NO")


# Proposed Regex
# Changes:
# 1. Allow optional symbols between blocks (e.g. after toss)
# 2. Make speed optional
proposed_pattern = re.compile(
    r"\b(?P<bat_team>[A-Za-z]{2,4})\s+"
    r"(?P<runs>\d+)\s*[-/]\s*(?P<wickets>\d+)\s+"
    r"(?P<phase>[A-Za-z])?\s*"
    r"(?P<overs>\d+\.\d|\d+)\s*/\s*(?P<max_overs>\d+)\s+"
    r"Toss\s+(?P<toss_team>[A-Za-z]{2,4})\s*"
    r"(?:[^A-Za-z0-9\s]\s*)?" # Optional symbol (like /) and space
    r"(?P<striker>[A-Za-z]+)\s+(?P<striker_runs>\d+)\*?\s*\((?P<striker_balls>\d+)\)\s+"
    r"(?P<nonstriker>[A-Za-z]+)\s+(?P<nonstriker_runs>\d+)\s*\((?P<nonstriker_balls>\d+)\)\s+"
    r"(?P<bowl_team>[A-Za-z]{2,4})" # Removed mandatory speed follow-up
    r"(?:\s+(?P<speed_mph>\d+)\s*mph\s*/\s*(?P<speed_kph>\d+)\s*kph)?", # Optional speed
    re.IGNORECASE
)

match_new = proposed_pattern.search(text_ocr_py)
print("New Match:", "YES" if match_new else "NO")
if match_new:
    print(match_new.groupdict())
