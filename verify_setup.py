
import sys
import os
from pathlib import Path

# Redirect stdout/stderr to a file for debugging
log_file = open("verify_log.txt", "w", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file

print("Verifying imports...")


try:
    from commentator import Commentator
    print("[OK] Commentator module imported successfully.")
except Exception as e:
    print(f"[FAIL] Failed to import Commentator: {e}")

try:
    from app import app
    print("[OK] FastAPI app imported successfully.")
except Exception as e:
    print(f"[FAIL] Failed to import FastAPI app: {e}")

print("Checking files...")
files = [
    "c:/project/Project-Display/commentator.py",
    "c:/project/Project-Display/app.py",
    "c:/project/Project-Display/static/index.html",
    "c:/project/Project-Display/static/style.css",
    "c:/project/Project-Display/static/script.js"
]

for f in files:
    if os.path.exists(f):
        print(f"[OK] Found {f}")
    else:
        print(f"[FAIL] Missing {f}")
