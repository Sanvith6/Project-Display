
import os
import requests
from dotenv import load_dotenv

# Force reload to get latest key from .env
load_dotenv(override=True)

JINA_API_KEY = os.getenv("JINA_API_KEY")
if not JINA_API_KEY:
    print("Error: JINA_API_KEY not found in environment!")
    exit(1)

print(f"Testing Key: {JINA_API_KEY[:15]}...")

# Endpoint as defined in config.py
url = "https://deepsearch.jina.ai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json"
}
data = {
    "model": "jina-deepsearch-v1",
    "messages": [{"role": "user", "content": "Hello, return the word 'Connected'."}],
    "stream": False
}

print(f"Sending request to {url}...")
try:
    resp = requests.post(url, headers=headers, json=data, timeout=30)
    print(f"Status Code: {resp.status_code}")
    
    if resp.status_code == 200:
        print("\n✅ SUCCESS: Jina API is working!")
        print("Response:", resp.json())
    elif resp.status_code == 402:
        print("\n❌ FAILED: Payment Required (Insufficient Balance).")
        print(resp.text)
    elif resp.status_code == 401:
        print("\n❌ FAILED: Unauthorized (Invalid Key).")
        print(resp.text)
    else:
        print(f"\n❌ FAILED: HTTP {resp.status_code}")
        print(resp.text)

except Exception as e:
    print(f"\n❌ FAILED: Network Error: {e}")
