import os
import requests
from pathlib import Path

# Load from environment
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
MODEL_ID = "eleven_multilingual_v2"

OUTPUT_PATH = Path("tts_test_output.mp3")

def test_tts():
    if not ELEVENLABS_API_KEY:
        print("❌ ELEVENLABS_API_KEY not set!")
        return
    if not ELEVENLABS_VOICE_ID:
        print("❌ ELEVENLABS_VOICE_ID not set!")
        return

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    payload = {
        "text": "This is a test voice for cricket commentary.",
        "model_id": MODEL_ID,
    }

    print("🔄 Sending request to ElevenLabs...")
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        with open(OUTPUT_PATH, "wb") as f:
            f.write(response.content)
        print(f"✅ Audio saved: {OUTPUT_PATH}")
        print("🎧 Play the file to check the voice output!")
    else:
        print(f"❌ Request failed: HTTP {response.status_code}")
        print(response.text[:300])


if __name__ == "__main__":
    test_tts()
