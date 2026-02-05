from pathlib import Path
import os
from tts import synthesize_commentary_audio
from config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID

def mask_key(k):
    if not k: return "None"
    return k[:4] + "*" * (len(k)-8) + k[-4:] if len(k) > 8 else "****"

def test_tts():
    print(f"ELEVENLABS_API_KEY: {mask_key(ELEVENLABS_API_KEY)}")
    print(f"ELEVENLABS_VOICE_ID: {ELEVENLABS_VOICE_ID}")
    
    output = Path("test_tts_output.mp3")
    if output.exists():
        output.unlink()
        
    text = "This is a test of the ElevenLabs implementation for cricket commentary."
    print(f"Generating audio for: '{text}'")
    
    success = synthesize_commentary_audio(text, output)
    
    if success and output.exists() and output.stat().st_size > 0:
        print(f"SUCCESS: Audio generated at {output.resolve()} (Size: {output.stat().st_size} bytes)")
    else:
        print("FAILURE: Audio generation failed.")

if __name__ == "__main__":
    test_tts()
