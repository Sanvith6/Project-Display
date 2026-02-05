import asyncio
import os
from pathlib import Path
import edge_tts
import subprocess
from config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID

# Default voice for Edge TTS: English (UK) - Ryan
EDGE_VOICE = "en-GB-RyanNeural"

# Try importing ElevenLabs SDK
try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import save
    HAS_ELEVENLABS = True
except ImportError:
    print("[TTS] ElevenLabs package not found. Using Edge TTS only.")
    HAS_ELEVENLABS = False

def _log_debug(msg):
    try:
        with open("tts_debug.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except:
        pass

async def _generate_audio_edge(text: str, output_path: str, voice: str) -> bool:
    """
    Async function to interact with edge-tts.
    """
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        return True
    except Exception as e:
        print(f"[TTS ERROR] edge-tts generation failed: {e}")
        return False

def _generate_audio_elevenlabs(text: str, output_path: str) -> bool:
    """
    Generates audio using ElevenLabs API.
    """
    if not HAS_ELEVENLABS:
        return False
        
    try:
        if not ELEVENLABS_API_KEY:
             print("[TTS] Missing ELEVENLABS_API_KEY.")
             return False

        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        
        # Use configured voice or default to a safe one if missing (though ID is preferred)
        voice_id = ELEVENLABS_VOICE_ID if ELEVENLABS_VOICE_ID else "JBFqnCBsd6RMkjVDRZzb" # Example ID (George)
        
        print(f"[TTS] Generating (ElevenLabs) voice={voice_id} model={ELEVENLABS_MODEL_ID}...")
        
        audio = client.generate(
            text=text,
            voice=voice_id,
            model=ELEVENLABS_MODEL_ID
        )
        
        path_obj = Path(output_path)
        # client.generate returns a generator in some versions, or bytes. 
        # The 'save' helper handles it, or we write bytes.
        save(audio, str(path_obj))
        
        return True
    except Exception as e:
        print(f"[TTS ERROR] ElevenLabs generation failed: {e}")
        return False

def synthesize_commentary_audio(commentary_text: str, output_path: Path) -> bool:
    """
    Generates audio from text. 
    Prioritizes ElevenLabs if configured, otherwise falls back to Microsoft Edge TTS.
    """
    if not commentary_text:
        print("[TTS] No commentary text provided.")
        return False

    output_str = str(output_path.resolve())

    # 1. Try ElevenLabs
    if HAS_ELEVENLABS and ELEVENLABS_API_KEY:
        print("[TTS] Attempting ElevenLabs generation...")
        if _generate_audio_elevenlabs(commentary_text, output_str):
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"[TTS] Success (ElevenLabs)! Saved to {output_path}")
                return True
            else:
                print("[TTS] ElevenLabs file empty. Falling back...")

    # 2. Fallback to Edge TTS (via Subprocess for Safety)
    print(f"[TTS] Using Edge TTS fallback (Voice: {EDGE_VOICE})...")
    
    try:
        # Write text to a temporary file to avoid command line length limits on Windows
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt") as tmp:
            tmp.write(commentary_text)
            tmp_path = tmp.name
            
        # Use subprocess to call edge-tts CLI directly. 
        cmd = [
            "edge-tts",
            "--voice", EDGE_VOICE,
            "--file", tmp_path,
            "--write-media", output_str
        ]
        
        # Run process
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Cleanup temp file
        try:
            os.remove(tmp_path)
        except:
            pass
        
        if result.returncode != 0:
            msg = f"[TTS ERROR] edge-tts process failed: {result.stderr}"
            print(msg)
            _log_debug(msg)
            return False
            
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[TTS] Success (Edge TTS)! Saved to {output_path}")
            return True
        else:
            print("[TTS] File not created or empty (Edge TTS).")
            return False

    except Exception as e:
        msg = f"[TTS ERROR] Wrapper failed: {e}"
        print(msg)
        _log_debug(msg)
        return False
