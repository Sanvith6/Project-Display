from pathlib import Path
from tts import synthesize_commentary_audio

def test_tts():
    print("Testing edge-tts integration...")
    text = "Hello! This is a test of the new free neural Text-to-Speech engine. Welcome to the cricket match."
    output_file = Path("tts_edge_test.mp3")
    
    if output_file.exists():
        output_file.unlink()
        
    success = synthesize_commentary_audio(text, output_file)
    
    if success:
        print(f"PASS: Audio generated at {output_file.resolve()}")
    else:
        print("FAIL: Audio generation failed.")

if __name__ == "__main__":
    test_tts()
