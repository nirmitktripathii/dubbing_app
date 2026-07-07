import os
import asyncio
import edge_tts

async def generate_audio_async(text: str, voice: str, output_path: str):
    """
    Asynchronously generate audio using edge_tts.
    """
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def generate_tts_for_segments(segments: list, language: str, output_dir: str):
    """
    Generates TTS audio files for each translated segment.
    
    Args:
        segments (list): Translated segments with 'start', 'end', and 'text'.
        language (str): Target language (e.g., "Hindi").
        output_dir (str): Directory to save individual segment audio files.
        
    Returns:
        list: The segments list updated with an 'audio_path' key for each segment.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Map target language to edge-tts voice
    voice_map = {
        "Hindi": "hi-IN-MadhurNeural",
        "Spanish": "es-ES-AlvaroNeural",
        "French": "fr-FR-HenriNeural",
        "German": "de-DE-KillianNeural",
        "English": "en-US-ChristopherNeural"
    }
    
    voice = voice_map.get(language, "en-US-ChristopherNeural")
    updated_segments = []
    
    for i, seg in enumerate(segments):
        audio_path = os.path.join(output_dir, f"segment_{i}.mp3")
        
        # Sometimes small chunks may be empty after translation
        if not seg["text"] or seg["text"].strip() == "":
            continue
            
        asyncio.run(generate_audio_async(seg["text"], voice, audio_path))
        
        seg_copy = seg.copy()
        seg_copy["audio_path"] = audio_path
        updated_segments.append(seg_copy)
        
    return updated_segments
