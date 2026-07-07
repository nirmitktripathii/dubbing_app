import os
import json
import time
import ssl
from google import genai
from google.genai import types

def generate_content_with_retry(client, **kwargs):
    """
    Helper function to request Gemini API with retries on transient errors (503/429).
    """
    for attempt in range(5):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as e:
            err_msg = str(e)
            is_transient = any(code in err_msg for code in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded"])
            if is_transient and attempt < 4:
                wait_time = (attempt + 1) * 3
                print(f"Transient Gemini API error: {err_msg}. Retrying in {wait_time}s (attempt {attempt + 1}/5)...")
                time.sleep(wait_time)
            else:
                raise e

def translate_segments(segments: list, target_language: str, api_key: str):
    """
    Translates a list of transcribed segments using Google Gemini API.
    
    Args:
        segments (list): A list of dicts with 'start', 'end', and 'text'.
        target_language (str): Target language for translation (e.g., "Hindi").
        api_key (str): Gemini API key.
        
    Returns:
        list: A new list of dicts with translated 'text', preserving 'start' and 'end'.
    """
    if not api_key:
        raise ValueError("Gemini API key is required for translation.")
        
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            client_args={"verify": ssl_context},
            async_client_args={"verify": ssl_context}
        )
    )
    
    texts_to_translate = [
        {"text": seg["text"], "duration_seconds": round(seg["end"] - seg["start"], 1)}
        for seg in segments
    ]
    
    prompt = (
        f"You are an expert video dubbing translator, localizer, and native speaker of {target_language}. "
        f"Translate the following array of English text segments (with their duration limits in seconds) into {target_language}.\n\n"
        f"CRITICAL INSTRUCTIONS:\n"
        f"1. Make the translation sound completely natural and conversational in {target_language}.\n"
        f"2. Ensure the script and spelling (e.g., Devanagari for Hindi) are 100% grammatically correct and contextually appropriate.\n"
        f"3. CRITICAL: The translated segments MUST be concise enough to be naturally spoken within their specified duration limit (duration_seconds). "
        f"If a direct translation is too long, paraphrase or use shorter synonyms so it fits the time limit.\n"
        f"4. Return ONLY a valid JSON array of strings containing the translated texts in the exact same order.\n"
        f"5. Do not include any Markdown formatting like ```json, just the raw JSON array.\n\n"
        f"Texts to translate: {json.dumps(texts_to_translate, ensure_ascii=False)}"
    )
    
    try:
        response = generate_content_with_retry(
            client,
            model='gemini-3.1-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
            )
        )
        content = response.text.strip()
        
        # Fallback stripping if the model still outputs markdown blocks
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        translated_texts = json.loads(content.strip())
        
        if len(translated_texts) != len(segments):
            raise ValueError("Translation length mismatch.")
            
        translated_segments = []
        for i, seg in enumerate(segments):
            translated_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": translated_texts[i]
            })
            
        return translated_segments
    except Exception as e:
        print(f"Error in batch translation: {e}. Falling back to sequential.")
        return _translate_sequential(client, segments, target_language)

def _translate_sequential(client, segments, target_language):
    translated = []
    for seg in segments:
        duration = round(seg["end"] - seg["start"], 1)
        prompt = (
            f"Translate the following English text to {target_language}. "
            f"CRITICAL: Keep the translation concise so it can be naturally spoken within {duration} seconds. "
            f"Return ONLY the direct translation, nothing else.\n\n"
            f"Text: {seg['text']}"
        )
        response = generate_content_with_retry(
            client,
            model='gemini-3.1-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
            )
        )
        t_text = response.text.strip()
        translated.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": t_text
        })
    return translated
