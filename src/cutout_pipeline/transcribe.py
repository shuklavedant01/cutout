"""
Audio Transcription & Translation Module

This module handles:
1. Audio transcription using Deepgram API
2. Language detection
3. Translation to English using Argos Translate
"""
import os
import json
from pathlib import Path
from deepgram import DeepgramClient
import argostranslate.package
import argostranslate.translate

from . import config


def transcribe_audio(audio_file_path, api_key):
    """
    Transcribe audio file using Deepgram API.
    
    Args:
        audio_file_path: Path to the audio file
        api_key: Deepgram API key
        
    Returns:
        dict: Transcription response from Deepgram
    """
    print(f"🎤 Transcribing: {Path(audio_file_path).name}")
    
    deepgram = DeepgramClient(api_key=api_key)
    
    # Load audio file
    with open(audio_file_path, "rb") as f:
        source = {
            "buffer": f.read(),
            "mimetype": f"audio/{Path(audio_file_path).suffix[1:]}"
        }
    
    # Transcribe
    response = deepgram.listen.prerecorded.v("1").transcribe_file(
        source,
        config.TRANSCRIPTION_OPTIONS
    )
    
    return response


def process_transcription(response, audio_filename):
    """
    Process transcription response into structured segments.
    
    Args:
        response: Deepgram API response
        audio_filename: Name of the audio file
        
    Returns:
        tuple: (original_segments, detected_language)
    """
    utterances = response["results"]["utterances"]
    detected_language = response["results"]["channels"][0]["detected_language"]
    
    original_segments = []
    
    print("\n================ TRANSCRIPTION ================\n")
    
    for u in utterances:
        role = "AGENT" if u["speaker"] in config.AGENT_SPEAKERS else "HUMAN"
        
        segment = {
            "role": role,
            "speaker_id": u["speaker"],
            "start": u["start"],
            "end": u["end"],
            "text": u["transcript"]
        }
        
        original_segments.append(segment)
        print(f"{role} (Speaker {u['speaker']}): {u['transcript']}")
    
    print(f"\n✅ Detected language: {detected_language}")
    
    return original_segments, detected_language


def translate_to_english(segments, source_language):
    """
    Translate segments to English using Argos Translate.
    
    Args:
        segments: List of transcription segments
        source_language: Source language code
        
    Returns:
        list: Segments with English translations
    """
    if source_language == "en":
        print("✅ Already in English, skipping translation")
        return [
            {**seg, "english_text": seg["text"]}
            for seg in segments
        ]
    
    print(f"\n🔄 Translating from {source_language} to English...")
    
    # Download and install translation package
    argostranslate.package.update_package_index()
    packages = argostranslate.package.get_available_packages()
    
    try:
        pkg = next(
            p for p in packages
            if p.from_code == source_language and p.to_code == "en"
        )
        argostranslate.package.install_from_path(pkg.download())
    except StopIteration:
        print(f"⚠️ Translation package not found for {source_language} → en")
        print("   Using original text as fallback")
        return [
            {**seg, "english_text": seg["text"]}
            for seg in segments
        ]
    
    # Translate each segment
    english_segments = []
    for seg in segments:
        translated_text = argostranslate.translate.translate(
            seg["text"],
            source_language,
            "en"
        )
        
        english_segments.append({
            **seg,
            "english_text": translated_text
        })
    
    print("✅ Translation complete")
    return english_segments


def save_transcription_files(audio_file_path, original_segments, english_segments, detected_language):
    """
    Save transcription results to JSON files.
    
    Args:
        audio_file_path: Path to the audio file
        original_segments: Segments in original language
        english_segments: Segments with English translation
        detected_language: Detected language code
        
    Returns:
        tuple: (original_json_path, english_json_path)
    """
    output_dir = config.get_output_dir(audio_file_path)
    audio_name = Path(audio_file_path).name
    base_name = Path(audio_file_path).stem
    
    # Save original transcription
    original_json = {
        "audio_file": audio_name,
        "detected_language": detected_language,
        "agent_speaker_ids": list(config.AGENT_SPEAKERS),
        "segments": original_segments
    }
    
    original_json_path = output_dir / f"{base_name}_original.json"
    with open(original_json_path, "w", encoding="utf-8") as f:
        json.dump(original_json, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved: {original_json_path}")
    
    # Save English transcription
    english_json = {
        "audio_file": audio_name,
        "source_language": detected_language,
        "target_language": "en",
        "agent_speaker_ids": list(config.AGENT_SPEAKERS),
        "segments": english_segments
    }
    
    english_json_path = output_dir / f"{base_name}_english.json"
    with open(english_json_path, "w", encoding="utf-8") as f:
        json.dump(english_json, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved: {english_json_path}")
    
    return original_json_path, english_json_path


def process_audio_file(audio_file_path, api_key=None):
    """
    Complete transcription pipeline for a single audio file.
    
    Args:
        audio_file_path: Path to the audio file
        api_key: Deepgram API key (optional, will use env var if not provided)
        
    Returns:
        tuple: (original_json_path, english_json_path)
    """
    if api_key is None:
        api_key = config.DEEPGRAM_API_KEY
    
    if not api_key:
        raise ValueError("Deepgram API key not found. Set DEEPGRAM_API_KEY environment variable.")
    
    print(f"\n{'='*60}")
    print(f"Processing: {Path(audio_file_path).name}")
    print(f"{'='*60}")
    
    # Step 1: Transcribe
    response = transcribe_audio(audio_file_path, api_key)
    
    # Step 2: Process transcription
    audio_filename = Path(audio_file_path).name
    original_segments, detected_language = process_transcription(response, audio_filename)
    
    # Step 3: Translate to English
    english_segments = translate_to_english(original_segments, detected_language)
    
    # Step 4: Save results
    original_json_path, english_json_path = save_transcription_files(
        audio_file_path,
        original_segments,
        english_segments,
        detected_language
    )
    
    print(f"\n✅ Transcription complete for {Path(audio_file_path).name}\n")
    
    return original_json_path, english_json_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m cutout_pipeline.transcribe <audio_file_path>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)
    
    try:
        process_audio_file(audio_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
