#!/usr/bin/env python3
"""
Audio Processing Pipeline - Main CLI Entry Point

This script processes all audio files in a directory:
1. Transcribes each audio file using Deepgram API
2. Translates to English if needed
3. Analyzes for cutouts and generates reports
4. Creates visualizations

Usage:
    python run_pipeline.py <directory_path>
    
Example:
    python run_pipeline.py c:/Users/aksha/cutout
"""
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from cutout_pipeline import config
from cutout_pipeline.transcribe import process_audio_file
from cutout_pipeline.analyze import analyze_transcription


def find_audio_files(directory):
    """
    Find all supported audio files in the directory.
    
    Args:
        directory: Path to directory to search
        
    Returns:
        list: List of Path objects for audio files
    """
    directory = Path(directory)
    audio_files = []
    
    for ext in config.SUPPORTED_AUDIO_FORMATS:
        audio_files.extend(directory.glob(f"*{ext}"))
    
    return sorted(audio_files)


def process_directory(directory_path, api_key=None):
    """
    Process all audio files in a directory.
    
    Args:
        directory_path: Path to directory containing audio files
        api_key: Deepgram API key (optional)
        
    Returns:
        dict: Summary of processing results
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return None
    
    if not directory.is_dir():
        print(f"❌ Not a directory: {directory}")
        return None
    
    # Find audio files
    audio_files = find_audio_files(directory)
    
    if not audio_files:
        print(f"❌ No audio files found in: {directory}")
        print(f"   Supported formats: {', '.join(config.SUPPORTED_AUDIO_FORMATS)}")
        return None
    
    print("\n" + "="*80)
    print(f"🎯 AUDIO PROCESSING PIPELINE")
    print("="*80)
    print(f"Directory: {directory}")
    print(f"Found {len(audio_files)} audio file(s):")
    for i, audio_file in enumerate(audio_files, 1):
        print(f"  {i}. {audio_file.name}")
    print("="*80 + "\n")
    
    # Process each audio file
    results = {
        "total_files": len(audio_files),
        "successful": 0,
        "failed": 0,
        "files": []
    }
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{'#'*80}")
        print(f"# Processing File {i}/{len(audio_files)}: {audio_file.name}")
        print(f"{'#'*80}\n")
        
        file_result = {
            "filename": audio_file.name,
            "status": "pending",
            "error": None,
            "outputs": {}
        }
        
        try:
            # Step 1: Transcription & Translation
            original_json, english_json = process_audio_file(str(audio_file), api_key)
            file_result["outputs"]["original_json"] = str(original_json)
            file_result["outputs"]["english_json"] = str(english_json)
            
            # Step 2: Analysis (use English version)
            analysis_result = analyze_transcription(str(english_json))
            file_result["outputs"]["analysis"] = analysis_result
            
            file_result["status"] = "success"
            results["successful"] += 1
            
            print(f"\n✅ Successfully processed: {audio_file.name}")
            
        except Exception as e:
            file_result["status"] = "failed"
            file_result["error"] = str(e)
            results["failed"] += 1
            
            print(f"\n❌ Failed to process: {audio_file.name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
        
        results["files"].append(file_result)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 PIPELINE SUMMARY")
    print("="*80)
    print(f"Total files: {results['total_files']}")
    print(f"✅ Successful: {results['successful']}")
    print(f"❌ Failed: {results['failed']}")
    
    if results["failed"] > 0:
        print("\nFailed files:")
        for file_result in results["files"]:
            if file_result["status"] == "failed":
                print(f"  - {file_result['filename']}: {file_result['error']}")
    
    print("="*80 + "\n")
    
    return results


def main():
    """Main entry point for the pipeline."""
    if len(sys.argv) < 2:
        print("❌ Usage: python run_pipeline.py <directory_path>")
        print("\nExample:")
        print("  python run_pipeline.py c:/Users/aksha/cutout")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    # Check for API key
    if not config.DEEPGRAM_API_KEY:
        print("❌ Deepgram API key not found!")
        print("   Please set the DEEPGRAM_API_KEY environment variable.")
        print("\nOn Windows (PowerShell):")
        print('  $env:DEEPGRAM_API_KEY="your_api_key_here"')
        print("\nOn Unix/Mac:")
        print('  export DEEPGRAM_API_KEY="your_api_key_here"')
        sys.exit(1)
    
    # Process directory
    try:
        results = process_directory(directory_path)
        
        if results is None:
            sys.exit(1)
        
        # Exit with error code if any files failed
        if results["failed"] > 0:
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
