"""
Configuration file for Audio Processing Pipeline
"""
import os
from pathlib import Path

# ============================
# API CONFIGURATION
# ============================
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

# ============================
# SPEAKER CONFIGURATION
# ============================
# Define which speaker IDs represent the AGENT
# All other speakers will be labeled as HUMAN
AGENT_SPEAKERS = {0, 2}

# ============================
# TRANSCRIPTION OPTIONS
# ============================
TRANSCRIPTION_OPTIONS = {
    "model": "nova-3",
    "diarize": True,
    "utterances": True,
    "punctuate": True,
    "smart_format": True,
    "detect_language": True
}

# ============================
# ANALYSIS PARAMETERS
# ============================
# Z-score thresholds for outlier detection
Z_SCORE_SILENCE = 3.0
Z_SCORE_GRAMMAR = 2.5

# Acknowledgment detection percentile
ACK_PERCENTILE = 0.20

# ============================
# AI MODEL CONFIGURATION
# ============================
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
FLUENCY_MODEL = "xlm-roberta-base"

# ============================
# FILE EXTENSIONS
# ============================
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.oga']

# ============================
# OUTPUT DIRECTORY
# ============================
def get_output_dir(audio_file_path):
    """
    Create output directory for a given audio file.
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        Path object for the output directory
    """
    audio_path = Path(audio_file_path)
    output_dir = audio_path.parent / f"{audio_path.stem}_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir
