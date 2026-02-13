# Audio Cutout Pipeline

A complete pipeline for transcribing, translating, and analyzing audio conversations for cutout detection.

## Features

- **Automatic Transcription**: Uses Deepgram API for high-quality speech-to-text with speaker diarization
- **Language Detection**: Automatically detects the language of the conversation
- **Translation**: Translates non-English conversations to English using Argos Translate
- **Cutout Detection**: Identifies problematic segments including:
  - Unusual silences
  - Grammar issues
  - Empty acknowledgments
- **Comprehensive Reports**: Generates CSV, JSON, and visual graphs for analysis
- **Batch Processing**: Process all audio files in a directory with one command

## Project Structure

```
cutout/
├── src/
│   └── cutout_pipeline/        # Main package
│       ├── __init__.py          # Package initialization
│       ├── config.py            # Configuration & constants
│       ├── transcribe.py        # Transcription & translation
│       └── analyze.py           # Cutout detection & analysis
├── run_pipeline.py              # Main CLI entry point
├── requirements.txt             # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
└── colab_quickstart.ipynb      # Google Colab notebook
```

## Output Files

For each audio file `audio1.wav`, the pipeline creates an `audio1_output/` directory containing:

- `audio1_original.json` - Transcription in original language
- `audio1_english.json` - English translation
- `audio1_results.csv` - Per-turn analysis data
- `audio1_report.json` - Statistics and cutout details
- `audio1_graph_1_timeline.png` - Conversation timeline visualization
- `audio1_graph_2_latency.png` - Agent response latency chart
- `audio1_graph_3_distribution.png` - Latency distribution histogram
- `audio1_analysis_dashboard.png` - Combined dashboard view

## Installation & Usage

### Option 1: Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd cutout
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**
   
   **Windows (PowerShell):**
   ```powershell
   $env:DEEPGRAM_API_KEY="your_api_key_here"
   ```
   
   **Mac/Linux:**
   ```bash
   export DEEPGRAM_API_KEY="your_api_key_here"
   ```

4. **Run the pipeline**
   ```bash
   python run_pipeline.py <directory_path>
   ```
   
   Example:
   ```bash
   python run_pipeline.py c:/Users/aksha/cutout
   ```

### Option 2: Google Colab

1. **Open a new Colab notebook** or use the provided `colab_quickstart.ipynb`

2. **Clone the repository**
   ```python
   !git clone <your-repo-url>
   %cd cutout
   ```

3. **Install dependencies**
   ```python
   !pip install -r requirements.txt
   ```

4. **Set your API key**
   ```python
   import os
   os.environ["DEEPGRAM_API_KEY"] = "your_api_key_here"
   ```

5. **Upload audio files**
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

6. **Run the pipeline**
   ```python
   !python run_pipeline.py /content/cutout
   ```

7. **Download results**
   ```python
   from google.colab import files
   import shutil
   
   # Zip and download output folders
   for folder in os.listdir():
       if folder.endswith('_output'):
           shutil.make_archive(folder, 'zip', folder)
           files.download(f'{folder}.zip')
   ```

## Configuration

Edit `src/cutout_pipeline/config.py` to customize:

- **AGENT_SPEAKERS**: Set which speaker IDs represent the agent (default: `{0, 2}`)
- **Z_SCORE_SILENCE**: Threshold for silence detection (default: `3.0`)
- **Z_SCORE_GRAMMAR**: Threshold for grammar issues (default: `2.5`)
- **ACK_PERCENTILE**: Percentile for acknowledgment detection (default: `0.20`)

## Individual Module Usage

You can also run modules individually using Python's `-m` flag:

### Transcription Only
```bash
python -m cutout_pipeline.transcribe audio1.wav
```

### Analysis Only
```bash
python -m cutout_pipeline.analyze audio1_output/audio1_english.json
```

## Supported Audio Formats

- WAV (`.wav`)
- MP3 (`.mp3`)
- M4A (`.m4a`)
- FLAC (`.flac`)
- OGG (`.ogg`)

## Requirements

- Python 3.7+
- Deepgram API key ([Get one here](https://console.deepgram.com/))
- Internet connection for API calls and model downloads

## Troubleshooting

### "Deepgram API key not found"
Make sure you've set the `DEEPGRAM_API_KEY` environment variable before running the pipeline.

### "No audio files found"
Ensure your audio files have supported extensions and are in the specified directory.

### "Translation package not found"
Some language pairs may not be available in Argos Translate. The pipeline will use the original text as a fallback.

### "Module not found" errors
Make sure you're running the pipeline from the project root directory (`cutout/`).

## License

MIT License - Feel free to use and modify as needed.

## Credits

- **Deepgram** - Audio transcription API
- **Argos Translate** - Offline translation
- **Sentence Transformers** - Semantic analysis
- **Hugging Face Transformers** - Fluency analysis
