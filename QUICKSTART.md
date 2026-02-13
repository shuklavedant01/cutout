# Quick Start Guide

## Running Locally

Since your audio files are already in the `audio_files/` folder:

```powershell
# Set your API key
$env:DEEPGRAM_API_KEY="your_deepgram_api_key_here"

# Run the pipeline
python run_pipeline.py audio_files
```

All output folders will be created inside `audio_files/`:
```
audio_files/
├── audio1.wav
├── audio1_output/          # Created by pipeline
│   ├── audio1_original.json
│   ├── audio1_english.json
│   ├── audio1_results.csv
│   └── ...
├── audio2.wav
└── audio2_output/          # Created by pipeline
    └── ...
```

---

## Running on Google Colab

**Super Quick Version - Single Cell:**

```python
# Clone and run (replace with your GitHub URL)
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>

# Install
!pip install -q -r requirements.txt

# Set API key
import os
os.environ["DEEPGRAM_API_KEY"] = "YOUR_KEY_HERE"

# Run pipeline on audio_files folder
!python run_pipeline.py audio_files

# Download results
from google.colab import files
import shutil
for item in os.listdir('audio_files'):
    if item.endswith('_output'):
        shutil.make_archive(item, 'zip', f'audio_files/{item}')
        files.download(f"{item}.zip")

print("✅ Done!")
```

---

## Important Notes

- **Output Location**: All `*_output/` folders are created in the same directory as the audio files
- **Git Tracking**: The `.gitignore` already excludes `*_output/` folders, so they won't be committed
- **Multiple Runs**: Safe to run multiple times - outputs will be overwritten

---

## Need More Details?

- See [`COLAB_GUIDE.md`](COLAB_GUIDE.md) for step-by-step breakdown
- See [`README.md`](README.md) for full documentation
- See [`colab_quickstart.ipynb`](colab_quickstart.ipynb) for interactive notebook
