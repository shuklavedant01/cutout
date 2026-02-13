# Running Audio Cutout Pipeline in Google Colab

Quick guide to run the pipeline in a fresh Colab notebook.

---

## Step 1: Clone Repository

```python
# Replace with your GitHub repo URL after uploading
!git clone https://github.com/<your-username>/<your-repo-name>.git
%cd <your-repo-name>
```

---

## Step 2: Install Dependencies

```python
!pip install -q -r requirements.txt
```

---

## Step 3: Set API Key

```python
import os

# Replace with your actual Deepgram API key
os.environ["DEEPGRAM_API_KEY"] = "YOUR_DEEPGRAM_API_KEY_HERE"
```

---

## Step 4: Check Audio Files (Already in Repo!)

Since your audio files are already in the `audio_files/` folder, you can skip uploading:

```python
import os

# List audio files in the repo
audio_dir = "audio_files"
audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg'))]

print(f"📁 Found {len(audio_files)} audio file(s) in {audio_dir}/:")
for file in audio_files:
    print(f"  - {file}")
```

**Optional:** If you want to add more audio files:
```python
from google.colab import files
uploaded = files.upload()

# Move uploaded files to audio_files folder
import shutil
for filename in uploaded.keys():
    shutil.move(filename, f"audio_files/{filename}")
```

---

## Step 5: Run the Pipeline

```python
# Process all audio files in the audio_files folder
!python run_pipeline.py audio_files
```

---

## Step 6: View Results

```python
import os

# List all output folders
output_folders = [f for f in os.listdir() if f.endswith('_output')]

print(f"📊 Found {len(output_folders)} output folder(s):\n")
for folder in output_folders:
    print(f"📁 {folder}/")
    for file in os.listdir(folder):
        print(f"   - {file}")
```

---

## Step 7: Display Graphs

```python
from IPython.display import Image, display

# Show the dashboard from the first output folder
output_folders = [f for f in os.listdir() if f.endswith('_output')]

if output_folders:
    folder = output_folders[0]
    dashboard = [f for f in os.listdir(folder) if 'dashboard' in f][0]
    
    print(f"📊 Dashboard: {folder}/{dashboard}\n")
    display(Image(filename=f"{folder}/{dashboard}"))
```

---

## Step 8: Download Results

```python
from google.colab import files
import shutil

# Zip and download all output folders
output_folders = [f for f in os.listdir() if f.endswith('_output')]

print(f"📦 Zipping {len(output_folders)} folders...\n")

for folder in output_folders:
    shutil.make_archive(folder, 'zip', folder)
    print(f"⬇️  Downloading: {folder}.zip")
    files.download(f"{folder}.zip")

print("\n✅ All downloads complete!")
```

---

## Advanced: Run Individual Modules

```python
# Just transcribe
!python -m cutout_pipeline.transcribe audio1.wav

# Just analyze
!python -m cutout_pipeline.analyze audio1_output/audio1_english.json
```

---

## Customize Configuration

```python
# Modify settings before running
import sys
sys.path.insert(0, 'src')

from cutout_pipeline import config

# Change agent speakers
config.AGENT_SPEAKERS = {0, 1}  # Adjust to your speaker IDs

# Adjust thresholds
config.Z_SCORE_SILENCE = 2.5
config.Z_SCORE_GRAMMAR = 2.0

print("✅ Configuration updated!")
```

---

## Complete One-Cell Script

For quick testing, paste this entire cell:

```python
# Complete pipeline in one cell (audio files already in repo!)

# 1. Clone repo (replace with your URL)
!git clone https://github.com/<username>/<repo>.git
%cd <repo>

# 2. Install
!pip install -q -r requirements.txt

# 3. Set API key
import os
os.environ["DEEPGRAM_API_KEY"] = "YOUR_KEY_HERE"

# 4. Check audio files
audio_files = [f for f in os.listdir('audio_files') if f.endswith(('.wav', '.mp3', '.m4a'))]
print(f"📁 Found {len(audio_files)} audio files")

# 5. Run pipeline
!python run_pipeline.py audio_files

# 6. Download results (outputs will be in audio_files/ directory)
import shutil
for item in os.listdir('audio_files'):
    if item.endswith('_output'):
        folder = f'audio_files/{item}'
        shutil.make_archive(item, 'zip', folder)
        files.download(f"{item}.zip")

from google.colab import files
print("✅ Pipeline complete!")
```
