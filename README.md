# LipNet

A Streamlit app and Jupyter notebook implementation of LipNet, a deep learning model
that reads lips from silent video and predicts the spoken text.

Includes:
- `LipNet.ipynb` — model architecture, training pipeline, and inference walkthrough
- `app/` — a Streamlit web app (`streamlitapp.py`) for interactively picking a video
  and viewing the model's prediction

## Setup

### 1. Create a virtual environment (Python 3.10–3.13)

```bash
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install ffmpeg

The app uses ffmpeg to convert `.mpg` videos to `.mp4` for browser playback.

- Windows: download from https://www.gyan.dev/ffmpeg/builds/ and add the `bin` folder to your PATH
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

### 4. Download the dataset

The GRID corpus (speaker s1) videos and alignments aren't included in this repo.
Download and extract them so you end up with:

```
data/
├── s1/                (video files, .mpg)
└── alignments/s1/     (alignment files, .align)
```

```bash
pip install gdown
python -m gdown 1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL -O data.zip
```
Then extract `data.zip` into `data/` (note: the zip contains its own top-level
`data` folder — flatten it so `s1` and `alignments` sit directly inside your
project's `data/` folder).

### 5. Add the pretrained model checkpoint

Not included in this repo (large binary files). Place the checkpoint files at:

```
models/
├── checkpoint
├── checkpoint.index
└── checkpoint.data-00000-of-00001
```

## Running the app

```bash
cd app
streamlit run streamlitapp.py
```

## Notes

- This implementation was originally written for an older TensorFlow/Keras version.
  If you're on Keras 3 (TensorFlow 2.16+), model checkpoint loading uses
  `tf.train.Checkpoint` rather than `model.load_weights()` directly, and GIF preview
  generation rescales frames to uint8 before saving (newer `imageio`/Pillow no longer
  accept raw float32 arrays).
