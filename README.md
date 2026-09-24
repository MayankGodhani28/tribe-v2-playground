# TRIBE v2 Playground

A small, Colab-friendly playground for testing Meta Research's TRIBE v2 pretrained model on video, audio, or text.

## What it does

TRIBE v2 predicts naturalistic fMRI brain responses from multimodal stimuli. Predictions are made on the fsaverage5 cortical mesh (~20k vertices) and are shifted by 5 seconds to account for hemodynamic lag.

Official model/API:
- https://github.com/facebookresearch/tribev2
- Hugging Face model: facebook/tribev2

## Quick start: Google Colab

Open `tribe_demo.ipynb` in Colab, then run the cells from top to bottom.

The notebook:
1. Installs the official TRIBE v2 package.
2. Downloads the pretrained model on first use.
3. Lets you upload an MP4.
4. Runs video -> predicted cortical activity inference.
5. Saves predictions and timing metadata.
6. Creates a simple heatmap over time.
7. Optionally creates a cortical surface plot when plotting dependencies are available.

## Local setup

Recommended: Python 3.11-3.12 with a recent PyTorch build.

```bash
git clone https://github.com/MayankGodhani28/tribe-v2-playground.git
cd tribe-v2-playground

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Then place a test video at `data/test.mp4` and run:

```bash
python run_tribe.py --video data/test.mp4 --output outputs
```

The script writes:
- `predictions.npy`
- `segments.json`
- `summary.json`
- `mean_activity.png` when matplotlib is available

## Important interpretation note

TRIBE v2 is an in-silico neuroscience model. Its prediction is a model-estimated average-subject brain response, not a measurement of your personal brain activity and not a diagnostic or "engagement score."

## Official reference

See the upstream README for the authoritative installation/API details:
https://github.com/facebookresearch/tribev2
