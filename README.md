# TRIBE v2 Playground

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MayankGodhani28/tribe-v2-playground/blob/main/tribe_demo.ipynb)

A small, Colab-friendly playground for testing Meta Research's TRIBE v2 pretrained model on video, audio, or text.

## What it does

TRIBE v2 is a multimodal brain-encoding model that predicts naturalistic fMRI responses from video, audio, and text. Its output lives on the fsaverage5 cortical mesh, roughly 20,000 cortical vertices, and the published predictions are shifted by 5 seconds to compensate for hemodynamic lag.

Official model/API:
- https://github.com/facebookresearch/tribev2
- https://huggingface.co/facebook/tribev2
- https://aidemos.atmeta.com/tribev2/

## Quick start: Google Colab

Click the **Open in Google Colab** badge above.

The notebook:
1. Installs the official TRIBE v2 package.
2. Downloads the pretrained model on first use.
3. Lets you upload a short video.
4. Runs video -> predicted cortical activity inference.
5. Saves predictions and timing metadata.
6. Creates a diagnostic heatmap over time.
7. Points to the upstream plotting workflow for 3D cortical visualization.

For the first experiment, use a short MP4 rather than a long movie. Model loading and inference can be resource-intensive.

## Local setup

Use Python 3.11 or newer.

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

## Interpreting the output

TRIBE v2 predicts an **average-subject** fMRI response. It is not a measurement of your personal brain activity and should not be treated as a medical, psychological, or diagnostic score.

## License

This playground follows the licensing terms of the upstream TRIBE v2 project. The upstream repository is licensed under **CC BY-NC 4.0**, so check those terms before using the model commercially.

## Official reference

For authoritative installation, API, plotting, and model details:
https://github.com/facebookresearch/tribev2
