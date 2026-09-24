import argparse
import json
from pathlib import Path

import numpy as np

from tribev2 import TribeModel


def save_summary(preds: np.ndarray, segments, output_dir: Path) -> None:
    summary = {
        "shape": list(preds.shape),
        "dtype": str(preds.dtype),
        "timesteps": int(preds.shape[0]),
        "vertices": int(preds.shape[1]),
        "prediction_min": float(np.nanmin(preds)),
        "prediction_max": float(np.nanmax(preds)),
        "prediction_mean": float(np.nanmean(preds)),
        "prediction_std": float(np.nanstd(preds)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    normalized_segments = []
    for segment in segments:
        if hasattr(segment, "to_dict"):
            segment = segment.to_dict()
        elif not isinstance(segment, dict):
            segment = {"value": str(segment)}
        normalized_segments.append(segment)

    (output_dir / "segments.json").write_text(
        json.dumps(normalized_segments, indent=2, default=str),
        encoding="utf-8",
    )


def save_heatmap(preds: np.ndarray, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Downsample vertices only for a readable diagnostic image.
    image = preds[:, ::100]
    plt.figure(figsize=(12, 5))
    plt.imshow(image.T, aspect="auto", interpolation="nearest")
    plt.xlabel("Prediction timestep")
    plt.ylabel("Cortical vertices (downsampled)")
    plt.title("TRIBE v2 predicted cortical activity")
    plt.colorbar(label="Predicted response")
    plt.tight_layout()
    plt.savefig(output_dir / "mean_activity.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to an MP4/video file")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument(
        "--cache",
        default="./cache",
        help="Hugging Face/model cache directory",
    )
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TRIBE v2 pretrained model...")
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=args.cache,
    )

    print(f"Building events from: {video}")
    events = model.get_events_dataframe(video_path=str(video))

    print("Running inference...")
    preds, segments = model.predict(events=events)
    preds = np.asarray(preds)

    np.save(output_dir / "predictions.npy", preds)
    save_summary(preds, segments, output_dir)
    save_heatmap(preds, output_dir)

    print("\nDone.")
    print(f"Predictions shape: {preds.shape}")
    print(f"Saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
