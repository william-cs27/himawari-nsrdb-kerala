"""Real infrared examples, common Kelvin scale, no image enhancement."""

from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from weather_fusion.common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
paths = [
    ROOT / f"data/patches/smoke/{day}T0400Z.npz" for day in ["20180101", "20180103"]
]
images = []
metadata = []
for path in paths:
    with np.load(path, allow_pickle=False) as data:
        images.append(np.where(data["valid"][0], data["image"][0], np.nan))
    metadata.append(json.loads(path.with_suffix(".json").read_text()))
lower = np.floor(min(np.nanmin(a) for a in images) / 5) * 5
upper = np.ceil(max(np.nanmax(a) for a in images) / 5) * 5
output = ROOT / "docs/figures"
output.mkdir(parents=True, exist_ok=True)
with plt.rc_context(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    }
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), layout="constrained")
    cmap = matplotlib.colormaps["cividis"].with_extremes(bad="#dedede")
    for ax, arr, meta, panel in zip(axes, images, metadata, ["A", "B"]):
        half = meta["extent_km"] / 2
        im = ax.imshow(
            arr,
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            interpolation="nearest",
            origin="upper",
            extent=(-half, half, -half, half),
        )
        ax.plot(0, 0, marker="+", color="white", ms=11, mew=3)
        ax.plot(0, 0, marker="+", color="black", ms=10, mew=1.3)
        ax.set(
            title=f"{panel}   {meta['nominal_time'][:10]} 04:00 UTC",
            xlabel="Eastward distance from site (km)",
            ylabel="Northward distance from site (km)",
            xticks=[-128, -64, 0, 64, 128],
            yticks=[-128, -64, 0, 64, 128],
        )
    fig.colorbar(im, ax=axes, shrink=0.85, label="B13 brightness temperature (K)")
    fig.suptitle("Real Himawari-8 crops around the Kerala site", fontsize=15)
    fig.supxlabel(
        "32 × 32 grid; 8 km grid spacing; no smoothing or parallax correction.\n+ marks 9.96°N, 76.25°E; nominal scan-start times shown.",
        fontsize=10,
    )
    for suffix in ["png", "svg"]:
        fig.savefig(
            output / f"himawari_sample.{suffix}",
            dpi=180,
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)
write_json(
    output / "himawari_sample_provenance.json",
    {
        "purpose": "general project diagnostic; no journal target",
        "figure_inches": [10, 5.2],
        "png_dpi": 180,
        "color_limits_K": [float(lower), float(upper)],
        "transformations": [
            "Satpy HSD calibration",
            "nearest-neighbor AEQD regional crop",
            "no enhancement or smoothing",
        ],
        "sources": [
            {
                "file": str(p.relative_to(ROOT)),
                "sha256": sha256(p),
                "nominal_time": m["nominal_time"],
                "observation_end": m["observation_end"],
                "min_K": float(np.nanmin(a)),
                "max_K": float(np.nanmax(a)),
                "valid_fraction": float(np.isfinite(a).mean()),
            }
            for p, m, a in zip(paths, metadata, images)
        ],
        "uncertainty": "two engineering example frames; not a statistical estimate",
        "alt_text": "Two regional infrared temperature crops centered on Kerala, January 1 and January 3 2018, on a shared physical scale. Spatial variability and differences between dates are visible; these examples do not demonstrate forecast skill.",
    },
)
print(output / "himawari_sample.png")
