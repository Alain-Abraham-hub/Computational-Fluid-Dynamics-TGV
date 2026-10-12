"""
make_gif.py
-----------
Stitches exported ParaView PNG animation frames into a looping GIF inside 'Para View Visualization/'.
"""

import glob
import os
import sys
from PIL import Image


def create_gif(
    search_dir=None,
    pattern="*.png",
    output_path=None,
    fps=6,
):
    base_viz_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Para View Visualization")
    )
    if search_dir is None:
        search_dir = os.path.join(base_viz_dir, "TGV decay png's")
    if output_path is None:
        output_path = os.path.join(base_viz_dir, "tgv_decay_animation.gif")

    files = sorted(glob.glob(os.path.join(search_dir, pattern)))
    if not files:
        print(f"No PNG frames found in: {search_dir}")
        sys.exit(1)

    print(f"Found {len(files)} frames. Creating {output_path} at {fps} fps...")
    frames = [Image.open(f) for f in files]
    duration_ms = int(1000 / fps)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"[*] Saved animation to: {output_path}")


if __name__ == "__main__":
    create_gif()
