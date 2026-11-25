"""Desk / wall / floor textures from the phone video (inputs/IMG_3040.MOV) -> assets/textures/.

    python scripts/build_textures.py      # needs ffmpeg (zscale) + numpy, opencv

Steps (scripts/textures/, run in order inside work/): rectify.py fuses the sharp desk frames into a metric
top-down patch; synth_wood.py extends it to the 1400 x 600 mm top; maps.py writes albedo/normal/roughness.
"""
import os, runpy, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"; (WORK / "lin").mkdir(parents=True, exist_ok=True)
os.environ["TEX_OUT"] = str(ROOT / "assets/textures")
# sharp desk frames at both ends of the clip + a few wall frames, kept HLG-encoded (decoded to scene-linear in colorlin.py)
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(ROOT / "inputs/IMG_3040.MOV"), "-vf",
                r"select='lt(n\,12)+gt(n\,328)+between(n\,100\,185)*not(mod(n\,6))',zscale=min=2020_ncl:rin=limited:m=gbr:r=full,format=gbrp16le",
                "-vsync", "0", "-frame_pts", "1", "-pix_fmt", "rgb48be", str(WORK / "lin/n_%04d.png")], check=True)
os.chdir(WORK); sys.path.insert(0, str(ROOT / "scripts/textures"))
for step in ("rectify", "synth_wood", "maps"):
    print(f"== {step}", flush=True); runpy.run_path(str(ROOT / f"scripts/textures/{step}.py"), run_name="__main__")
