"""Annotate a video:  python run_video.py input.mp4 output.mp4"""
import argparse

from posture.video import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--weights", default="yolo11x-pose.pt")
    a = ap.parse_args()
    print(f"done; {run(a.input, a.output, a.weights)} person-frames labelled running")
