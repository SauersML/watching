"""Paint the robot, cat and owl with p5.brush in headless Chrome and save them as PNGs."""

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "scripts/draw_characters.html"
CHROME = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
)


def chrome():
    for candidate in CHROME:
        found = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if found:
            return found
    raise SystemExit("Chrome or Chromium is required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=["robot", "cat", "owl"])
    parser.add_argument("--out", type=Path, default=ROOT / "figures/assets")
    args = parser.parse_args()
    for name in args.names:
        subprocess.run(
            [
                chrome(),
                "--headless=new",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                "--allow-file-access-from-files",
                "--hide-scrollbars",
                "--window-size=600,600",
                "--virtual-time-budget=8000",
                f"--screenshot={args.out / f'{name}.png'}",
                f"{PAGE.as_uri()}#{name}",
            ],
            check=True,
            capture_output=True,
        )


if __name__ == "__main__":
    main()
