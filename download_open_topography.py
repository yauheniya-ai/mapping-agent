#!/usr/bin/env python3
import subprocess
import os
from pathlib import Path

# Download files example: aws s3 cp s3://pc-bulk/WA23_JohnHenry/ . --recursive --endpoint-url https://opentopography.s3.sdsc.edu --no-sign-request
# Source: https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.112024.6339.2

TARGET_DIR = "WA23_JohnHenry"
S3_PATH = "s3://pc-bulk/WA23_JohnHenry/"
ENDPOINT = "https://opentopography.s3.sdsc.edu"

def main():
    print("📥 Downloading OpenTopography LiDAR dataset...")
    Path(TARGET_DIR).mkdir(exist_ok=True)

    cmd = [
        "aws", "s3", "cp", S3_PATH, TARGET_DIR,
        "--recursive",
        "--endpoint-url", ENDPOINT,
        "--no-sign-request"
    ]

    try:
        subprocess.run(cmd, check=True)
        files = list(Path(TARGET_DIR).glob("*"))
        total_size = sum(f.stat().st_size for f in files) / (1024**2)
        print(f"\n✅ Downloaded {len(files)} files ({total_size:.2f} MB) to '{TARGET_DIR}'")
    except subprocess.CalledProcessError:
        print("❌ Download failed. Make sure AWS CLI is installed and accessible.")

if __name__ == "__main__":
    main()
