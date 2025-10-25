#!/usr/bin/env bash
# build.sh

# Install system dependencies for pydub
apt-get update
apt-get install -y ffmpeg libavcodec-extra

# Install Python dependencies
pip install -r requirements.txt
