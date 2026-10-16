#!/usr/bin/env python3
# Copyright (c) 2026 Hasan Wadhah. All rights reserved.
"""Explicit, one-time U2-Net download. The desktop app never downloads models."""
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

URL = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx'
EXPECTED_MD5 = '60024c5c889badc19c04ad937298a77b'


def valid(path):
    if not path.is_file():
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'md5').hexdigest() == EXPECTED_MD5


def main():
    directory = Path(__file__).resolve().parent / 'models'
    directory.mkdir(exist_ok=True)
    destination = directory / 'u2net.onnx'
    if valid(destination):
        print('The local U2-Net model is already verified.')
        return
    fd, name = tempfile.mkstemp(prefix='u2net-', suffix='.part', dir=directory)
    os.close(fd)
    temporary = Path(name)
    try:
        print('Downloading U2-Net (about 176 MB) from the official rembg release...')
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if not valid(temporary):
            raise RuntimeError('Model checksum mismatch. The downloaded file was rejected.')
        os.replace(temporary, destination)
        print('Verified. Smart detection is now available offline.')
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
