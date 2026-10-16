FROM mcr.microsoft.com/devcontainers/python:3.2.6-3.12-bookworm

# Libraries needed by PyQt5's XCB platform plugin in a real desktop session.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libegl1 libdbus-1-3 libfontconfig1 libxkbcommon-x11-0 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
    libxcb-xinerama0 libxcb-randr0 libxcb-shape0 libxcb-xfixes0 \
    fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*
