# The whole library in one image: the C++ ray-time bundle adjuster (pyARBA)
# plus the Python package that drives it.
#
# Build (from repo root):
#   docker build -t pyarba .
#
# Run the quickstart:
#   docker run --rm pyarba python3 -m arba.quickstart
#
# Run against your own working tree:
#   docker run --rm -v "$PWD":/workspace -w /workspace \
#     -e PYTHONPATH=/workspace:/opt/pyarba/build \
#     pyarba python3 -m arba.quickstart
#
# Ubuntu 24.04 is not incidental: it is the oldest Ubuntu whose packaged
# Ceres (2.2.0) has the ceres::Manifold API this code uses, which is what
# lets the whole build come from apt with nothing compiled from source.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libeigen3-dev \
        libceres-dev \
        libsuitesparse-dev \
        libtbb-dev \
        libboost-log-dev \
        libboost-system-dev \
        libboost-filesystem-dev \
        libboost-test-dev \
        python3 \
        python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# pybind11 to build against, and the three runtime packages: numpy, scipy,
# tqdm. Unpinned on purpose; see the file for why.
COPY requirements-run.txt /tmp/requirements-run.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements-run.txt

COPY . /opt/pyarba
RUN cmake -S /opt/pyarba/arcore_bundle_adjuster \
          -B /opt/pyarba/build \
          -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/pyarba/build -j "$(nproc)" \
    && cmake --install /opt/pyarba/build

ENV PYTHONPATH=/opt/pyarba
WORKDIR /opt/pyarba
