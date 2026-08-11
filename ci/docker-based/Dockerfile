# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# triton-xdna dev container with GitHub Actions runner
# To run with NPU device passthrough, use:
# docker run -v /dev/accel/accel0:/dev/accel/accel0 --device-cgroup-rule 'c 261:* rmw' --ulimit memlock=-1:-1 <image>

# To build this docker, run from the repo root:
# docker build -f ci/docker-based/Dockerfile -t triton-xdna-dev-github-runner

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    # MLIR-AIE build requirements  
    build-essential ninja-build cmake python3 python3-dev python3.12-venv python3-pip unzip \
    libssl-dev uuid-dev \
    # Runtime deps of the host-mounted XRT tools. xrt-smi links against
    # libboost_filesystem/libboost_program_options; without them it exits 127
    # with "error while loading shared libraries" and every XRT query that
    # shells out to it silently reports no NPU.
    libboost-filesystem1.83.0 libboost-program-options1.83.0 \
    # Required by MHA kernel
    llvm-18 \
    # Clang toolchain for Triton build
    clang lld \
    # GitHub Actions runner requirements
    git jq curl tar \
    # Node.js runtime for JavaScript-based GitHub Actions (e.g., ccache action)
    nodejs npm \
 && rm -rf /var/lib/apt/lists/*

ARG USER=triton-xdna
RUN useradd -m -s /bin/bash ${USER}
RUN mkdir -p /workspace && chown ${USER}:${USER} /workspace

USER ${USER}
WORKDIR /workspace

# COPY ./requirements.txt /workspace/requirements.txt

# Environment setup 
# ---

ENV HOME=/workspace 

# Ensure XRT is mounted by the host
# This errors if it does not exist.
# copy entrypoint into image and make it executable (ensure file is in the build context)
COPY --chown=${USER}:${USER} ./ci/docker-based/inside_docker /opt/inside_docker/
RUN chmod +x /opt/inside_docker/entrypoint.sh /opt/inside_docker/actions_runner.sh
ENTRYPOINT ["/opt/inside_docker/entrypoint.sh"]

CMD ["/opt/inside_docker/actions_runner.sh"]
