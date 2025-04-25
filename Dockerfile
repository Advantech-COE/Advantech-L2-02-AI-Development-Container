# Dockerfile for Jetson Orin with JetPack 5.1.1-5.1.3
# Production-ready with full hardware acceleration support
# Maintainer: Samir Singh <samir.singh@advantech.com>

# Base image: NVIDIA L4T with JetPack 5.1.2 (compatible with 5.1.1-5.1.3)
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Add labels for container identification and metadata
LABEL maintainer="Samir Singh <samir.singh@advantech.com>" \
      version="1.2" \
      description="Advantech Jetson Orin ML Container with full hardware acceleration" \
      vendor="Advantech" \
      project="Advantech Edge AI"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all,compute,video,utility,graphics \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set up working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    python3-numpy \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libgstrtspserver-1.0-0 \
    libgstrtspserver-1.0-dev \
    v4l-utils \
    libv4l-dev \
    libx264-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install HDF5 dependencies required for TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    zlib1g-dev \
    zip \
    libjpeg8-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install h5py first (needed for TensorFlow)
RUN python3 -m pip install --no-cache-dir \
    Cython \
    numpy \
    six \
    && MAKEFLAGS="-j$(nproc)" python3 -m pip install --no-cache-dir --no-build-isolation \
    h5py==3.7.0

# Install core ML/AI frameworks
RUN python3 -m pip install --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v511 \
    tensorflow==2.12.0+nv23.05 \
    onnx

# Install ONNX Runtime
RUN python3 -m pip install --no-cache-dir onnxruntime || \
    python3 -m pip install --no-cache-dir "onnxruntime==1.16.0"

# Install TFLite runtime
RUN python3 -m pip install --no-cache-dir tflite-runtime || \
    echo "TFLite runtime not available for this architecture"

# Install other Python packages
RUN python3 -m pip install --no-cache-dir \
    matplotlib \
    pillow \
    scipy \
    scikit-learn \
    pandas \
    pycuda

# Create directories for app structure
RUN mkdir -p /app/src /app/utils /app/data /app/models /app/diagnostics

# Copy hardware acceleration setup utilities
COPY ./utils/hw_setup.sh /app/utils/
RUN chmod +x /app/utils/hw_setup.sh

# Create basic entrypoint script for hardware setup
RUN echo '#!/bin/bash\n\
# Setup hardware acceleration\n\
/app/utils/hw_setup.sh\n\
\n\
# Execute command passed to docker run\n\
exec "$@"\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["python3", "-c", "import tensorflow as tf; print('TensorFlow GPU Available:', len(tf.config.list_physical_devices(\"GPU\")) > 0); import torch; print('PyTorch CUDA Available:', torch.cuda.is_available()); print('Ready to use!')"]