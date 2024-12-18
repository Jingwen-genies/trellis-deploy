FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ninja-build \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*


# Set environment variables
ENV CUDA_VERSION=11.8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Pytorch
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --extra-index-url https://download.pytorch.org/whl/cu118 && \

# Install other Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CUDA-specific packages
RUN pip install --no-cache-dir xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir flash-attn && \
    pip install --no-cache-dir spconv-cu118 && \
    pip install --no-cache-dir "tbb>=2021.6.0"

# Install Kaolin
RUN pip install --no-cache-dir kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html

# Install NVDIFFRAST
RUN git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && \
    pip install /tmp/nvdiffrast && \
    rm -rf /tmp/nvdiffrast

# Install DIFFOCTREERAST
RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast && \
    pip install /tmp/diffoctreerast && \
    rm -rf /tmp/diffoctreerast

# Install MIP-Splatting
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting && \
    pip install /tmp/mip-splatting/submodules/diff-gaussian-rasterization/ && \
    rm -rf /tmp/mip-splatting


# Make port 8000 available to the world outside this container
EXPOSE 8000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "300", "app:__hug_wsgi__"]