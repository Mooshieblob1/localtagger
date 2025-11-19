FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# clip-interrogator handles EVA/ViT deps. onnxruntime-gpu handles WD14.
# We pin numpy<2.0 to avoid binary incompatibility with PyTorch/ONNX wheels
# We pin transformers to ensure compatibility with PyTorch 2.1
# We pin accelerate to match transformers 4.35 vintage (avoiding huggingface_hub conflicts)
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    fastapi \
    uvicorn \
    python-multipart \
    clip-interrogator==0.6.0 \
    pandas \
    opencv-python-headless \
    huggingface_hub \
    "transformers==4.35.0" \
    "accelerate==0.25.0" \
    "nvidia-cublas-cu11" \
    "nvidia-cudnn-cu11" \
    "requests" \
    && pip install --no-cache-dir "onnxruntime-gpu==1.17.1" --extra-index-url https://aiinfra.pkgs.visualstudio.com/Public/packages/onnxruntime-cuda-12

# Add nvidia-cublas-cu11 and nvidia-cudnn-cu11 to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/python3.10/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib

COPY main.py .

# Expose the LAN port
EXPOSE 8000

# Run the server
CMD ["python", "main.py"]
