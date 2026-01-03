# Orpheus TTS vLLM Server
# Uses official vLLM image + Orpheus TTS package
#
# Build: docker build -t orpheus-vllm -f deploy/Dockerfile .
# Run:   docker run --gpus all -p 8000:8000 orpheus-vllm

# Use vLLM 0.8.3 with V1 engine disabled for FP8 quantization support
# FP8 is critical for real-time performance (83+ tokens/s needed)
FROM vllm/vllm-openai:v0.8.3

WORKDIR /app

# Install dependencies
# - snac: Neural audio codec for decoding tokens to audio
# - fastapi/uvicorn: For the API server
RUN pip install --no-cache-dir snac fastapi uvicorn

# Copy the orpheus_tts package from the repo
COPY orpheus_tts_pypi/orpheus_tts /app/orpheus_tts
COPY orpheus_tts_pypi/setup.py /app/
COPY orpheus_tts_pypi/README.md /app/

# Install orpheus_tts as a package
RUN pip install -e .

# Copy the API server
COPY deploy/server.py /app/server.py

# Pre-download the model and SNAC codec on build (optional, speeds up cold start)
# Uncomment to bake model weights into the image (increases image size by ~6GB)
# RUN python3 -c "from orpheus_tts import OrpheusModel; OrpheusModel(model_name='canopylabs/orpheus-tts-0.1-finetune-prod')"
# RUN python3 -c "from snac import SNAC; SNAC.from_pretrained('hubertsiuzdak/snac_24khz')"

# Expose API port
EXPOSE 8000

# Environment variables
ENV SNAC_DEVICE=cuda
ENV HF_HOME=/root/.cache/huggingface
ENV CUDA_VISIBLE_DEVICES=0
# Disable V1 engine to avoid 30+ min cudagraph compilation
ENV VLLM_USE_V1=0
# Default to FP8 quantization for fast inference
ENV QUANTIZATION=fp8

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
ENTRYPOINT ["python3", "/app/server.py"]
