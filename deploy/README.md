# Orpheus TTS vLLM Deployment

OpenAI-compatible API server for [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) using vLLM for efficient inference.

## Quick Start

### Using Pre-built Image

```bash
# Pull from GitHub Container Registry
docker pull ghcr.io/bizyb/orpheus:latest

# Run with GPU support
docker run --gpus all -p 8000:8000 ghcr.io/bizyb/orpheus:latest
```

### Building Locally

```bash
# From repository root
docker build -t orpheus-vllm -f deploy/Dockerfile .

# Run
docker run --gpus all -p 8000:8000 orpheus-vllm
```

## API Endpoints

### OpenAI-Compatible Speech Endpoint

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "orpheus-tts",
    "input": "Hello, this is a test of Orpheus TTS!",
    "voice": "tara",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### Simple TTS Endpoint

```bash
# GET request
curl "http://localhost:8000/tts?prompt=Hello%20world&voice=tara" --output speech.wav

# POST request
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "voice": "tara"}' \
  --output speech.wav
```

### Health Check

```bash
curl http://localhost:8000/health
```

### List Voices

```bash
curl http://localhost:8000/v1/voices
```

## Available Voices

| Voice | Description |
|-------|-------------|
| `tara` | Default voice |
| `zoe` | Female voice |
| `zac` | Male voice |
| `jess` | Female voice |
| `leo` | Male voice |
| `mia` | Female voice |
| `julia` | Female voice |
| `leah` | Female voice |

## Emotional Tags

Orpheus TTS supports emotional tags in the input text:

- `<laugh>` - Laughter
- `<chuckle>` - Chuckle
- `<sigh>` - Sigh
- `<cough>` - Cough
- `<sniffle>` - Sniffle
- `<groan>` - Groan
- `<yawn>` - Yawn
- `<gasp>` - Gasp

Example:
```json
{
  "input": "Oh my goodness! <laugh> I can't believe it worked! <sigh> Finally.",
  "voice": "mia"
}
```

## RunPod Deployment

1. Create a new GPU pod on [RunPod](https://runpod.io)
2. Select a template with GPU support (e.g., RTX 4090, A100)
3. Use the Docker image: `ghcr.io/bizyb/orpheus:latest`
4. Expose port 8000
5. Set environment variables (optional):
   - `ORPHEUS_MODEL`: Model name (default: `canopylabs/orpheus-tts-0.1-finetune-prod`)
   - `MAX_MODEL_LEN`: Max sequence length (default: `4096`)
   - `SNAC_DEVICE`: Device for SNAC decoder (`cuda` or `cpu`)

### RunPod Template Settings

```
Container Image: ghcr.io/bizyb/orpheus:latest
Docker Command: (leave empty, uses ENTRYPOINT)
Exposed HTTP Ports: 8000
Volume Mount Path: /root/.cache/huggingface
Volume Size: 20GB (for model weights)
```

## Testing

```bash
# Test single generation
node deploy/test-single.mjs http://your-pod-url:8000

# Test concurrent generation
node deploy/test-concurrent.mjs http://your-pod-url:8000

# Test all voices
node deploy/test-voices.mjs http://your-pod-url:8000
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ORPHEUS_MODEL` | `canopylabs/orpheus-tts-0.1-finetune-prod` | HuggingFace model name |
| `MAX_MODEL_LEN` | `4096` | Maximum sequence length |
| `SNAC_DEVICE` | `cuda` | Device for SNAC decoder |
| `HF_HOME` | `/root/.cache/huggingface` | HuggingFace cache directory |

### Server Arguments

```bash
python server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model canopylabs/orpheus-tts-0.1-finetune-prod \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

## VRAM Requirements

| GPU | VRAM | Notes |
|-----|------|-------|
| RTX 4090 | 24GB | Full model in bfloat16 |
| RTX 3090 | 24GB | Full model in bfloat16 |
| A100 40GB | 40GB | Headroom for batching |
| A100 80GB | 80GB | Large batch sizes |

The model requires approximately 6-8GB VRAM for inference. The remaining VRAM is used for KV cache.

## Troubleshooting

### Model Download Slow

Pre-download the model before running:
```bash
huggingface-cli download canopylabs/orpheus-tts-0.1-finetune-prod
huggingface-cli download hubertsiuzdak/snac_24khz
```

### CUDA Out of Memory

Reduce `--gpu-memory-utilization`:
```bash
python server.py --gpu-memory-utilization 0.8
```

### Audio Quality Issues

- Adjust temperature (lower = more deterministic): `"temperature": 0.3`
- Adjust repetition penalty: `"repetition_penalty": 1.2`
- Use shorter input texts

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      API Server (Flask)                       │
│                                                               │
│  /v1/audio/speech ──► OrpheusModel ──► SNAC Decoder ──► WAV  │
│                           │                   │               │
│                      vLLM Engine          Neural Codec        │
│                      (Llama-3b)           (24kHz audio)       │
└─────────────────────────────────────────────────────────────┘
```

1. **Text Input**: Receives text and voice selection
2. **vLLM Engine**: Generates audio tokens using Llama-3b backbone
3. **SNAC Decoder**: Converts tokens to 24kHz audio
4. **WAV Output**: Returns audio as WAV file

## License

Apache-2.0 (follows Orpheus TTS license)
