#!/usr/bin/env python3
"""
Orpheus TTS - OpenAI-compatible API Server (FastAPI)

Provides an OpenAI-compatible /v1/audio/speech endpoint for text-to-speech generation.
Uses vLLM for efficient inference with continuous batching support.

Usage:
    python server.py [--host HOST] [--port PORT] [--model MODEL]

Environment variables:
    ORPHEUS_MODEL: Model name (default: canopylabs/orpheus-tts-0.1-finetune-prod)
    SNAC_DEVICE: Device for SNAC decoder (cuda/cpu, default: cuda)
    MAX_MODEL_LEN: Maximum sequence length (default: 4096)
"""

import os
import io
import struct
import time
import logging
import argparse
import asyncio
from typing import Optional, List, AsyncGenerator
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
from snac import SNAC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('orpheus-server')

# Available voices
AVAILABLE_VOICES = ["tara", "zoe", "zac", "jess", "leo", "mia", "julia", "leah"]
DEFAULT_VOICE = "tara"

# Global instances
engine: Optional[AsyncLLMEngine] = None
tokenizer: Optional[AutoTokenizer] = None
snac_model: Optional[SNAC] = None
snac_device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Pydantic Models
# =============================================================================

class SpeechRequest(BaseModel):
    model: str = "orpheus-tts"
    input: str = Field(..., description="Text to convert to speech")
    voice: str = Field(default=DEFAULT_VOICE, description="Voice to use")
    response_format: str = Field(default="wav", description="Audio format (wav or pcm)")
    speed: float = Field(default=1.0, description="Speed (ignored)")
    stream: bool = Field(default=False, description="Stream response")


class SimpleTTSRequest(BaseModel):
    prompt: str = Field(..., description="Text to convert to speech")
    voice: str = Field(default=DEFAULT_VOICE, description="Voice to use")


class HealthResponse(BaseModel):
    status: str
    model: str
    available_voices: List[str]


class VoicesResponse(BaseModel):
    voices: List[str]
    default: str


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "canopylabs"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# =============================================================================
# Audio Processing (from orpheus_tts decoder)
# =============================================================================

def convert_to_audio(multiframe: List[int], count: int) -> Optional[bytes]:
    """Convert token frames to audio bytes using SNAC decoder"""
    import numpy as np

    if len(multiframe) < 7:
        return None

    codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)

    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames * 7]

    for j in range(num_frames):
        i = 7 * j
        if codes_0.shape[0] == 0:
            codes_0 = torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)
        else:
            codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)])

        if codes_1.shape[0] == 0:
            codes_1 = torch.tensor([frame[i + 1]], device=snac_device, dtype=torch.int32)
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i + 4]], device=snac_device, dtype=torch.int32)])
        else:
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i + 1]], device=snac_device, dtype=torch.int32)])
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i + 4]], device=snac_device, dtype=torch.int32)])

        if codes_2.shape[0] == 0:
            codes_2 = torch.tensor([frame[i + 2]], device=snac_device, dtype=torch.int32)
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 3]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 5]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 6]], device=snac_device, dtype=torch.int32)])
        else:
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 2]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 3]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 5]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 6]], device=snac_device, dtype=torch.int32)])

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

    # Validate token range
    if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or
        torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or
        torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
        return None

    with torch.inference_mode():
        audio_hat = snac_model.decode(codes)

    audio_slice = audio_hat[:, :, 2048:4096]
    detached_audio = audio_slice.detach().cpu()
    audio_np = detached_audio.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    return audio_int16.tobytes()


def turn_token_into_id(token_string: str, index: int) -> Optional[int]:
    """Parse custom token from text to get audio code ID"""
    token_string = token_string.strip()
    last_token_start = token_string.rfind("<custom_token_")

    if last_token_start == -1:
        return None

    last_token = token_string[last_token_start:]

    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    return None


# =============================================================================
# Audio Generation
# =============================================================================

def format_prompt(text: str, voice: str) -> str:
    """Format text prompt with voice and special tokens"""
    if voice:
        adapted_prompt = f"{voice}: {text}"
    else:
        adapted_prompt = text

    prompt_tokens = tokenizer(adapted_prompt, return_tensors="pt")
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
    all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
    return tokenizer.decode(all_input_ids[0])


def create_wav_header(sample_rate: int = 24000, bits_per_sample: int = 16,
                      channels: int = 1, data_size: int = 0) -> bytes:
    """Create a WAV file header"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header


async def generate_speech_async(
    text: str,
    voice: str = DEFAULT_VOICE,
    temperature: float = 0.4,
    top_p: float = 0.9,
    max_tokens: int = 2000,
    repetition_penalty: float = 1.1
) -> AsyncGenerator[bytes, None]:
    """Generate speech as async generator of audio chunks"""

    if voice not in AVAILABLE_VOICES:
        logger.warning(f"Unknown voice '{voice}', using default '{DEFAULT_VOICE}'")
        voice = DEFAULT_VOICE

    prompt_string = format_prompt(text, voice)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_token_ids=[128258],
        repetition_penalty=repetition_penalty,
    )

    request_id = f"req-{time.time()}"

    buffer = []
    count = 0

    async for result in engine.generate(prompt=prompt_string, sampling_params=sampling_params, request_id=request_id):
        token_text = result.outputs[0].text
        token = turn_token_into_id(token_text, count)

        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples


async def generate_full_audio(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Generate complete audio and return as WAV bytes"""
    audio_chunks = []

    async for chunk in generate_speech_async(text, voice):
        audio_chunks.append(chunk)

    audio_data = b''.join(audio_chunks)
    wav_header = create_wav_header(data_size=len(audio_data))
    return wav_header + audio_data


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup"""
    global engine, tokenizer, snac_model, snac_device

    model_name = os.environ.get('ORPHEUS_MODEL', 'canopylabs/orpheus-tts-0.1-finetune-prod')
    max_model_len = int(os.environ.get('MAX_MODEL_LEN', 4096))
    gpu_memory_utilization = float(os.environ.get('GPU_MEMORY_UTILIZATION', 0.9))
    snac_device = os.environ.get('SNAC_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

    quantization = os.environ.get('QUANTIZATION', 'fp8')

    logger.info(f"Initializing Orpheus TTS server")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Max model len: {max_model_len}")
    logger.info(f"  GPU memory utilization: {gpu_memory_utilization}")
    logger.info(f"  Quantization: {quantization}")
    logger.info(f"  SNAC device: {snac_device}")

    # Initialize vLLM engine with FP8 quantization for fast inference
    # FP8 is critical for real-time performance (need 83+ tokens/s)
    engine_args = AsyncEngineArgs(
        model=model_name,
        dtype="auto",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        quantization=quantization if quantization != 'none' else None,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('canopylabs/orpheus-3b-0.1-pretrained')

    # Initialize SNAC model
    logger.info("Loading SNAC model...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(snac_device)

    logger.info("Server ready!")

    yield

    # Cleanup
    logger.info("Shutting down...")


app = FastAPI(
    title="Orpheus TTS API",
    description="OpenAI-compatible text-to-speech API using Orpheus TTS and vLLM",
    version="1.0.0",
    lifespan=lifespan
)


# =============================================================================
# Routes
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model=os.environ.get('ORPHEUS_MODEL', 'canopylabs/orpheus-tts-0.1-finetune-prod'),
        available_voices=AVAILABLE_VOICES
    )


@app.post("/v1/audio/speech")
async def openai_speech(request: SpeechRequest):
    """
    OpenAI-compatible text-to-speech endpoint

    Returns audio as WAV binary data.
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="No input text provided")

    if request.response_format not in ['wav', 'pcm']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response format: {request.response_format}. Only wav and pcm supported."
        )

    logger.info(f"Generating speech: voice={request.voice}, text_len={len(request.input)}")
    start_time = time.time()

    if request.stream:
        async def stream_audio():
            yield create_wav_header()
            async for chunk in generate_speech_async(request.input, request.voice):
                yield chunk

        return StreamingResponse(
            stream_audio(),
            media_type="audio/wav",
            headers={
                "Transfer-Encoding": "chunked",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        audio_data = await generate_full_audio(request.input, request.voice)
        elapsed = time.time() - start_time
        logger.info(f"Generated {len(audio_data)} bytes in {elapsed:.2f}s")

        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Length": str(len(audio_data)),
                "X-Generation-Time": f"{elapsed:.2f}s"
            }
        )


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models (OpenAI-compatible)"""
    return ModelsResponse(
        data=[ModelInfo(id="orpheus-tts")]
    )


@app.get("/v1/voices", response_model=VoicesResponse)
async def list_voices():
    """List available voices"""
    return VoicesResponse(voices=AVAILABLE_VOICES, default=DEFAULT_VOICE)


@app.get("/tts")
@app.post("/tts")
async def simple_tts(prompt: str = "Hello, this is a test.", voice: str = DEFAULT_VOICE):
    """Simple TTS endpoint for testing"""
    logger.info(f"Simple TTS: voice={voice}, text={prompt[:50]}...")

    async def stream_audio():
        yield create_wav_header()
        async for chunk in generate_speech_async(prompt, voice):
            yield chunk

    return StreamingResponse(stream_audio(), media_type="audio/wav")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Orpheus TTS API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--model', default=None, help='Model name (sets ORPHEUS_MODEL env)')
    parser.add_argument('--max-model-len', type=int, default=None, help='Max sequence length')
    parser.add_argument('--gpu-memory-utilization', type=float, default=None, help='GPU memory util')

    args = parser.parse_args()

    # Set environment variables from args
    if args.model:
        os.environ['ORPHEUS_MODEL'] = args.model
    if args.max_model_len:
        os.environ['MAX_MODEL_LEN'] = str(args.max_model_len)
    if args.gpu_memory_utilization:
        os.environ['GPU_MEMORY_UTILIZATION'] = str(args.gpu_memory_utilization)

    logger.info(f"Starting server on {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
