#!/usr/bin/env python3
"""
Orpheus TTS - OpenAI-compatible API Server

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
from typing import Optional
from dataclasses import dataclass

from flask import Flask, Response, request, jsonify
from orpheus_tts import OrpheusModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('orpheus-server')

# Flask app
app = Flask(__name__)

# Global engine instance (initialized on startup)
engine: Optional[OrpheusModel] = None

# Available voices in Orpheus TTS
AVAILABLE_VOICES = ["tara", "zoe", "zac", "jess", "leo", "mia", "julia", "leah"]
DEFAULT_VOICE = "tara"


@dataclass
class TTSConfig:
    """Configuration for TTS generation"""
    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 2000
    repetition_penalty: float = 1.1
    stop_token_ids: list = None

    def __post_init__(self):
        if self.stop_token_ids is None:
            self.stop_token_ids = [128258]


def create_wav_header(sample_rate: int = 24000, bits_per_sample: int = 16, channels: int = 1, data_size: int = 0) -> bytes:
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


def generate_speech(text: str, voice: str = DEFAULT_VOICE, config: TTSConfig = None) -> bytes:
    """Generate speech from text and return WAV bytes"""
    if config is None:
        config = TTSConfig()

    if voice not in AVAILABLE_VOICES:
        logger.warning(f"Unknown voice '{voice}', using default '{DEFAULT_VOICE}'")
        voice = DEFAULT_VOICE

    # Collect all audio chunks
    audio_chunks = []

    try:
        syn_tokens = engine.generate_speech(
            prompt=text,
            voice=voice,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            repetition_penalty=config.repetition_penalty,
            stop_token_ids=config.stop_token_ids
        )

        for chunk in syn_tokens:
            if chunk:
                audio_chunks.append(chunk)
    except Exception as e:
        logger.error(f"Speech generation failed: {e}")
        raise

    # Combine audio data
    audio_data = b''.join(audio_chunks)

    # Create complete WAV file
    wav_header = create_wav_header(data_size=len(audio_data))
    return wav_header + audio_data


def generate_speech_streaming(text: str, voice: str = DEFAULT_VOICE, config: TTSConfig = None):
    """Generate speech as a streaming response"""
    if config is None:
        config = TTSConfig()

    if voice not in AVAILABLE_VOICES:
        logger.warning(f"Unknown voice '{voice}', using default '{DEFAULT_VOICE}'")
        voice = DEFAULT_VOICE

    # Yield WAV header first (with unknown size - will be fixed by client)
    yield create_wav_header()

    try:
        syn_tokens = engine.generate_speech(
            prompt=text,
            voice=voice,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            repetition_penalty=config.repetition_penalty,
            stop_token_ids=config.stop_token_ids
        )

        for chunk in syn_tokens:
            if chunk:
                yield chunk
    except Exception as e:
        logger.error(f"Streaming speech generation failed: {e}")
        raise


# ============================================================================
# API Routes
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': os.environ.get('ORPHEUS_MODEL', 'canopylabs/orpheus-tts-0.1-finetune-prod'),
        'available_voices': AVAILABLE_VOICES
    })


@app.route('/v1/audio/speech', methods=['POST'])
def openai_speech():
    """
    OpenAI-compatible text-to-speech endpoint

    Request body:
    {
        "model": "orpheus-tts" (ignored, uses configured model),
        "input": "Text to convert to speech",
        "voice": "tara" (one of: tara, zoe, zac, jess, leo, mia, julia, leah),
        "response_format": "wav" (only wav supported currently),
        "speed": 1.0 (ignored)
    }

    Returns: audio/wav binary data
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        text = data.get('input', '')
        if not text:
            return jsonify({'error': 'No input text provided'}), 400

        voice = data.get('voice', DEFAULT_VOICE)
        response_format = data.get('response_format', 'wav')

        if response_format not in ['wav', 'pcm']:
            return jsonify({'error': f'Unsupported response format: {response_format}. Only wav and pcm supported.'}), 400

        logger.info(f"Generating speech: voice={voice}, text_len={len(text)}")
        start_time = time.time()

        # Check if streaming is requested
        stream = data.get('stream', False)

        if stream:
            return Response(
                generate_speech_streaming(text, voice),
                mimetype='audio/wav',
                headers={
                    'Transfer-Encoding': 'chunked',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            audio_data = generate_speech(text, voice)
            elapsed = time.time() - start_time
            logger.info(f"Generated {len(audio_data)} bytes in {elapsed:.2f}s")

            return Response(
                audio_data,
                mimetype='audio/wav',
                headers={
                    'Content-Length': str(len(audio_data)),
                    'X-Generation-Time': f'{elapsed:.2f}s'
                }
            )

    except Exception as e:
        logger.exception("Speech generation failed")
        return jsonify({'error': str(e)}), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models (OpenAI-compatible)"""
    return jsonify({
        'object': 'list',
        'data': [
            {
                'id': 'orpheus-tts',
                'object': 'model',
                'created': 1700000000,
                'owned_by': 'canopylabs'
            }
        ]
    })


@app.route('/v1/voices', methods=['GET'])
def list_voices():
    """List available voices"""
    return jsonify({
        'voices': AVAILABLE_VOICES,
        'default': DEFAULT_VOICE
    })


@app.route('/tts', methods=['GET', 'POST'])
def simple_tts():
    """
    Simple TTS endpoint for testing

    GET /tts?prompt=Hello&voice=tara
    POST /tts with JSON body: {"prompt": "Hello", "voice": "tara"}
    """
    if request.method == 'GET':
        text = request.args.get('prompt', 'Hello, this is a test of Orpheus TTS.')
        voice = request.args.get('voice', DEFAULT_VOICE)
    else:
        data = request.get_json() or {}
        text = data.get('prompt', 'Hello, this is a test of Orpheus TTS.')
        voice = data.get('voice', DEFAULT_VOICE)

    logger.info(f"Simple TTS: voice={voice}, text={text[:50]}...")

    return Response(
        generate_speech_streaming(text, voice),
        mimetype='audio/wav'
    )


# ============================================================================
# Server Initialization
# ============================================================================

def init_engine(model_name: str, max_model_len: int = 4096, gpu_memory_utilization: float = 0.9):
    """Initialize the Orpheus TTS engine"""
    global engine

    logger.info(f"Initializing Orpheus TTS engine with model: {model_name}")
    logger.info(f"  max_model_len: {max_model_len}")
    logger.info(f"  gpu_memory_utilization: {gpu_memory_utilization}")

    start_time = time.time()

    engine = OrpheusModel(
        model_name=model_name,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization
    )

    elapsed = time.time() - start_time
    logger.info(f"Engine initialized in {elapsed:.2f}s")

    return engine


def main():
    parser = argparse.ArgumentParser(description='Orpheus TTS API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--model', default=os.environ.get('ORPHEUS_MODEL', 'canopylabs/orpheus-tts-0.1-finetune-prod'),
                        help='Model name or path')
    parser.add_argument('--max-model-len', type=int, default=int(os.environ.get('MAX_MODEL_LEN', 4096)),
                        help='Maximum sequence length')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='GPU memory utilization (0-1)')

    args = parser.parse_args()

    # Initialize the engine
    init_engine(
        model_name=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Available voices: {', '.join(AVAILABLE_VOICES)}")

    # Run Flask server
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False
    )


if __name__ == '__main__':
    main()
