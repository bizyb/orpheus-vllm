#!/usr/bin/env node
/**
 * Test all available voices in Orpheus TTS
 *
 * Usage: node test-voices.mjs [server-url]
 * Example: node test-voices.mjs https://your-pod-url:8000
 */

import fs from 'fs';

const SERVER_URL = process.argv[2] || 'http://localhost:8000';

const text = "Hello! This is a demonstration of my voice. I hope you like it.";

async function generateSpeech(voice) {
  const start = Date.now();
  console.log(`[${voice}] Generating...`);

  try {
    const response = await fetch(`${SERVER_URL}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'orpheus-tts',
        input: text,
        voice: voice,
        response_format: 'wav'
      })
    });

    const elapsed = ((Date.now() - start) / 1000).toFixed(2);

    if (!response.ok) {
      console.error(`[${voice}] Failed: ${await response.text()}`);
      return null;
    }

    const audioBuffer = Buffer.from(await response.arrayBuffer());
    const filename = `voice-${voice}.wav`;
    fs.writeFileSync(filename, audioBuffer);
    console.log(`[${voice}] Saved: ${filename} (${(audioBuffer.length / 1024).toFixed(1)} KB) in ${elapsed}s`);

    return { voice, file: filename, elapsed: parseFloat(elapsed) };
  } catch (e) {
    console.error(`[${voice}] Error: ${e.message}`);
    return null;
  }
}

async function main() {
  console.log(`\nTesting all voices at ${SERVER_URL}\n`);

  // Get available voices
  let voices;
  try {
    const voicesRes = await fetch(`${SERVER_URL}/v1/voices`);
    if (!voicesRes.ok) {
      console.error('Failed to get voices list');
      return;
    }
    const voicesData = await voicesRes.json();
    voices = voicesData.voices;
    console.log(`Available voices: ${voices.join(', ')}\n`);
  } catch (e) {
    console.error(`Cannot connect to server: ${e.message}`);
    return;
  }

  console.log(`Generating "${text}"\n`);

  const totalStart = Date.now();

  // Generate sequentially to hear each voice clearly
  const results = [];
  for (const voice of voices) {
    const result = await generateSpeech(voice);
    if (result) results.push(result);
  }

  const totalElapsed = ((Date.now() - totalStart) / 1000).toFixed(2);

  console.log(`\n${'='.repeat(50)}`);
  console.log(`Generated ${results.length}/${voices.length} voices in ${totalElapsed}s`);
  console.log(`${'='.repeat(50)}\n`);
}

main().catch(console.error);
