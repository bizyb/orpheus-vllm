#!/usr/bin/env node
/**
 * Test single audio generation with Orpheus TTS
 *
 * Usage: node test-single.mjs [server-url]
 * Example: node test-single.mjs https://your-pod-url:8000
 */

import fs from 'fs';

const SERVER_URL = process.argv[2] || 'http://localhost:8000';

async function main() {
  console.log(`\nTesting Orpheus TTS at ${SERVER_URL}\n`);

  // Check health first
  try {
    const healthRes = await fetch(`${SERVER_URL}/health`);
    if (!healthRes.ok) {
      console.error(`Health check failed: ${healthRes.status}`);
      return;
    }
    const health = await healthRes.json();
    console.log('Health:', JSON.stringify(health, null, 2));
  } catch (e) {
    console.error(`Cannot connect to server: ${e.message}`);
    return;
  }

  console.log('\n--- Generating speech ---\n');

  const start = Date.now();

  const response = await fetch(`${SERVER_URL}/v1/audio/speech`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'orpheus-tts',
      input: 'Hello and welcome to Orpheus, the open source text to speech system built on Llama. This is a test of the API server.',
      voice: 'tara',
      response_format: 'wav'
    })
  });

  const elapsed = ((Date.now() - start) / 1000).toFixed(2);

  if (!response.ok) {
    console.error(`Failed: ${await response.text()}`);
    return;
  }

  const audioBuffer = Buffer.from(await response.arrayBuffer());
  const filename = 'output-single.wav';
  fs.writeFileSync(filename, audioBuffer);

  console.log(`Saved: ${filename}`);
  console.log(`Size: ${(audioBuffer.length / 1024).toFixed(1)} KB`);
  console.log(`Time: ${elapsed}s`);
  console.log(`Generation time header: ${response.headers.get('X-Generation-Time') || 'N/A'}`);
}

main().catch(console.error);
