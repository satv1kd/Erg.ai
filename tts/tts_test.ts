// Gemini-TTS via Application Default Credentials (no API key).
// Setup once: gcloud auth application-default login
//             gcloud auth application-default set-quota-project <project-id>
// Usage: node tts_test.ts ["text to speak"] [out.wav]
import { writeFile } from 'node:fs/promises';
import { TextToSpeechClient } from '@google-cloud/text-to-speech';

const text = process.argv[2] ?? 'test';
const outPath = process.argv[3] ?? 'output.wav';

const client = new TextToSpeechClient(); // uses ADC, no key
const [res] = await client.synthesizeSpeech({
  input: { text, prompt: 'Say the following naturally' },
  voice: { languageCode: 'en-US', name: 'Kore', modelName: 'gemini-2.5-flash-tts' },
  audioConfig: { audioEncoding: 'LINEAR16' },
});

if (!res.audioContent || res.audioContent.length === 0) {
  throw new Error('TTS returned no audio');
}
await writeFile(outPath, res.audioContent);
console.log(`Wrote ${res.audioContent.length} bytes to ${outPath}`);
