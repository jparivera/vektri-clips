import assert from "node:assert/strict";
import test from "node:test";
import { initialSpeechSettings } from "../frontend/src/analysisSettings.ts";
import type { Transcript } from "../frontend/src/types.ts";

const transcript = (settings: Partial<Transcript>): Transcript => ({
  segments: [],
  words: [],
  ...settings,
});

test("new projects retain English defaults", () => {
  assert.deepEqual(initialSpeechSettings(), {
    language: "en",
    model: "tiny.en",
    audioStream: 0,
  });
});
test("reopened Portuguese text keeps the model and selected audio track", () => {
  assert.deepEqual(
    initialSpeechSettings(
      transcript({
        language: "pt",
        model: "Systran/faster-whisper-tiny",
        audio_stream: 2,
      }),
    ),
    { language: "pt", model: "tiny", audioStream: 2 },
  );
});
test("language detection remains automatic after detecting Portuguese", () => {
  assert.deepEqual(
    initialSpeechSettings(
      transcript({
        language: "pt",
        configuration: { requested_model: "tiny", requested_language: "auto" },
      }),
    ),
    { language: "auto", model: "tiny", audioStream: 0 },
  );
});
test("English recognized with multilingual tiny does not silently switch models", () => {
  assert.equal(
    initialSpeechSettings(
      transcript({
        language: "en",
        configuration: { requested_model: "tiny", requested_language: "en" },
      }),
    ).model,
    "tiny",
  );
});
test("other detected languages use multilingual detection and invalid track values are rejected", () => {
  assert.deepEqual(
    initialSpeechSettings(transcript({ language: "fr", audio_stream: -1 })),
    { language: "auto", model: "tiny", audioStream: 0 },
  );
});
