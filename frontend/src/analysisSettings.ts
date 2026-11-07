import type { Transcript } from "./types";

/** Reopening a project must preserve the ASR settings that produced its text. */
export function initialSpeechSettings(transcript?: Transcript) {
  const requested = transcript?.configuration?.requested_language;
  const savedLanguage = requested ?? transcript?.language ?? "en";
  const language: "en" | "pt" | "auto" =
    savedLanguage === "en" || savedLanguage === "pt" || savedLanguage === "auto"
      ? savedLanguage
      : "auto";
  const savedModel =
    transcript?.configuration?.requested_model ?? transcript?.model;
  const model: "tiny.en" | "tiny" =
    language !== "en" ||
    savedModel === "tiny" ||
    savedModel === "Systran/faster-whisper-tiny"
      ? "tiny"
      : "tiny.en";
  const stream = transcript?.audio_stream;
  return {
    language,
    model,
    audioStream:
      typeof stream === "number" &&
      Number.isInteger(stream) &&
      stream >= 0 &&
      stream <= 31
        ? stream
        : 0,
  };
}
