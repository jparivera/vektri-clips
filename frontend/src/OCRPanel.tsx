import { useState } from "react";
import { post, seconds } from "./api";
import type { Project, OCRModel, OCRMode } from "./types";
import { Field, NumberField, Icon } from "./ui";
import { Jobs } from "./App";

export function OCRPanel({
  project: p,
  invoke,
  busy,
}: {
  project: Project;
  invoke: (fn: () => Promise<unknown>) => Promise<boolean>;
  busy: boolean;
}) {
  const [spacing, setSpacing] = useState(5);
  const [mode, setMode] = useState<OCRMode>("baseline");
  const [model, setModel] = useState<OCRModel>("ppocr-v4");
  const [qualitySamples, setQualitySamples] = useState(1);
  const running = p.jobs?.some(
    (j) => j.kind === "ocr" && ["queued", "running"].includes(j.status),
  );
  const result = p.ocr;
  const rows =
    result?.ocr.samples.flatMap((sample) =>
      sample.detections.map((line) => ({
        ...line,
        time: sample.time,
        frame: sample.frame_ref,
      })),
    ) ?? [];
  const artifact = result
    ? `/api/v1/artifacts/${encodeURIComponent(p.id)}/${result.artifact_dir}/`
    : "";
  return (
    <details className="info-note ocr-panel">
      <summary>Read on-screen text (OCR)</summary>
      <p className="ocr-lead">
        <Icon name="text" size={15} /> Extract Chinese or English text from
        sampled frames. Review errors and omissions; OCR does not change
        captions or select clips.
      </p>
      <Field label="OCR model">
        <select
          value={model}
          onChange={(e) => setModel(e.target.value as OCRModel)}
        >
          <option value="ppocr-v4">PP-OCRv4 (default)</option>
          <option value="ppocr-v5-mobile">
            PP-OCRv5 mobile (experimental)
          </option>
        </select>
      </Field>
      <Field label="OCR detail">
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value as OCRMode)}
        >
          <option value="baseline">Standard: reduced frames</option>
          <option value="detail">
            Detail: larger bounded frames and overlapping tiles
          </option>
        </select>
      </Field>
      <Field label="Nearby frames to compare for sharpness">
        <select
          value={qualitySamples}
          onChange={(e) => setQualitySamples(Number(e.target.value))}
        >
          <option value={1}>1 (requested frame)</option>
          <option value={2}>2 (up to 0.25 seconds nearby)</option>
          <option value={3}>3 (up to 0.25 seconds nearby)</option>
        </select>
      </Field>
      <p className="tiny subtle">
        Detail takes longer and is bounded by memory and evidence limits.
        Sharpness can select a different moment; every reading links to its
        actual frame time. Models must be downloaded separately.
      </p>
      <NumberField
        label="OCR sample spacing (seconds)"
        value={spacing}
        min={1}
        max={600}
        step={1}
        decimals={1}
        onCommit={setSpacing}
      />
      <button
        className="button secondary"
        disabled={
          busy ||
          running ||
          !Number.isFinite(spacing) ||
          spacing < 1 ||
          spacing > 600
        }
        onClick={() =>
          void invoke(() =>
            post(
              `/projects/${p.id}/ocr`,
              {
                ocr_sample_seconds: spacing,
                ocr_max_samples: 120,
                ocr_model: model,
                ocr_mode: mode,
                ocr_quality_samples: qualitySamples,
              },
              true,
            ),
          )
        }
      >
        {running ? "Reading text…" : "Extract on-screen text"}
      </button>
      <Jobs
        compact
        jobs={p.jobs?.filter((j) => j.kind === "ocr") ?? []}
        onAction={async (fn) => {
          await invoke(fn);
        }}
      />
      {result && (
        <>
          <p className="tiny subtle">
            Result:{" "}
            {result.ocr.configuration?.ocr_model ??
              result.ocr.model_id ??
              "ppocr-v4"}
            , {result.ocr.configuration?.ocr_mode ?? "baseline"}; up to{" "}
            {result.ocr.configuration?.ocr_quality_samples ?? 1} nearby frame(s)
            compared; OCR reads the selected frame.
          </p>
          {(result.ocr.sampling_complete === false ||
            result.ocr.raw_readings_truncated) && (
            <p role="status">
              Partial evidence: sampling or text-region limits were reached.
              Review the full report for omitted work.
            </p>
          )}
          <p>
            {result.ocr.text_count} text regions from {result.ocr.sample_count}{" "}
            frames (overlapping tile readings may repeat text). Largest sample
            gap: {seconds(result.ocr.maximum_sample_gap_seconds)}.
          </p>
          <p className="tiny subtle">{result.ocr.limitations.join(" ")}</p>
          <a className="text-link" href={artifact + "ocr.json"} download>
            <Icon name="download" size={14} /> Download full OCR evidence
          </a>
          <ul className="ocr-readings">
            {rows.slice(0, 50).map((line, index) => (
              <li key={index}>
                <a
                  className="mono"
                  href={artifact + line.frame}
                  target="_blank"
                  rel="noreferrer"
                >
                  {seconds(line.time)}
                </a>
                {" — "}
                {line.text}
              </li>
            ))}
          </ul>
          {rows.length > 50 && (
            <p>
              Showing the first 50 regions; download the evidence for all
              results.
            </p>
          )}
        </>
      )}
    </details>
  );
}
