export type Crop = {
  mode: "fit" | "crop" | "split";
  x: number;
  y: number;
  zoom: number;
  second_x: number;
  keyframes: { time: number; x: number; y: number }[];
};
export type Segment = {
  id: string;
  start: number;
  end: number;
  crop: Crop;
  gain_db: number;
};
export type Word = { start: number; end: number; word: string };
export type Timeline = {
  schema_version: number;
  segments: Segment[];
  output: { width: number; height: number; fps: number; crf: number };
  captions: {
    enabled: boolean;
    burn_in: boolean;
    style: string;
    font_size: number;
    font_family?: "DejaVu Sans" | "DejaVu Serif" | "DejaVu Sans Mono";
    color?: string;
    highlight_color?: string;
    stroke_width?: number;
    margin: number;
    alignment: number;
    background: boolean;
    highlight: boolean;
    words: Word[];
  };
  audio: { normalize: boolean; fade: number; stream_index?: number };
};
export type Transcript = {
  segments: { start: number; end: number; text: string }[];
  words: Word[];
  alignment?: string;
  provider?: string;
  model?: string;
  language?: string | null;
  audio_stream?: number;
  configuration?: { requested_model?: string; requested_language?: string };
  status?: string;
  limitations?: string[];
  segment_text_corrected?: boolean;
  revision?: number;
};
export type ReviewObservation = {
  status: "observed" | "unknown";
  summary: string;
  evidence: Record<string, unknown>[];
};
export type CandidateReview = {
  schema_version: 1;
  method: "deterministic_evidence_review";
  status: "needs_human_review";
  fingerprint: string;
  source_sha256: string;
  observations: Record<"hook" | "context" | "payoff", ReviewObservation> & {
    signals?: ReviewObservation;
  };
  playback: {
    start: number;
    end: number;
    context_start: number;
    context_end: number;
  };
  frames: {
    role: string;
    time: number;
    approximate_seek: boolean;
    path?: string | null;
    status: string;
  }[];
  warnings: string[];
  limitations: string[];
};
export type Candidate = {
  id: string;
  start: number;
  end: number;
  title: string;
  rationale: string;
  review?: CandidateReview;
  confidence?: number;
  uncertainty?: string | string[];
  evidence?: unknown[];
  scores?: Record<string, number>;
  method?: string;
  decision?: string;
};
export type Job = {
  id: string;
  project_id?: string;
  kind: string;
  status: string;
  stage: string;
  created_at: string;
  updated_at: string;
  started_at?: string;
  elapsed_seconds?: number;
  attempts: number;
  max_attempts: number;
  error?: unknown;
  result?: Record<string, unknown>;
  params?: { revision?: number; [key: string]: unknown };
  cancel_requested?: boolean;
};
export type Export = {
  id: string;
  revision: number;
  created_at?: string | number;
  video_url?: string;
  thumbnail_url?: string;
  download_url?: string;
  package_url?: string | null;
  srt_url?: string | null;
  vtt_url?: string | null;
  clean_url?: string | null;
  project_id?: string;
  project_title?: string;
  manifest?: Record<string, unknown>;
  artifact_path?: string;
  path?: string;
  [key: string]: unknown;
};
export type OCRModel = "ppocr-v4" | "ppocr-v5-mobile";
export type OCRMode = "baseline" | "detail";
export type OCRResult = {
  artifact_dir: string;
  source_sha256: string;
  ocr: {
    model_id?: OCRModel;
    configuration?: {
      ocr_model: OCRModel;
      ocr_mode: OCRMode;
      ocr_quality_samples: number;
    };
    sampling_complete?: boolean;
    raw_readings_truncated?: boolean;
    temporal_agreement?: { links: unknown[]; truncated: boolean };
    sample_count: number;
    text_count: number;
    maximum_sample_gap_seconds: number;
    limitations: string[];
    samples: {
      time: number;
      frame_ref: string;
      detections: { text: string; model_score: number }[];
    }[];
  };
};
export type Project = {
  id: string;
  title: string;
  tags: string[];
  genre: string;
  status: string;
  revision: number;
  created_at?: string;
  updated_at?: string;
  duration?: number;
  thumbnail_url?: string;
  source?: {
    duration?: number;
    width?: number;
    height?: number;
    fps?: number;
    rotation?: number;
    size?: number;
    audio?: boolean;
    sha256?: string;
    original_name?: string;
    proxy_url?: string;
    thumbnail_url?: string;
    waveform_url?: string;
    url?: string;
    provenance?: Record<string, unknown>;
    metadata?: Record<string, unknown>;
    [key: string]: unknown;
  };
  timeline: Timeline;
  transcript?: Transcript;
  ocr?: OCRResult | null;
  artifact_base?: string;
  analysis?: {
    artifact_dir?: string;
    configuration?: {
      query?: string;
      visual_search?: boolean;
      visual_model?: string;
      audio_stream?: number;
      impact_detection?: boolean;
      impact_low_hz?: number;
      impact_high_hz?: number;
      impact_threshold?: number;
      impact_min_gap?: number;
      impact_lead?: number;
      impact_follow?: number;
    };
    candidates: Candidate[];
    provider?: string;
    model?: string;
    limitations?: string[];
  };
  exports: Export[];
  jobs: Job[];
  feedback?: {
    candidate_id: string;
    decision: "accept" | "reject";
    note: string;
  }[];
  [key: string]: unknown;
};
export type Fixture = {
  id?: string;
  name?: string;
  filename?: string;
  genre?: string;
  duration?: number;
  thumbnail_url?: string;
  [key: string]: unknown;
};
export type Preset = { id: string; name: string; timeline: Timeline };
export type Health = {
  status: string;
  worker_active?: boolean;
  queue_length?: number;
  worker?: {
    alive: boolean;
    state: string;
    workers: {
      pid: number;
      state: string;
      job_id?: string | null;
      available_ram_bytes?: number | null;
      minimum_available_ram_bytes?: number;
    }[];
    minimum_available_ram_bytes?: number;
  };
};
export type PublicationPreflight = {
  can_publish: boolean;
  manual_handoff_ready: boolean;
  manual_blockers: string[];
  remote_blockers: string[];
  warnings: string[];
  limitations: string[];
  current_file_integrity_verified?: boolean;
  evaluated_at_unix?: number;
  [key: string]: unknown;
};
export type Publication = {
  id: string;
  project_id: string;
  export_id: string;
  destination: string;
  account: string;
  caption: string;
  visibility: string;
  schedule?: string | null;
  timezone?: string;
  status: string;
  revision?: number;
  asset_fingerprint?: string;
  draft?: Record<string, unknown>;
  approval?: Record<string, unknown> | null;
  prerequisite?: string;
  [key: string]: unknown;
};
export type Revision = {
  project_id: string;
  revision: number;
  timeline: Timeline;
  fingerprint: string;
  created_at: number;
};
export type StorageSummary = {
  data_directory: string;
  generated_at: number;
  disk: { total: number; used: number; free: number } | null;
  directories: Record<string, number>;
  other_directories: Record<string, number>;
  database_bytes: number;
  file_count: number;
  symlinks_skipped: number;
  complete: boolean;
  policy: string;
};
export type InventoryTotals = {
  file_count: number;
  logical_bytes: number;
  unique_inode_bytes: number;
  allocated_inode_bytes: number;
};
export type StoragePlan = {
  schema_version: number;
  data_directory: string;
  generated_at: number;
  read_only: boolean;
  automatic_deletion_supported: boolean;
  status: string;
  maintenance_required: boolean;
  maintenance_reasons: string[];
  blockers: string[];
  settings: Record<string, number>;
  inventory: Record<string, InventoryTotals>;
  inventory_complete: boolean;
  protected_paths: { path: string; reasons: string[] }[];
  retained_caches: { path: string; reason: string }[];
  review_candidates: {
    path: string;
    kind: string;
    reason: string;
    file_count: number;
    logical_bytes: number;
    metadata_fingerprint: string;
  }[];
  estimated_reclaimable_bytes: number;
  snapshot_fingerprint: string | null;
  limitations: string[];
  plan_fingerprint: string;
  worker?: { alive: boolean; uncertain: boolean };
};
export type PlanVerification = {
  valid: boolean;
  read_only: boolean;
  automatic_deletion_supported: boolean;
  reasons: string[];
  current_fingerprint: string | null;
  maintenance_required: boolean;
  maintenance_reasons: string[];
  meaning: string;
};
export type AuditEntry = {
  seq: number;
  project_id: string | null;
  event: string;
  detail: Record<string, unknown>;
  created_at: number;
};
