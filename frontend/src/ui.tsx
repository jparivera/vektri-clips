import {
  useEffect,
  useRef,
  useId,
  useState,
  cloneElement,
  isValidElement,
  type ReactElement,
  type ReactNode,
} from "react";
const paths: Record<string, string> = {
  library: "M3 5h18v14H3z M3 9h18 M7 5v14",
  plus: "M12 5v14 M5 12h14",
  search: "M21 21l-5-5 M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0",
  play: "m8 5 12 7-12 7z",
  pause: "M8 5v14 M16 5v14",
  arrow: "M5 12h14 m-5-5 5 5-5 5",
  back: "M19 12H5 m5-5-5 5 5 5",
  download: "M12 3v12 m-5-5 5 5 5-5 M4 16v5h16v-5",
  upload: "M12 16V4 m-5 5 5-5 5 5 M4 16v5h16v-5",
  close: "m6 6 12 12 M18 6 6 18",
  check: "m5 12 4 4L19 6",
  cut: "m8 8 12 12 M8 16 20 4 M9 5a3 3 0 1 1-6 0 3 3 0 0 1 6 0 M9 19a3 3 0 1 1-6 0 3 3 0 0 1 6 0",
  film: "M3 3h18v18H3z M7 3v18 M17 3v18 M3 8h4 M3 16h4 M17 8h4 M17 16h4",
  spark: "m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5z",
  queue: "M8 6h13 M8 12h13 M8 18h13 M3 6h1 M3 12h1 M3 18h1",
  settings:
    "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8 M12 2v3 M12 19v3 M2 12h3 M19 12h3 M5 5l2 2 M17 17l2 2 M19 5l-2 2 M7 17l-2 2",
  clock: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18 M12 7v5l3 2",
  undo: "M4 10h9a6 6 0 0 1 0 12 M8 5l-5 5 5 5",
  redo: "M20 10h-9a6 6 0 0 0 0 12 M16 5l5 5-5 5",
  trash: "M3 6h18 M9 6V3h6v3 M6 6l1 15h10l1-15 M10 10v7 M14 10v7",
  chevron: "m9 5 7 7-7 7",
  "chevron-left": "m15 5-7 7 7 7",
  "chevron-down": "m5 9 7 7 7-7",
  file: "M5 2h9l5 5v15H5z M14 2v6h5",
  volume: "M3 9h4l5-4v14l-5-4H3z M16 8a6 6 0 0 1 0 8 M19 5a10 10 0 0 1 0 14",
  keyboard: "M2 5h20v14H2z M6 9h1 M11 9h1 M16 9h1 M6 14h12",
  eye: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12 M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6",
  warning: "M12 3 2 21h20L12 3z M12 10v5 M12 18v.5",
  info: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18 M12 11v6 M12 8v.5",
  text: "M4 6h16 M12 6v14 M8 20h8",
  image: "M3 5h18v14H3z M3 15l5-5 4 4 3-3 6 6 M16 9h.5",
  tag: "M3 12V3h9l9 9-9 9z M7 7h.5",
  external: "M14 4h6v6 M20 4 10 14 M18 14v6H4V6h6",
  refresh: "M20 12a8 8 0 1 1-2.3-5.7 M20 4v5h-5",
  dot: "M12 10a2 2 0 1 0 0 4 2 2 0 0 0 0-4",
  drive: "M3 5h18v6H3z M3 13h18v6H3z M6.5 8h.5 M6.5 16h.5",
  history: "M3 12a9 9 0 1 0 2.6-6.4 M3 3v5h5 M12 7v5l3 2",
  grip: "M9 6h.5 M15 6h.5 M9 12h.5 M15 12h.5 M9 18h.5 M15 18h.5",
  sun: "M12 3v2 M12 19v2 M3 12h2 M19 12h2 M5.6 5.6l1.4 1.4 M17 17l1.4 1.4 M5.6 18.4 7 17 M17 7l1.4-1.4 M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8",
  moon: "M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z",
  monitor: "M3 5h18v11H3z M8 20h8 M12 16v4",
  layers: "m12 3 9 5-9 5-9-5 9-5z M3 13l9 5 9-5",
};
export function Icon({ name, size = 18 }: { name: string; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.65"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={paths[name] ?? paths.film} />
    </svg>
  );
}
export function Badge({
  children,
  tone = "muted",
}: {
  children: ReactNode;
  tone?: string;
}) {
  return <span className={`badge ${tone}`}>{children}</span>;
}
export function Empty({
  icon = "film",
  title,
  children,
  action,
}: {
  icon?: string;
  title: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <div className="empty-icon">
        <Icon name={icon} size={28} />
      </div>
      <h2>{title}</h2>
      <p>{children}</p>
      {action}
    </div>
  );
}
export function Modal({
  title,
  children,
  onClose,
  wide = false,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  useEffect(() => {
    ref.current?.showModal();
    const dialog = ref.current;
    return () => dialog?.close();
  }, []);
  return (
    <dialog
      ref={ref}
      className={wide ? "wide" : undefined}
      aria-labelledby={titleId}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => {
        // A click on the backdrop lands on the dialog element itself, never on its content.
        if (e.target === ref.current) onClose();
      }}
    >
      <div className="modal-frame">
        <header className="modal-header">
          <h2 id={titleId}>{title}</h2>
          <button
            className="icon-btn"
            aria-label="Close dialog"
            onClick={onClose}
          >
            <Icon name="close" />
          </button>
        </header>
        {children}
      </div>
    </dialog>
  );
}
export function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  const hintId = useId();
  return (
    <label className="field">
      <span>{label}</span>
      {isValidElement(children)
        ? cloneElement(children as ReactElement<Record<string, unknown>>, {
            "aria-label": label,
            "aria-describedby": hint ? hintId : undefined,
          })
        : children}
      {hint && <small id={hintId}>{hint}</small>}
    </label>
  );
}
/**
 * A number input that only commits finite, in-range values. Clearing the box
 * or typing an intermediate value never writes NaN/0 into the edit, and the
 * displayed text snaps back to the committed value on blur.
 */
export function NumberField({
  label,
  value,
  onCommit,
  min,
  max,
  step,
  hint,
  decimals = 3,
  validate,
}: {
  label: string;
  value: number;
  onCommit: (n: number) => void;
  min?: number;
  max?: number;
  step?: number;
  hint?: string;
  decimals?: number;
  validate?: (n: number) => boolean;
}) {
  const display = String(+Number(value).toFixed(decimals));
  const [text, setText] = useState(display);
  const [focused, setFocused] = useState(false);
  const [invalid, setInvalid] = useState(false);
  useEffect(() => {
    if (!focused) {
      setText(display);
      setInvalid(false);
    }
  }, [display, focused]);
  const accepts = (n: number) =>
    Number.isFinite(n) &&
    (min === undefined || n >= min - 1e-9) &&
    (max === undefined || n <= max + 1e-9) &&
    (!validate || validate(n));
  return (
    <Field label={label} hint={hint}>
      <input
        type="number"
        inputMode="decimal"
        min={min}
        max={max}
        step={step}
        value={text}
        aria-invalid={invalid || undefined}
        className={invalid ? "invalid" : undefined}
        onChange={(e) => {
          const raw = e.target.value;
          setText(raw);
          const n = Number(raw);
          const ok = raw.trim() !== "" && accepts(n);
          setInvalid(!ok);
          if (ok && n !== value) onCommit(n);
        }}
        onFocus={() => setFocused(true)}
        onBlur={() => {
          setFocused(false);
          setText(display);
          setInvalid(false);
        }}
      />
    </Field>
  );
}
export function ErrorBox({
  error,
  onDismiss,
}: {
  error: string;
  onDismiss?: () => void;
}) {
  return (
    <div className="error-box" role="alert">
      <Icon name="warning" size={16} />
      <span>{error}</span>
      {onDismiss && (
        <button aria-label="Dismiss error" onClick={onDismiss}>
          <Icon name="close" />
        </button>
      )}
    </div>
  );
}
/** Human labels for job kinds and statuses; raw identifiers stay in evidence. */
export function jobLabel(kind: string) {
  const labels: Record<string, string> = {
    import: "Import",
    "import-url": "URL import",
    analyze: "Analysis",
    transcribe: "Transcription",
    render: "Render",
    track: "Tracking",
    ocr: "On-screen text",
    "model-download": "Model download",
  };
  return labels[kind] ?? kind.replace(/[_-]/g, " ");
}
export function jobIcon(kind: string) {
  return kind.includes("render")
    ? "download"
    : kind.includes("analy")
      ? "spark"
      : kind === "transcribe"
        ? "text"
        : kind === "track"
          ? "eye"
          : kind === "ocr"
            ? "text"
            : kind === "model-download"
              ? "download"
              : "film";
}
export function statusTone(status: string) {
  return status === "succeeded"
    ? "success"
    : status === "failed"
      ? "danger"
      : status === "running"
        ? "accent"
        : status === "cancelled"
          ? "warn"
          : "muted";
}
