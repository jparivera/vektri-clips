const BASE = "/api/v1";
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (typeof init.body === "string" && !headers.has("Content-Type"))
    headers.set("Content-Type", "application/json");
  const response = await fetch(BASE + path, { ...init, headers });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      const d = body.detail ?? body.error ?? body.message;
      message =
        typeof d === "string"
          ? d
          : Array.isArray(d)
            ? d
                .slice(0, 3)
                .map((x) =>
                  typeof x === "object" && x
                    ? `${(x.loc ?? []).filter((v: unknown) => v !== "body").join(".")}: ${x.msg ?? "Invalid value"}`
                    : String(x),
                )
                .join("; ") +
              (d.length > 3 ? ` (${d.length - 3} more validation errors)` : "")
            : JSON.stringify(d ?? body);
    } catch {
      /* retain HTTP error */
    }
    throw new ApiError(message, response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json();
}
export function post<T>(path: string, body?: unknown, idempotent = false) {
  return api<T>(path, {
    method: "POST",
    body: body === undefined ? undefined : JSON.stringify(body),
    headers: idempotent
      ? { "Idempotency-Key": crypto.randomUUID() }
      : undefined,
  });
}

/** Resume only a user-selected file with the same identity and import options. */
export async function resumableUpload(
  file: File,
  options: {
    title: string;
    genre: string;
    attribution: string;
    permission_status: string;
  },
  signal: AbortSignal,
  progress: (sent: number, total: number) => void,
): Promise<{ project_id: string }> {
  const identity = JSON.stringify([
    file.name,
    file.size,
    file.lastModified,
    options,
  ]);
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(identity),
  );
  const key =
    "vektri-upload-v1-" +
    Array.from(new Uint8Array(digest), (v) =>
      v.toString(16).padStart(2, "0"),
    ).join("");
  let id: string | null = null;
  try {
    id = localStorage.getItem(key);
  } catch {
    /* upload still works without local persistence */
  }
  let state: {
    id: string;
    offset: number;
    size: number;
    status?: string;
    project_id?: string;
  } | null = null;
  if (id) {
    try {
      state = await api("/uploads/" + encodeURIComponent(id), { signal });
    } catch (error) {
      if (!(error instanceof ApiError && error.status === 404)) throw error;
    }
  }
  if (!state) {
    state = await api("/uploads", {
      method: "POST",
      body: JSON.stringify({
        filename: file.name,
        size: file.size,
        ...options,
      }),
      signal,
    });
    try {
      localStorage.setItem(key, state!.id);
    } catch {
      /* no storage permission */
    }
  }
  if (state!.status === "complete" && state!.project_id)
    return { project_id: state!.project_id };
  progress(state!.offset, file.size);
  while (state!.offset < file.size) {
    signal.throwIfAborted();
    const offset = state!.offset;
    state = await api("/uploads/" + encodeURIComponent(state!.id), {
      method: "PATCH",
      body: file.slice(offset, offset + 4 * 1024 * 1024),
      headers: {
        "Content-Type": "application/octet-stream",
        "Upload-Offset": String(offset),
      },
      signal,
    });
    progress(state!.offset, file.size);
  }
  const result = await api<{ project_id: string }>(
    "/uploads/" + encodeURIComponent(state!.id) + "/complete",
    { method: "POST", signal },
  );
  try {
    localStorage.removeItem(key);
  } catch {
    /* no storage permission */
  }
  return result;
}
export function list<T>(value: unknown, key: string): T[] {
  if (Array.isArray(value)) return value as T[];
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return (
      Array.isArray(record[key])
        ? record[key]
        : Array.isArray(record.items)
          ? record.items
          : []
    ) as T[];
  }
  return [];
}
/** mm:ss.d, with an hour prefix only for sources that need it. */
export const seconds = (n = 0) => {
  const v = Math.max(0, Number.isFinite(n) ? n : 0);
  // Work in whole tenths so 13.2 never shows as 13.1 through float error.
  const tenths = Math.floor(v * 10 + 1e-6);
  const d = tenths % 10,
    s = Math.floor(tenths / 10) % 60,
    m = Math.floor(tenths / 600) % 60,
    h = Math.floor(tenths / 36000);
  const core = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${d}`;
  return h ? `${h}:${core}` : core;
};
export const bytes = (n?: number | null) =>
  n == null || !Number.isFinite(n)
    ? "—"
    : n >= 1073741824
      ? `${(n / 1073741824).toFixed(1)} GB`
      : n >= 1048576
        ? `${(n / 1048576).toFixed(1)} MB`
        : n >= 1024
          ? `${Math.round(n / 1024)} KB`
          : `${Math.round(n)} B`;
export function when(value?: string | number | null) {
  if (value == null || value === "") return "";
  const date = new Date(typeof value === "number" ? value * 1000 : value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}
export function asset(projectId: string, path: string) {
  return `${BASE}/artifacts/${encodeURIComponent(projectId)}/${path.split("/").map(encodeURIComponent).join("/")}`;
}
export function message(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}
