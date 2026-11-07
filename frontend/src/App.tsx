import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import {
  api,
  post,
  list,
  message,
  seconds,
  bytes,
  when,
  resumableUpload,
} from "./api";
import { createRequestScope } from "./requestScope";
import { submitSelectedBatch, retainUnqueuedSelection } from "./batchSelection";
import type {
  Project,
  Job,
  Fixture,
  Publication,
  PublicationPreflight,
  Export,
  Health,
  StorageSummary,
  StoragePlan,
  PlanVerification,
  AuditEntry,
} from "./types";
import {
  Icon,
  Badge,
  Empty,
  Modal,
  Field,
  ErrorBox,
  jobLabel,
  jobIcon,
  statusTone,
} from "./ui";
import { Editor } from "./Editor";
import { Analysis } from "./Analysis";
import { AUDIT_GROUPS, auditSummary, auditTone, shortId } from "./audit";
import { THEMES, applyTheme, nextTheme, readTheme, type Theme } from "./theme";

export function thumb(p: Project) {
  return p.thumbnail_url || p.source?.thumbnail_url || undefined;
}
export function video(p: Project) {
  return (
    String(p.source?.preview_url ?? p.source?.proxy_url ?? "") ||
    String(p.proxy_url ?? "") ||
    undefined
  );
}
function normalizeProject(p: Project): Project {
  return {
    ...p,
    source: { ...p.source, ...p.source?.metadata },
    jobs: p.jobs ?? [],
    exports: p.exports ?? [],
    tags: p.tags ?? [],
  };
}
const PAGES = [
  ["library", "library", "Library"],
  ["exports", "download", "Exports"],
  ["jobs", "queue", "Job queue"],
  ["publishing", "upload", "Publishing"],
  ["models", "spark", "AI & models"],
  ["storage", "drive", "Storage"],
  ["audit", "history", "Audit trail"],
] as const;
type Page = (typeof PAGES)[number][0];
const TABS = [
  ["source", "01", "Source"],
  ["analysis", "02", "Analysis & candidates"],
  ["editor", "03", "Editor"],
  ["export", "04", "Export"],
] as const;
const GENRES: Record<string, string> = {
  speech: "Podcast / interview",
  tutorial: "Tutorial / screen",
  action: "Action / nonverbal",
  mixed: "Mixed footage",
};
const ACTIVE = ["running", "queued"];

/** The URL is the navigation state: ?project=ID[&tab=editor] or ?page=jobs. */
function readLocation(): { page: Page; projectId: string | null; tab: string } {
  const params = new URLSearchParams(location.search);
  const page = params.get("page") ?? "library";
  const tab = params.get("tab") ?? "source";
  return {
    page: PAGES.some(([id]) => id === page) ? (page as Page) : "library",
    projectId: params.get("project"),
    tab: TABS.some(([id]) => id === tab) ? tab : "source",
  };
}
function locationFor(page: Page, projectId: string | null, tab: string) {
  const params = new URLSearchParams();
  if (projectId) {
    params.set("project", projectId);
    if (tab !== "source") params.set("tab", tab);
  } else if (page !== "library") params.set("page", page);
  const query = params.toString();
  return query ? "?" + query : "/";
}

export default function App() {
  const initial = useMemo(readLocation, []);
  const [page, setPage] = useState<Page>(initial.page),
    [projectId, setProjectId] = useState<string | null>(initial.projectId),
    [tab, setTab] = useState(initial.tab),
    [projects, setProjects] = useState<Project[]>([]),
    [jobs, setJobs] = useState<Job[]>([]),
    [allExports, setAllExports] = useState<Export[]>([]),
    [health, setHealth] = useState<Health | null>(null),
    [project, setProject] = useState<Project | null>(null),
    [search, setSearch] = useState(""),
    [importOpen, setImportOpen] = useState(false),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(true),
    [selected, setSelected] = useState<Set<string>>(new Set()),
    [notice, setNotice] = useState(""),
    [online, setOnline] = useState(true),
    [theme, setTheme] = useState<Theme>(readTheme);
  const themeLabelId = useId();
  useEffect(() => {
    applyTheme(theme);
    if (theme !== "system") return;
    // Following the system: keep the browser chrome colour in step when it flips.
    const media = matchMedia("(prefers-color-scheme: light)");
    const sync = () => applyTheme("system");
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, [theme]);
  const refresh = useCallback(async () => {
    try {
      const [ps, js, es, hs] = await Promise.all([
        api<unknown>("/projects?q=" + encodeURIComponent(search)),
        api<unknown>("/jobs"),
        api<unknown>("/exports"),
        api<unknown>("/health").catch(() => null),
      ]);
      const exportList = list<Export>(es, "exports");
      setProjects(
        list<Project>(ps, "projects").map((p) => ({
          ...p,
          exports: exportList.filter((e) => e.project_id === p.id),
        })),
      );
      setJobs(list<Job>(js, "jobs"));
      setAllExports(exportList);
      setHealth(
        hs && typeof hs === "object" && !Array.isArray(hs)
          ? (hs as Health)
          : null,
      );
      setOnline(true);
    } catch {
      setOnline(false);
    } finally {
      setLoading(false);
    }
  }, [search]);
  const refreshProject = useCallback(async () => {
    if (!projectId) return;
    try {
      setProject(
        normalizeProject(await api<Project>("/projects/" + projectId)),
      );
    } catch (e) {
      setError(message(e));
    }
  }, [projectId]);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  useEffect(() => {
    void refreshProject();
  }, [refreshProject]);
  useEffect(() => {
    const tick = () => {
      if (document.hidden) return;
      void refresh();
      void refreshProject();
    };
    const timer = setInterval(tick, 3000);
    const visible = () => {
      if (!document.hidden) tick();
    };
    document.addEventListener("visibilitychange", visible);
    return () => {
      clearInterval(timer);
      document.removeEventListener("visibilitychange", visible);
    };
  }, [refresh, refreshProject]);
  useEffect(() => {
    const restore = () => {
      const next = readLocation();
      setPage(next.page);
      setProjectId(next.projectId);
      setTab(next.tab);
      if (!next.projectId) setProject(null);
    };
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, []);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 7000);
    return () => clearTimeout(timer);
  }, [notice]);
  const navigate = useCallback(
    (
      next: { page?: Page; projectId?: string | null; tab?: string },
      replace = false,
    ) => {
      const nextPage = next.page ?? "library",
        nextProject = next.projectId ?? null,
        nextTab = next.tab ?? "source";
      const url = locationFor(nextPage, nextProject, nextTab);
      if (replace) history.replaceState({}, "", url);
      else history.pushState({}, "", url);
      setPage(nextPage);
      setProjectId(nextProject);
      setTab(nextTab);
      if (!nextProject) setProject(null);
    },
    [],
  );
  const openProject = (id: string) => navigate({ projectId: id });
  const leaveProject = () => navigate({ page: "library" });
  const action = async (fn: () => Promise<unknown>, success?: string) => {
    try {
      await fn();
      if (success) setNotice(success);
      await refresh();
      await refreshProject();
    } catch (e) {
      setError(message(e));
    }
  };
  const active = jobs.filter((j) => ACTIVE.includes(j.status));
  const current = project?.id === projectId ? project : null;
  const titles = useMemo(() => {
    const map = new Map<string, string>();
    for (const p of projects) map.set(p.id, p.title);
    if (project) map.set(project.id, project.title);
    return map;
  }, [projects, project]);
  const workerState = health?.worker?.state;
  const status = !online
    ? { tone: "danger", text: "Workspace offline" }
    : !health
      ? { tone: "muted", text: "Local workspace" }
      : !health.worker?.alive
        ? { tone: "warn", text: "Worker unavailable" }
        : workerState === "busy"
          ? {
              tone: "accent",
              text:
                "Working" +
                (health.queue_length ? ` · ${health.queue_length} queued` : ""),
            }
          : workerState === "waiting_memory"
            ? { tone: "warn", text: "Waiting for 1.5 GiB free RAM" }
            : {
                tone: "success",
                text:
                  "Worker idle" +
                  (health.queue_length
                    ? ` · ${health.queue_length} queued`
                    : ""),
              };
  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Workspace">
        <a
          className="brand"
          href="/"
          onClick={(e) => {
            e.preventDefault();
            leaveProject();
          }}
          title="Back to the library"
        >
          <span className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          <span className="brand-text">
            vektri<span className="brand-sub">CLIP STUDIO</span>
          </span>
        </a>
        <div className="workspace-label">Workspace</div>
        <nav aria-label="Main navigation">
          {PAGES.map(([id, icon, label]) => (
            <button
              key={id}
              className={`nav-item ${page === id && !projectId ? "active" : ""}`}
              aria-current={page === id && !projectId ? "page" : undefined}
              onClick={() => navigate({ page: id })}
            >
              <Icon name={icon} />
              <span>{label}</span>
              {id === "jobs" && active.length > 0 && (
                <span className="count">{active.length}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="local-status" title={status.text}>
            <span className={`status-dot ${status.tone}`} />
            <div>
              <strong>Local workspace</strong>
              <small>{status.text}</small>
            </div>
          </div>
          <div className="theme-label" id={themeLabelId}>
            Appearance
          </div>
          <div
            className="theme-switch"
            role="radiogroup"
            aria-labelledby={themeLabelId}
          >
            {THEMES.map(([id, label, icon]) => (
              <button
                key={id}
                role="radio"
                aria-checked={theme === id}
                onClick={() => setTheme(id)}
              >
                <Icon name={icon} size={14} />
                {label}
              </button>
            ))}
          </div>
          <a href="/docs" target="_blank" rel="noreferrer" className="api-link">
            API reference <Icon name="external" size={14} />
          </a>
          <div className="operator">
            <span aria-hidden="true">VS</span>
            <div>
              <strong>Local studio</strong>
              <small>Local operator · loopback only</small>
            </div>
          </div>
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <div className="breadcrumb">
            <button className="crumb" onClick={leaveProject}>
              Workspace
            </button>
            <Icon name="chevron" size={13} />
            {current ? (
              <>
                <button
                  className="crumb"
                  onClick={() => navigate({ page: "library" })}
                >
                  Library
                </button>
                <Icon name="chevron" size={13} />
                <span>{current.title}</span>
              </>
            ) : (
              <span>
                {projectId
                  ? "Loading…"
                  : PAGES.find(([id]) => id === page)?.[2]}
              </span>
            )}
          </div>
          <div className="top-actions">
            <span className={`status-chip ${status.tone}`} role="status">
              <span className="status-dot" />
              {status.text}
            </span>
            <button
              className="icon-btn theme-cycle"
              aria-label={`Appearance: ${THEMES.find(([id]) => id === theme)?.[1]}. Switch to ${THEMES.find(([id]) => id === nextTheme(theme))?.[1].toLowerCase()}`}
              onClick={() => setTheme(nextTheme(theme))}
            >
              <Icon
                name={THEMES.find(([id]) => id === theme)?.[2] ?? "monitor"}
              />
            </button>
            <button
              className="button primary small"
              onClick={() => setImportOpen(true)}
            >
              <Icon name="plus" size={16} />
              Import media
            </button>
          </div>
        </header>
        {!online && (
          <div className="notice-wrap">
            <div className="offline-box" role="status">
              <Icon name="warning" size={16} />
              Cannot reach the local API at this address. Start it with{" "}
              <code>./scripts/start.sh</code> and this page will reconnect.
            </div>
          </div>
        )}
        {error && (
          <div className="notice-wrap">
            <ErrorBox error={error} onDismiss={() => setError("")} />
          </div>
        )}
        {notice && (
          <div className="toast" role="status">
            <Icon name="check" size={16} />
            {notice}
            <button aria-label="Dismiss notice" onClick={() => setNotice("")}>
              <Icon name="close" size={16} />
            </button>
          </div>
        )}
        {projectId ? (
          !current ? (
            <div className="loading">Loading project…</div>
          ) : (
            <>
              <div className="project-heading">
                <button
                  className="icon-btn"
                  aria-label="Back to library"
                  onClick={leaveProject}
                >
                  <Icon name="back" />
                </button>
                <div className="project-title">
                  <div className="eyebrow">Source workspace</div>
                  <h1>{current.title}</h1>
                </div>
                <Badge
                  tone={
                    current.status === "ready"
                      ? "success"
                      : current.status === "failed"
                        ? "danger"
                        : "accent"
                  }
                >
                  {current.status}
                </Badge>
                <div className="project-meta">
                  <span>{seconds(current.source?.duration)}</span>
                  <span className="dot">·</span>
                  <span>{GENRES[current.genre] ?? current.genre}</span>
                  {current.source?.width && current.source?.height ? (
                    <>
                      <span className="dot">·</span>
                      <span>
                        {current.source.width} × {current.source.height}
                      </span>
                    </>
                  ) : null}
                </div>
              </div>
              <div
                className="project-tabs"
                role="tablist"
                aria-label="Project workflow"
              >
                {TABS.map(([id, num, label]) => (
                  <button
                    role="tab"
                    aria-selected={tab === id}
                    key={id}
                    className={tab === id ? "active" : ""}
                    onClick={() => navigate({ projectId, tab: id }, true)}
                  >
                    <span>{num}</span>
                    {label}
                  </button>
                ))}
              </div>
              {tab === "source" && (
                <Source
                  project={current}
                  onChange={refreshProject}
                  onAnalyze={() =>
                    navigate({ projectId, tab: "analysis" }, true)
                  }
                  onEdit={() => navigate({ projectId, tab: "editor" }, true)}
                  onError={setError}
                  onNotice={setNotice}
                />
              )}
              {tab === "analysis" && (
                <Analysis
                  key={current.id}
                  project={current}
                  onChange={refreshProject}
                  onEdit={() => navigate({ projectId, tab: "editor" }, true)}
                  onError={setError}
                />
              )}
              {tab === "editor" &&
                (current.timeline ? (
                  <Editor
                    key={current.id}
                    project={current}
                    onChange={refreshProject}
                    onError={setError}
                    onRendered={() => {
                      setNotice("Render queued. You can keep editing.");
                      navigate({ projectId, tab: "export" }, true);
                    }}
                  />
                ) : (
                  <Empty title="Preparing your source…">
                    The editor will open here as soon as source metadata and
                    preview preparation finish. Check the source view for the
                    current job stage.
                  </Empty>
                ))}
              {tab === "export" && (
                <ExportPanel
                  project={current}
                  jobs={current.jobs ?? []}
                  onAction={action}
                />
              )}
            </>
          )
        ) : page === "library" ? (
          <Library
            projects={projects}
            loading={loading}
            search={search}
            onSearch={setSearch}
            selected={selected}
            onSelect={setSelected}
            onOpen={openProject}
            onImport={() => setImportOpen(true)}
            onBatch={() => {
              const items = projects
                .filter((p) => selected.has(p.id))
                .map((p) => ({ project_id: p.id, revision: p.revision }));
              void action(async () => {
                const result = await submitSelectedBatch(
                  items.map((item) => item.project_id),
                  () =>
                    post("/batch/render", {
                      items,
                      idempotency_key: crypto.randomUUID(),
                    }),
                  (queued) =>
                    setSelected((current) =>
                      retainUnqueuedSelection(current, queued),
                    ),
                );
                if (result.queuedIds.size)
                  setNotice(
                    `${result.queuedIds.size} of ${items.length} selected projects queued. Open Job queue for results.`,
                  );
                if (result.failedIds.length)
                  setError(
                    `${result.failedIds.length} selected project(s) were not confirmed queued and remain selected. ${result.errors.join("; ")}`.trim(),
                  );
              });
            }}
          />
        ) : page === "jobs" ? (
          <JobsPage jobs={jobs} titles={titles} onAction={action} />
        ) : page === "exports" ? (
          <GlobalExports
            exports={allExports}
            titles={titles}
            onAction={action}
            onOpen={openProject}
          />
        ) : page === "publishing" ? (
          <Publishing onError={setError} titles={titles} />
        ) : page === "storage" ? (
          <StoragePage onError={setError} />
        ) : page === "audit" ? (
          <AuditPage titles={titles} onOpen={openProject} onError={setError} />
        ) : (
          <Models onAction={action} jobs={jobs} />
        )}
      </main>
      {importOpen && (
        <ImportModal
          onClose={() => setImportOpen(false)}
          onImported={(id) => {
            setImportOpen(false);
            openProject(id);
            void refresh();
          }}
        />
      )}
    </div>
  );
}

function Library({
  projects,
  loading,
  search,
  onSearch,
  selected,
  onSelect,
  onOpen,
  onImport,
  onBatch,
}: {
  projects: Project[];
  loading: boolean;
  search: string;
  onSearch: (s: string) => void;
  selected: Set<string>;
  onSelect: (s: Set<string>) => void;
  onOpen: (id: string) => void;
  onImport: () => void;
  onBatch: () => void;
}) {
  const [genre, setGenre] = useState("all"),
    [sort, setSort] = useState("recent");
  const shown = useMemo(() => {
    const items = projects.filter((p) => genre === "all" || p.genre === genre);
    const stamp = (p: Project) => {
      const v = p.updated_at ?? p.created_at ?? 0;
      return typeof v === "number" ? v : Date.parse(String(v)) || 0;
    };
    if (sort === "title") items.sort((a, b) => a.title.localeCompare(b.title));
    else if (sort === "duration")
      items.sort(
        (a, b) =>
          Number(b.source?.duration ?? b.duration ?? 0) -
          Number(a.source?.duration ?? a.duration ?? 0),
      );
    else items.sort((a, b) => stamp(b) - stamp(a));
    return items;
  }, [projects, genre, sort]);
  const toggle = (id: string, on: boolean) => {
    const n = new Set(selected);
    on ? n.add(id) : n.delete(id);
    onSelect(n);
  };
  return (
    <div className="page-content">
      <div className="page-heading">
        <div>
          <div className="eyebrow">The edit starts here</div>
          <h1>
            Your library<span className="heading-dot">.</span>
          </h1>
          <p>Find the moment. Make it yours.</p>
        </div>
        <button className="button secondary" onClick={onImport}>
          <Icon name="upload" />
          Add a source
        </button>
      </div>
      <div className="library-toolbar">
        <label className="search">
          <Icon name="search" />
          <input
            aria-label="Search projects"
            value={search}
            onChange={(e) => onSearch(e.target.value)}
            placeholder="Search projects or tags…"
          />
          {search && (
            <button
              className="icon-btn"
              aria-label="Clear search"
              onClick={() => onSearch("")}
            >
              <Icon name="close" size={14} />
            </button>
          )}
        </label>
        <select
          aria-label="Filter library by type"
          value={genre}
          onChange={(e) => setGenre(e.target.value)}
        >
          <option value="all">All footage types</option>
          {Object.entries(GENRES).map(([id, label]) => (
            <option key={id} value={id}>
              {label}
            </option>
          ))}
        </select>
        <select
          aria-label="Sort library"
          value={sort}
          onChange={(e) => setSort(e.target.value)}
        >
          <option value="recent">Recently updated</option>
          <option value="title">Title A–Z</option>
          <option value="duration">Longest first</option>
        </select>
        <span className="subtle count-label">
          {shown.length} {shown.length === 1 ? "source" : "sources"}
        </span>
        {selected.size > 0 && (
          <div className="row selection-actions">
            <button className="button primary small" onClick={onBatch}>
              <Icon name="download" size={15} />
              Render {selected.size} selected
            </button>
            <button className="text-button" onClick={() => onSelect(new Set())}>
              Clear
            </button>
          </div>
        )}
      </div>
      {loading ? (
        <div className="loading">Opening your workspace…</div>
      ) : shown.length === 0 ? (
        <div className="library-empty">
          <div className="empty-timeline" aria-hidden="true">
            <span />
            <span />
            <span />
            <i />
          </div>
          <Empty
            title={
              search || genre !== "all"
                ? "No matching projects"
                : "Every great clip starts with a source."
            }
            icon="film"
            action={
              <button className="button primary" onClick={onImport}>
                <Icon name="plus" />
                Import your first video
              </button>
            }
          >
            {search || genre !== "all"
              ? "Try a different title, tag or footage type."
              : "Drop in a video or audio file, import a public media URL, or explore the studio with a clearly labeled synthetic sample."}
          </Empty>
          <div className="empty-features">
            <span>
              <Icon name="file" />
              Originals stay untouched
            </span>
            <span>
              <Icon name="spark" />
              Local AI, on your terms
            </span>
            <span>
              <Icon name="download" />
              Real files. Ready to share.
            </span>
          </div>
        </div>
      ) : (
        <div className="project-grid">
          {shown.map((p) => {
            const ready = p.status === "ready";
            return (
              <article
                className={`project-card ${selected.has(p.id) ? "selected" : ""}`}
                key={p.id}
              >
                <button
                  className="thumbnail-button"
                  onClick={() => onOpen(p.id)}
                >
                  <span className="sr-only">Open {p.title}</span>
                  {thumb(p) && (
                    <img
                      src={thumb(p)}
                      alt=""
                      loading="lazy"
                      onError={(e) => {
                        e.currentTarget.style.visibility = "hidden";
                      }}
                    />
                  )}
                  <span className="thumb-placeholder">
                    <Icon name="film" size={30} />
                  </span>
                  <span className="duration">
                    {seconds(p.source?.duration ?? p.duration)}
                  </span>
                  <span className="thumb-play">
                    <Icon name="play" />
                  </span>
                </button>
                <div className="card-body">
                  <div className="card-topline">
                    <span className="eyebrow">
                      {GENRES[p.genre] ?? p.genre ?? "Media"}
                    </span>
                    <input
                      type="checkbox"
                      aria-label={"Select " + p.title + " for batch render"}
                      checked={selected.has(p.id)}
                      disabled={!ready}
                      onChange={(e) => toggle(p.id, e.target.checked)}
                    />
                  </div>
                  <button className="title-button" onClick={() => onOpen(p.id)}>
                    {p.title}
                  </button>
                  <div className="card-bottom">
                    <span className={ready ? "" : "working-text"}>
                      {ready
                        ? "Ready to edit"
                        : p.status === "failed"
                          ? "Import failed"
                          : "Preparing…"}
                    </span>
                    <span>
                      {p.exports?.length ?? 0}{" "}
                      {p.exports?.length === 1 ? "export" : "exports"}
                    </span>
                  </div>
                  {p.tags?.length > 0 && (
                    <div className="tags">
                      {p.tags.map((tag) => (
                        <span key={tag}>#{tag}</span>
                      ))}
                    </div>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      )}
      <div className="library-footnote">
        <Icon name="eye" size={15} />
        Private editing and exporting are separate from publishing. Source
        provenance stays with each project.
      </div>
    </div>
  );
}

function ImportModal({
  onClose,
  onImported,
}: {
  onClose: () => void;
  onImported: (id: string) => void;
}) {
  const [mode, setMode] = useState("file"),
    [file, setFile] = useState<File | null>(null),
    [url, setUrl] = useState(""),
    [title, setTitle] = useState(""),
    [genre, setGenre] = useState("mixed"),
    [attribution, setAttribution] = useState(""),
    [permission, setPermission] = useState("unknown"),
    [fixtures, setFixtures] = useState<Fixture[]>([]),
    [busy, setBusy] = useState(false),
    [dragging, setDragging] = useState(false),
    [error, setError] = useState("");
  const controller = useRef<AbortController | null>(null);
  const [uploadProgress, setUploadProgress] = useState<{
    sent: number;
    total: number;
  } | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    api<unknown>("/fixtures")
      .then((r) => setFixtures(list<Fixture>(r, "fixtures")))
      .catch((e) => setError(message(e)));
  }, []);
  const run = async (fn: () => Promise<{ project_id: string }>) => {
    setBusy(true);
    setError("");
    try {
      const r = await fn();
      onImported(r.project_id);
    } catch (e) {
      setError(
        e instanceof Error && e.name === "AbortError"
          ? "Upload paused. Select this same file with the same project details to resume from the stored offset."
          : message(e),
      );
      setBusy(false);
    }
  };
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (mode === "url")
      void run(() =>
        post("/imports/url", {
          url,
          title,
          genre,
          attribution,
          permission_status: permission,
        }),
      );
    else if (file) {
      controller.current = new AbortController();
      void run(() =>
        resumableUpload(
          file,
          {
            title: title || file.name.replace(/\.[^.]+$/, ""),
            genre,
            attribution,
            permission_status: permission,
          },
          controller.current!.signal,
          (sent, total) => setUploadProgress({ sent, total }),
        ),
      );
    }
  };
  return (
    <Modal
      title="Bring a source into your studio"
      onClose={() => {
        controller.current?.abort();
        onClose();
      }}
    >
      <div className="modal-body">
        <p className="subtle lead">
          Keep the original. Build the edit. Decide where it goes later.
        </p>
        <div className="segmented">
          {[
            ["file", "Local file"],
            ["url", "Media URL"],
            ["sample", "Synthetic samples"],
          ].map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setMode(id)}
              className={mode === id ? "active" : ""}
            >
              {label}
            </button>
          ))}
        </div>
        {error && <ErrorBox error={error} />}
        {uploadProgress && (
          <div className="upload-progress">
            <progress
              value={uploadProgress.sent}
              max={uploadProgress.total}
              aria-label="Bytes uploaded"
            />
            <div className="row between">
              <span className="tiny subtle">
                {bytes(uploadProgress.sent)} of {bytes(uploadProgress.total)}{" "}
                transferred
              </span>
              {busy && (
                <button
                  className="text-button"
                  onClick={() => controller.current?.abort()}
                >
                  Pause upload
                </button>
              )}
            </div>
          </div>
        )}
        {mode === "sample" ? (
          <>
            <p className="info-note">
              Original procedural fixtures with known timings. Synthetic motion
              checks mechanics; it does not validate understanding of real
              sports or gameplay.
            </p>
            <div className="fixture-list">
              {fixtures.map((f) => (
                <button
                  disabled={busy}
                  key={String(f.id ?? f.name)}
                  onClick={() =>
                    void run(() =>
                      post(
                        "/fixtures/" +
                          encodeURIComponent(String(f.id ?? f.name)),
                      ),
                    )
                  }
                >
                  <span className="fixture-symbol">
                    <Icon name="film" />
                  </span>
                  <span>
                    <strong>
                      {String(
                        f.title ?? f.name ?? f.id ?? f.filename,
                      ).replaceAll("_", " ")}
                    </strong>
                    <small>
                      {String(f.genre ?? "Synthetic fixture")}
                      {f.duration ? " · " + seconds(f.duration) : ""}
                      {typeof f.size === "number" ? " · " + bytes(f.size) : ""}
                    </small>
                  </span>
                  <Icon name="plus" />
                </button>
              ))}
            </div>
          </>
        ) : (
          <form onSubmit={submit}>
            {mode === "file" ? (
              <label
                className={`drop-zone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragging(false);
                  setFile(e.dataTransfer.files[0] ?? null);
                }}
              >
                <input
                  type="file"
                  accept="video/*,audio/*,.mkv,.mov,.webm,.m4a,.flac,.wav"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                  required={!file}
                />
                <Icon name={file ? "film" : "upload"} size={26} />
                <strong>
                  {file ? file.name : "Drop a video or audio file here"}
                </strong>
                <span>
                  {file
                    ? bytes(file.size) + " · choose another file to replace"
                    : "or choose a file from your device"}
                </span>
              </label>
            ) : (
              <Field
                label="Direct public media URL"
                hint="Direct downloadable media only. Unsupported platforms can be imported as a local file."
              >
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  required
                  placeholder="https://example.com/video.mp4"
                />
              </Field>
            )}
            <div className="form-grid">
              <Field label="Project title">
                <input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="A name for this source"
                />
              </Field>
              <Field label="Footage type">
                <select
                  value={genre}
                  onChange={(e) => setGenre(e.target.value)}
                >
                  <option value="mixed">Mixed / not sure</option>
                  <option value="speech">Podcast or interview</option>
                  <option value="tutorial">Tutorial or screen</option>
                  <option value="action">Action / nonverbal</option>
                </select>
              </Field>
            </div>
            <Field label="Source attribution">
              <input
                value={attribution}
                onChange={(e) => setAttribution(e.target.value)}
                placeholder="Creator or source, if known"
              />
            </Field>
            <Field
              label="Recorded permission status"
              hint="This is provenance, not proof of legal rights. Importing never authorizes publishing."
            >
              <select
                value={permission}
                onChange={(e) => setPermission(e.target.value)}
              >
                <option value="unknown">Unknown / not recorded</option>
                <option value="owned">I created this material</option>
                <option value="licensed">Licensed material</option>
                <option value="permission-recorded">Permission recorded</option>
              </select>
            </Field>
            <button
              className="button primary full"
              type="submit"
              disabled={busy}
            >
              {busy ? "Importing source…" : "Import into workspace"}
              <Icon name="arrow" />
            </button>
          </form>
        )}
        {busy && (
          <p role="status" className="subtle">
            Import in progress. Preparation continues as a background job.
          </p>
        )}
      </div>
    </Modal>
  );
}

function Source({
  project: p,
  onChange,
  onAnalyze,
  onEdit,
  onError,
  onNotice,
}: {
  project: Project;
  onChange: () => Promise<void>;
  onAnalyze: () => void;
  onEdit: () => void;
  onError: (e: string) => void;
  onNotice: (s: string) => void;
}) {
  const s = {
      ...(p.source ?? {}),
      ...(p.source?.metadata ?? {}),
    } as NonNullable<Project["source"]>,
    metadata = p.source?.metadata ?? {};
  const permission = String(
    s.provenance?.permission_status ?? p.permission_status ?? "unknown",
  );
  const attribution = String(s.provenance?.attribution ?? p.attribution ?? "");
  const [editing, setEditing] = useState(false),
    [title, setTitle] = useState(p.title),
    [tags, setTags] = useState(p.tags?.join(", ") ?? ""),
    [genre, setGenre] = useState(p.genre),
    [attributionDraft, setAttributionDraft] = useState(attribution),
    [permissionDraft, setPermissionDraft] = useState(permission),
    [saving, setSaving] = useState(false);
  const ready = p.status === "ready";
  const startEditing = () => {
    setTitle(p.title);
    setTags(p.tags?.join(", ") ?? "");
    setGenre(p.genre);
    setAttributionDraft(attribution);
    setPermissionDraft(permission);
    setEditing(true);
  };
  const save = async () => {
    setSaving(true);
    try {
      await api("/projects/" + p.id, {
        method: "PATCH",
        body: JSON.stringify({
          title: title.trim() || p.title,
          genre,
          tags: tags
            .split(",")
            .map((t) => t.trim())
            .filter(Boolean),
          attribution: attributionDraft,
          permission_status: permissionDraft,
        }),
      });
      setEditing(false);
      await onChange();
      onNotice("Source details saved. Any manual-handoff approval was reset.");
    } catch (e) {
      onError(message(e));
    } finally {
      setSaving(false);
    }
  };
  const importJob = p.jobs?.find((j) => j.kind.startsWith("import"));
  return (
    <div className="source-layout">
      <section className="source-main">
        <div className="section-kicker">
          <span>Source preview</span>
          <Badge>Original preserved</Badge>
        </div>
        <div className="source-player">
          {video(p) ? (
            <video
              src={video(p)}
              poster={thumb(p)}
              controls
              preload="metadata"
              aria-label="Source video preview"
            />
          ) : (
            <div className="player-placeholder">
              <Icon name="film" size={34} />
              <span>
                {p.status === "failed"
                  ? "Import failed. Inspect the job below."
                  : "Preparing preview…"}
              </span>
            </div>
          )}
        </div>
        <div className="source-actions">
          <div>
            <h2>A good edit starts with context.</h2>
            <p>
              Review your source, discover candidates, or jump straight into a
              manual edit.
            </p>
          </div>
          <button
            className="button secondary"
            onClick={onEdit}
            disabled={!ready}
          >
            <Icon name="cut" />
            Manual edit
          </button>
          <button
            className="button primary"
            onClick={onAnalyze}
            disabled={!ready}
          >
            <Icon name="spark" />
            Find moments
          </button>
        </div>
        {!ready && importJob && (
          <p className="info-note">
            <Icon name="info" size={16} />
            <span>
              Source preparation runs as a background job. Current stage:{" "}
              <strong>{importJob.stage || importJob.status}</strong>.
            </span>
          </p>
        )}
        <Jobs
          jobs={
            p.jobs?.filter((j) =>
              ["queued", "running", "failed"].includes(j.status),
            ) ?? []
          }
          compact
          onAction={async (f) => {
            try {
              await f();
              await onChange();
            } catch (e) {
              onError(message(e));
            }
          }}
        />
      </section>
      <aside className="source-details" aria-label="Source details">
        <div className="panel-heading">
          <h2>Source details</h2>
          {!editing && (
            <button className="text-button" onClick={startEditing}>
              Edit
            </button>
          )}
        </div>
        {editing ? (
          <div className="stack">
            <Field label="Title">
              <input value={title} onChange={(e) => setTitle(e.target.value)} />
            </Field>
            <Field label="Tags, separated by commas">
              <input value={tags} onChange={(e) => setTags(e.target.value)} />
            </Field>
            <Field label="Footage type">
              <select value={genre} onChange={(e) => setGenre(e.target.value)}>
                {Object.entries(GENRES).map(([id, label]) => (
                  <option key={id} value={id}>
                    {label}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Source attribution">
              <input
                value={attributionDraft}
                onChange={(e) => setAttributionDraft(e.target.value)}
                placeholder="Creator or source, if known"
              />
            </Field>
            <Field
              label="Recorded permission status"
              hint="Provenance only; it does not establish legal rights."
            >
              <select
                value={permissionDraft}
                onChange={(e) => setPermissionDraft(e.target.value)}
              >
                <option value="unknown">Unknown / not recorded</option>
                <option value="owned">I created this material</option>
                <option value="licensed">Licensed material</option>
                <option value="permission-recorded">Permission recorded</option>
                {permission === "synthetic" && (
                  <option value="synthetic">Synthetic fixture</option>
                )}
              </select>
            </Field>
            <div className="row">
              <button
                className="button primary small"
                disabled={saving}
                onClick={() => void save()}
              >
                {saving ? "Saving…" : "Save details"}
              </button>
              <button className="text-button" onClick={() => setEditing(false)}>
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <>
            <h3>{p.title}</h3>
            {p.tags?.length ? (
              <div className="tags">
                {p.tags.map((t) => (
                  <span key={t}>#{t}</span>
                ))}
              </div>
            ) : (
              <p className="tiny subtle">No tags yet.</p>
            )}
          </>
        )}
        <dl className="metadata">
          <div>
            <dt>Filename</dt>
            <dd>{s.original_name ?? "Preparing…"}</dd>
          </div>
          <div>
            <dt>Footage type</dt>
            <dd>{GENRES[p.genre] ?? p.genre}</dd>
          </div>
          <div>
            <dt>Duration</dt>
            <dd>{seconds(s.duration)}</dd>
          </div>
          <div>
            <dt>Dimensions</dt>
            <dd>{s.width && s.height ? `${s.width} × ${s.height}` : "—"}</dd>
          </div>
          <div>
            <dt>Frame rate</dt>
            <dd>{s.fps ? `${Number(s.fps).toFixed(3)} fps` : "—"}</dd>
          </div>
          <div>
            <dt>Rotation</dt>
            <dd>{s.rotation ?? 0}°</dd>
          </div>
          <div>
            <dt>File size</dt>
            <dd>{bytes(s.size as number | undefined)}</dd>
          </div>
          <div>
            <dt>Audio</dt>
            <dd>
              {s.audio
                ? Number(s.audio_stream_count ?? 1) > 1
                  ? `${s.audio_stream_count} tracks`
                  : "Available"
                : "No audio detected"}
            </dd>
          </div>
          <div>
            <dt>Permission</dt>
            <dd>{permission}</dd>
          </div>
          <div>
            <dt>Attribution</dt>
            <dd>{attribution || "Not recorded"}</dd>
          </div>
          <div>
            <dt>Imported</dt>
            <dd>
              {when(s.provenance?.imported_at as number | undefined) || "—"}
            </dd>
          </div>
        </dl>
        {s.sha256 && (
          <details>
            <summary>Source integrity & provenance</summary>
            <p className="tiny mono wrap">
              SHA-256
              <br />
              {s.sha256}
            </p>
            {s.original_url ? (
              <p>
                <a className="text-link" href={String(s.original_url)} download>
                  <Icon name="download" size={14} /> Download the untouched
                  original
                </a>
              </p>
            ) : null}
            <pre>
              {JSON.stringify(
                {
                  ...s.provenance,
                  time_base: metadata.time_base,
                  streams: metadata.streams,
                },
                null,
                2,
              )}
            </pre>
          </details>
        )}
        <div className="info-note">
          <Icon name="file" />
          <p>
            Your source is immutable. Timeline edits create new revisions.
            Exporting and publishing are separate decisions.
          </p>
        </div>
      </aside>
    </div>
  );
}

export function Jobs({
  jobs,
  onAction,
  compact = false,
  titles,
  limit,
}: {
  jobs: Job[];
  onAction: (fn: () => Promise<unknown>, success?: string) => Promise<void>;
  compact?: boolean;
  titles?: Map<string, string>;
  limit?: number;
}) {
  if (!jobs.length)
    return compact ? null : (
      <Empty icon="queue" title="Nothing in the queue.">
        Imports, model downloads, analysis, and renders appear here as real
        background jobs.
      </Empty>
    );
  return (
    <div className="job-list">
      {jobs.slice(0, limit ?? (compact ? 5 : 100)).map((j) => {
        const title = titles?.get(j.project_id ?? "");
        const modelName =
          j.kind === "model-download" ? String(j.params?.name ?? "") : "";
        return (
          <article className="job-row" key={j.id} data-status={j.status}>
            <div
              className={`job-symbol ${j.status === "running" ? "working" : ""}`}
            >
              <Icon name={jobIcon(j.kind)} />
            </div>
            <div className="job-info">
              <strong>
                {jobLabel(j.kind)}
                {modelName ? ` · ${modelName}` : ""}
                {j.kind === "render" && j.params?.revision
                  ? ` · revision ${j.params.revision}`
                  : ""}
              </strong>
              <span>
                {!compact && (title || j.project_id) ? (
                  <>
                    <em>{title ?? "Project " + j.project_id?.slice(0, 8)}</em>
                    {" · "}
                  </>
                ) : null}
                {j.stage && j.stage !== j.status
                  ? j.stage
                  : j.status === "running"
                    ? "running"
                    : j.status}
                {j.cancel_requested ? " · cancellation requested" : ""}
              </span>
              {Boolean(j.error) && (
                <p className="job-error">
                  {typeof j.error === "string"
                    ? j.error
                    : JSON.stringify(j.error)}
                </p>
              )}
            </div>
            <div className="job-time">
              {typeof j.elapsed_seconds === "number" && j.started_at
                ? `${Math.round(j.elapsed_seconds)}s`
                : `Attempt ${j.attempts ?? 0}/${j.max_attempts ?? 3}`}
              {j.created_at ? <small>{when(j.created_at)}</small> : null}
            </div>
            <Badge tone={statusTone(j.status)}>{j.status}</Badge>
            {ACTIVE.includes(j.status) ? (
              <button
                className="text-button"
                disabled={j.cancel_requested}
                onClick={() =>
                  void onAction(() => post("/jobs/" + j.id + "/cancel"))
                }
              >
                Cancel
              </button>
            ) : ["failed", "cancelled"].includes(j.status) ? (
              <button
                className="text-button"
                onClick={() =>
                  void onAction(() => post("/jobs/" + j.id + "/retry"))
                }
              >
                Retry
              </button>
            ) : (
              <span className="job-spacer" />
            )}
          </article>
        );
      })}
    </div>
  );
}

function JobsPage({
  jobs,
  titles,
  onAction,
}: {
  jobs: Job[];
  titles: Map<string, string>;
  onAction: (fn: () => Promise<unknown>, success?: string) => Promise<void>;
}) {
  const [filter, setFilter] = useState("all"),
    [shown, setShown] = useState(40);
  const activeJobs = jobs.filter((j) => ACTIVE.includes(j.status));
  const history = jobs.filter(
    (j) =>
      !ACTIVE.includes(j.status) &&
      (filter === "all" ||
        (filter === "failed"
          ? ["failed", "cancelled"].includes(j.status)
          : j.kind === filter)),
  );
  const kinds = [...new Set(jobs.map((j) => j.kind))];
  return (
    <div className="page-content">
      <div className="page-heading">
        <div>
          <div className="eyebrow">Work in progress</div>
          <h1>
            Job queue<span className="heading-dot">.</span>
          </h1>
          <p>Real stages, recoverable work, and room to change your mind.</p>
        </div>
        <Badge tone={activeJobs.length ? "accent" : "muted"}>
          {activeJobs.length} active
        </Badge>
      </div>
      <section className="jobs-section">
        <div className="section-kicker">
          <span>Active</span>
          <span>{activeJobs.length ? "" : "The worker is idle"}</span>
        </div>
        {activeJobs.length ? (
          <Jobs jobs={activeJobs} titles={titles} onAction={onAction} />
        ) : (
          <p className="subtle quiet">
            Nothing is running or queued right now.
          </p>
        )}
      </section>
      <section className="jobs-section">
        <div className="section-kicker">
          <span>History</span>
          <select
            aria-label="Filter job history"
            value={filter}
            onChange={(e) => {
              setFilter(e.target.value);
              setShown(40);
            }}
          >
            <option value="all">All jobs</option>
            <option value="failed">Failed or cancelled</option>
            {kinds.map((k) => (
              <option key={k} value={k}>
                {jobLabel(k)}
              </option>
            ))}
          </select>
        </div>
        {history.length ? (
          <>
            <Jobs
              jobs={history}
              titles={titles}
              onAction={onAction}
              limit={shown}
            />
            {history.length > shown && (
              <button
                className="button secondary small"
                onClick={() => setShown((n) => n + 40)}
              >
                Show more ({history.length - shown} older)
              </button>
            )}
          </>
        ) : (
          <Empty icon="queue" title="Nothing in the queue.">
            Imports, model downloads, analysis, and renders appear here as real
            background jobs.
          </Empty>
        )}
      </section>
    </div>
  );
}

function ExportPanel({
  project,
  jobs,
  onAction,
}: {
  project: Project;
  jobs: Job[];
  onAction: (fn: () => Promise<unknown>, success?: string) => Promise<void>;
}) {
  const renders = jobs.filter((j) => j.kind === "render");
  const latestRendered = project.exports?.some(
    (e) => e.revision === project.revision,
  );
  return (
    <div className="page-content">
      <div className="page-heading smaller">
        <div>
          <h2>Make it a real file.</h2>
          <p>
            Export revision {project.revision}.{" "}
            {latestRendered
              ? "This revision already has a verified export below."
              : "Your edit stays available while the worker renders."}
          </p>
        </div>
        <button
          className="button primary"
          onClick={() =>
            void onAction(
              () =>
                post(
                  "/projects/" + project.id + "/render",
                  { revision: project.revision },
                  true,
                ),
              "Render queued.",
            )
          }
        >
          <Icon name="download" />
          Render current edit
        </button>
      </div>
      <Jobs jobs={renders} compact onAction={onAction} />
      <ExportCards project={project} onAction={onAction} />
    </div>
  );
}

type ExportContext = Pick<Project, "id" | "title" | "source" | "exports">;
function ExportCards({
  project: p,
  onAction,
}: {
  project: ExportContext;
  onAction: (fn: () => Promise<unknown>, success?: string) => Promise<void>;
}) {
  const [draft, setDraft] = useState<Export | null>(null);
  return (
    <>
      {!p.exports?.length ? (
        <Empty icon="download" title="No finished exports yet.">
          Queue a render to create playable video, caption sidecars, a
          thumbnail, and a manifest. A queued job is not a finished render.
        </Empty>
      ) : (
        <div className="export-grid">
          {p.exports.map((e) => {
            const m = (e.manifest ?? {}) as Record<string, unknown>;
            const files = (m.files ?? {}) as Record<string, { bytes?: number }>;
            const packageUrl =
              String(e.package_url ?? e.download_url ?? "") ||
              "/api/v1/exports/" + e.id + "/download";
            return (
              <article className="export-card" key={e.id}>
                <div className="export-preview">
                  {e.video_url ? (
                    <video
                      src={e.video_url}
                      controls
                      preload="metadata"
                      poster={e.thumbnail_url}
                      aria-label={"Export revision " + e.revision}
                    />
                  ) : (
                    <Icon name="film" size={38} />
                  )}
                </div>
                <div className="export-card-body">
                  <div className="row between">
                    <h3>Revision {e.revision}</h3>
                    <Badge tone="success">Verified export</Badge>
                  </div>
                  <p className="tiny subtle export-meta">
                    {when(e.created_at) || e.id}
                    {m.width && m.height ? ` · ${m.width} × ${m.height}` : ""}
                    {typeof m.fps === "number" ? ` · ${m.fps} fps` : ""}
                    {typeof m.duration === "number"
                      ? ` · ${seconds(m.duration)}`
                      : ""}
                    {files["video.mp4"]?.bytes
                      ? ` · ${bytes(files["video.mp4"].bytes)}`
                      : ""}
                  </p>
                  <div className="row wrap">
                    <a
                      className="button primary small"
                      href={packageUrl}
                      download
                    >
                      <Icon name="download" size={15} />
                      Download package
                    </a>
                    <button
                      className="button secondary small"
                      onClick={() => setDraft(e)}
                    >
                      Prepare handoff
                    </button>
                  </div>
                  <div className="sidecars">
                    {e.video_url && (
                      <a href={e.video_url} download>
                        <Icon name="film" size={13} /> MP4
                      </a>
                    )}
                    {e.srt_url && (
                      <a href={e.srt_url} download>
                        <Icon name="text" size={13} /> SRT
                      </a>
                    )}
                    {e.vtt_url && (
                      <a href={e.vtt_url} download>
                        <Icon name="text" size={13} /> VTT
                      </a>
                    )}
                    {e.thumbnail_url && (
                      <a href={e.thumbnail_url} download>
                        <Icon name="image" size={13} /> Thumbnail
                      </a>
                    )}
                  </div>
                  <details>
                    <summary>Render manifest</summary>
                    <pre>{JSON.stringify(e.manifest ?? e, null, 2)}</pre>
                  </details>
                </div>
              </article>
            );
          })}
        </div>
      )}
      {draft && (
        <PublicationModal
          project={p}
          exported={draft}
          onClose={() => setDraft(null)}
          onSave={async (body) => {
            await post("/publications", body);
            await onAction(
              async () => {},
              "Manual handoff draft saved. Nothing was published.",
            );
            setDraft(null);
          }}
        />
      )}
    </>
  );
}

function PublicationModal({
  project,
  exported,
  onClose,
  onSave,
}: {
  project: ExportContext;
  exported: Export;
  onClose: () => void;
  onSave: (body: unknown) => Promise<void>;
}) {
  const [draftError, setDraftError] = useState("");
  const [destination, setDestination] = useState("manual"),
    [account, setAccount] = useState(""),
    [caption, setCaption] = useState(""),
    [visibility, setVisibility] = useState("private"),
    [schedule, setSchedule] = useState(""),
    [timezone, setTimezone] = useState(
      Intl.DateTimeFormat().resolvedOptions().timeZone,
    );
  return (
    <Modal title="Prepare a manual publishing handoff" onClose={onClose}>
      <form
        className="modal-body stack"
        onSubmit={(e) => {
          e.preventDefault();
          void onSave({
            project_id: project.id,
            export_id: exported.id,
            destination,
            account,
            caption,
            visibility,
            schedule: schedule || null,
            timezone,
            permission_status: String(
              project.source?.provenance?.permission_status ?? "unknown",
            ),
          }).catch((e) => setDraftError(message(e)));
        }}
      >
        {draftError && <ErrorBox error={draftError} />}
        <div className="info-note">
          <Icon name="info" size={16} />
          <p>
            No platform account is connected. This saves a draft for review and
            manual upload; it cannot publish.
          </p>
        </div>
        <Field label="Destination">
          <select
            value={destination}
            onChange={(e) => setDestination(e.target.value)}
          >
            <option value="manual">Manual upload</option>
            <option value="youtube">YouTube</option>
            <option value="instagram">Instagram</option>
            <option value="tiktok">TikTok</option>
          </select>
        </Field>
        <Field label="Intended account">
          <input
            value={account}
            onChange={(e) => setAccount(e.target.value)}
            placeholder="Record the exact destination account"
          />
        </Field>
        <Field label="Caption">
          <textarea
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            rows={4}
          />
        </Field>
        <div className="form-grid">
          <Field label="Visibility">
            <select
              value={visibility}
              onChange={(e) => setVisibility(e.target.value)}
            >
              <option>private</option>
              <option>unlisted</option>
              <option>public</option>
            </select>
          </Field>
          <Field label="Planned schedule" hint="Metadata only; nothing posts.">
            <input
              type="datetime-local"
              value={schedule}
              onChange={(e) => setSchedule(e.target.value)}
            />
          </Field>
        </div>
        <Field label="Time zone" hint="IANA name, e.g. Australia/Perth">
          <input
            value={timezone}
            onChange={(e) => setTimezone(e.target.value)}
          />
        </Field>
        <p className="tiny subtle">
          Asset: {project.title} · revision {exported.revision}. Rights remain
          user recorded. Editing the asset requires a fresh review.
        </p>
        <button className="button primary" type="submit">
          Save handoff draft
        </button>
      </form>
    </Modal>
  );
}

const PUBLICATION_STATUS: Record<string, [string, string]> = {
  draft: ["Draft", "muted"],
  "ready-for-manual-upload": ["Approved for manual upload", "success"],
  "approval-invalidated": ["Approval invalidated", "warn"],
};
const humanize = (s: string) => s.replace(/[_-]/g, " ");

function Publishing({
  onError,
  titles,
}: {
  onError: (s: string) => void;
  titles: Map<string, string>;
}) {
  const [items, setItems] = useState<Publication[]>([]),
    [review, setReview] = useState<Publication | null>(null),
    [ack, setAck] = useState(""),
    [confirmed, setConfirmed] = useState(false),
    [busy, setBusy] = useState(false),
    [reviewError, setReviewError] = useState(""),
    [checks, setChecks] = useState<Record<string, PublicationPreflight>>({}),
    [checking, setChecking] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    const result = await api<unknown>("/publications");
    setItems(
      list<Publication>(result, "publications").map((p) => ({
        ...p,
        ...((p.draft as Record<string, unknown>) ?? {}),
      })),
    );
  }, []);
  useEffect(() => {
    void refresh().catch((e) => onError(message(e)));
  }, [refresh, onError]);
  const preflight = async (id: string) => {
    setChecking(id);
    try {
      const result = await api<PublicationPreflight>(
        "/publications/" + id + "/preflight",
      );
      setChecks((c) => ({ ...c, [id]: result }));
    } catch (e) {
      onError(message(e));
    } finally {
      setChecking(null);
    }
  };
  return (
    <div className="page-content">
      <div className="page-heading">
        <div>
          <div className="eyebrow">Review before release</div>
          <h1>
            Publishing queue<span className="heading-dot">.</span>
          </h1>
          <p>A deliberate handoff for every finished clip.</p>
        </div>
        <Badge>Manual upload</Badge>
      </div>
      <div className="info-note">
        <Icon name="info" size={16} />
        <p>
          Publishing adapters are not connected. Account authorization,
          destination review, and supported platform access are required before
          an automated post. Approval here records a manual handoff only.
        </p>
      </div>
      {!items.length ? (
        <Empty icon="upload" title="Make the final call here.">
          Render a clip, then choose “Prepare handoff” to record its
          destination, account, caption, visibility, and schedule.
        </Empty>
      ) : (
        <div className="publication-list">
          {items.map((p) => {
            const [label, tone] = PUBLICATION_STATUS[p.status ?? "draft"] ?? [
              p.status ?? "draft",
              "muted",
            ];
            const check = checks[p.id];
            return (
              <article className="publication-card" key={p.id}>
                <div className="row between wrap">
                  <h2>
                    {p.destination}{" "}
                    <span className="subtle">
                      / {p.account || "Account not recorded"}
                    </span>
                  </h2>
                  <Badge tone={tone}>{label}</Badge>
                </div>
                <p className="caption-text">
                  {p.caption || "No caption recorded."}
                </p>
                <div className="tiny subtle publication-meta">
                  <span>
                    {titles.get(p.project_id) ??
                      "Project " + p.project_id.slice(0, 8)}
                  </span>
                  <span>Revision {String(p.revision ?? "—")}</span>
                  <span>{p.visibility}</span>
                  <span>
                    {p.schedule
                      ? `${p.schedule} ${p.timezone}`
                      : `Unscheduled · ${p.timezone}`}
                  </span>
                  <span className="mono">Asset {p.export_id}</span>
                  <span className="mono">Status: {p.status ?? "draft"}</span>
                </div>
                <div className="row wrap publication-actions">
                  <button
                    className="button secondary small"
                    onClick={() => {
                      setReview(p);
                      setAck("");
                      setConfirmed(false);
                      setReviewError("");
                    }}
                  >
                    Review exact handoff
                  </button>
                  <button
                    className="button secondary small"
                    disabled={checking === p.id}
                    onClick={() => void preflight(p.id)}
                  >
                    <Icon name="refresh" size={14} />
                    {checking === p.id
                      ? "Checking…"
                      : check
                        ? "Recheck readiness"
                        : "Check readiness"}
                  </button>
                  <span className="tiny subtle">
                    {String(p.prerequisite ?? "Manual upload only.")}
                  </span>
                </div>
                {check && (
                  <div
                    className={`preflight ${check.manual_handoff_ready ? "ready" : ""}`}
                  >
                    <div className="row between wrap">
                      <strong>
                        {check.manual_handoff_ready
                          ? "Ready for manual handoff"
                          : "Not ready for manual handoff"}
                      </strong>
                      <span className="tiny subtle">
                        Checked {when(check.evaluated_at_unix)} · posting{" "}
                        {check.can_publish ? "available" : "unavailable"}
                      </span>
                    </div>
                    {check.manual_blockers?.length ? (
                      <div className="chips">
                        {check.manual_blockers.map((b) => (
                          <span className="chip danger" key={b}>
                            {humanize(b)}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <p className="tiny">
                        Current export bytes, approval and revision all match.
                      </p>
                    )}
                    {check.warnings?.map((w) => (
                      <p className="tiny warn-text" key={w}>
                        {w}
                      </p>
                    ))}
                    <p className="tiny subtle">
                      Remote posting blockers:{" "}
                      {(check.remote_blockers ?? []).map(humanize).join(", ") ||
                        "none reported"}
                      .
                    </p>
                  </div>
                )}
              </article>
            );
          })}
        </div>
      )}
      {review && (
        <Modal title="Review manual handoff" onClose={() => setReview(null)}>
          <div className="modal-body stack">
            <div className="info-note">
              <Icon name="info" size={16} />
              <p>
                This does not upload or publish. Confirm the exact version and
                destination before you download and upload it yourself. Later
                edits invalidate this approval.
              </p>
            </div>
            <dl className="metadata-list">
              <dt>Destination / account</dt>
              <dd>
                {review.destination} / {review.account || "Not recorded"}
              </dd>
              <dt>Asset / revision</dt>
              <dd className="mono wrap-anywhere">
                {review.export_id} / {String(review.revision)}
              </dd>
              <dt>Visibility</dt>
              <dd>{review.visibility}</dd>
              <dt>Schedule / time zone</dt>
              <dd>
                {review.schedule || "Unscheduled"} / {review.timezone}
              </dd>
              <dt>Caption</dt>
              <dd>{review.caption || "No caption"}</dd>
              <dt>Recorded permission</dt>
              <dd>{String(review.permission_status ?? "unknown")}</dd>
              <dt>Asset fingerprint</dt>
              <dd className="mono wrap-anywhere">
                {String(review.asset_fingerprint)}
              </dd>
            </dl>
            <Field
              label="Rights review acknowledgment"
              hint="Record your review and any unresolved rights. This record does not establish legal permission."
            >
              <textarea
                rows={3}
                value={ack}
                onChange={(e) => setAck(e.target.value)}
                placeholder="What have you reviewed for this intended use?"
              />
            </Field>
            <label className="check-label">
              <input
                type="checkbox"
                checked={confirmed}
                onChange={(e) => setConfirmed(e.target.checked)}
              />
              I confirm this exact asset, destination, account, caption,
              visibility, schedule and time zone for manual handoff.
            </label>
            {reviewError && <ErrorBox error={reviewError} />}
            <button
              className="button primary"
              disabled={busy || !confirmed || !ack.trim()}
              onClick={async () => {
                setBusy(true);
                try {
                  await post("/publications/" + review.id + "/approve", {
                    draft: review.draft,
                    asset_fingerprint: review.asset_fingerprint,
                    rights_acknowledgment: ack,
                    confirm_manual_handoff: true,
                  });
                  await refresh();
                  setReview(null);
                } catch (e) {
                  setReviewError(message(e));
                } finally {
                  setBusy(false);
                }
              }}
            >
              {busy ? "Saving review…" : "Approve manual handoff only"}
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}

const MODEL_NAMES: Record<string, string> = {
  "tiny.en": "Whisper tiny.en",
  tiny: "Whisper tiny · multilingual",
  minilm: "MiniLM sentence embeddings",
  "clip-vit-b32": "CLIP ViT-B/32",
  "xclip-b16-k600": "X-CLIP ViT-B/16 · Kinetics-600",
  "ppocr-v4": "PP-OCRv4",
  "ppocr-v5-mobile": "PP-OCRv5 mobile",
};

function Models({
  onAction,
  jobs,
}: {
  onAction: (f: () => Promise<unknown>, s?: string) => Promise<void>;
  jobs: Job[];
}) {
  const [data, setData] = useState<Record<string, unknown>>({}),
    [error, setError] = useState("");
  useEffect(() => {
    const refresh = () =>
      api<Record<string, unknown>>("/capabilities")
        .then((d) => setData(d && typeof d === "object" ? d : {}))
        .catch((e) => setError(message(e)));
    void refresh();
    const timer = setInterval(() => {
      if (!document.hidden) void refresh();
    }, 3000);
    return () => clearInterval(timer);
  }, []);
  const models = Array.isArray(data.models)
    ? data.models
    : Array.isArray((data.models as Record<string, unknown>)?.models)
      ? ((data.models as Record<string, unknown>).models as unknown[])
      : [];
  const downloads = jobs.filter((j) => j.kind === "model-download");
  const ready = models.filter(
    (m) => (m as Record<string, unknown>).status === "ready",
  ).length;
  return (
    <div className="page-content">
      <div className="page-heading">
        <div>
          <div className="eyebrow">Intelligence, with a clear source</div>
          <h1>
            AI & models<span className="heading-dot">.</span>
          </h1>
          <p>Small local models. Explicit downloads. Transparent limits.</p>
        </div>
        <Badge>CPU conservative</Badge>
      </div>
      {error && <ErrorBox error={error} />}
      <div className="mode-grid">
        <article>
          <Icon name="spark" />
          <h2>Local inference</h2>
          <p>
            Transcription and semantic suggestions run on this machine after
            their models are downloaded. Manual editing always stays available.
          </p>
          <Badge tone="accent">No paid API required</Badge>
        </article>
        <article>
          <Icon name="file" />
          <h2>Agent assisted</h2>
          <p>
            Export timestamped evidence for an operator. Import schema-validated
            decisions. External agents are not connected.
          </p>
          <Badge>Auditable exchange</Badge>
        </article>
        <article>
          <Icon name="settings" />
          <h2>Cloud / BYOK</h2>
          <p>
            Cloud analysis needs configured credentials, approved transfer,
            price estimates, and a hard budget. No subscription API access is
            assumed.
          </p>
          <Badge>Not configured</Badge>
        </article>
      </div>
      <div className="section-kicker section-title-row">
        <h2 className="section-title">Model catalog</h2>
        <span>{models.length ? `${ready} of ${models.length} ready` : ""}</span>
      </div>
      {models.length ? (
        models.map((m, i) => {
          const x = m as Record<string, unknown>;
          const id = String(x.id ?? x.name ?? i);
          const job = downloads.find(
            (j) => String(j.params?.name ?? "") === id,
          );
          const downloading = job ? ACTIVE.includes(job.status) : false;
          const failed =
            job && job.status === "failed" && x.status !== "ready" ? job : null;
          return (
            <div className="model-row" key={id}>
              <div className="model-symbol">
                <Icon name="spark" />
              </div>
              <div>
                <h3>
                  {MODEL_NAMES[id] ?? String(x.name ?? id)} <code>{id}</code>
                </h3>
                <p>{String(x.description ?? x.capability ?? "")}</p>
                <small>
                  {bytes(Number(x.download_bytes ?? x.size_bytes ?? 0))}{" "}
                  download · {bytes(Number(x.disk_required_bytes ?? 0))} free
                  disk needed · {String(x.license ?? "See model license notes")}
                </small>
                {failed && (
                  <p className="job-error">
                    Last download failed:{" "}
                    {typeof failed.error === "string"
                      ? failed.error
                      : JSON.stringify(failed.error)}
                  </p>
                )}
              </div>
              <Badge
                tone={
                  x.status === "ready"
                    ? "success"
                    : downloading
                      ? "accent"
                      : "muted"
                }
              >
                {x.status === "ready"
                  ? "Ready"
                  : downloading
                    ? job?.status === "running"
                      ? "Downloading…"
                      : "Download queued"
                    : x.runtime_available === false
                      ? "Runtime unavailable"
                      : "Download required"}
              </Badge>
              {x.status !== "ready" && (
                <button
                  className="button secondary small"
                  disabled={downloading}
                  onClick={() =>
                    void onAction(
                      () =>
                        post("/models/" + encodeURIComponent(id) + "/download"),
                      "Model download queued; check Job queue.",
                    )
                  }
                >
                  <Icon name="download" size={15} />
                  {failed ? "Retry download" : "Download"}
                </button>
              )}
            </div>
          );
        })
      ) : (
        <div className="info-note">
          <Icon name="info" size={16} />
          <p>
            Model availability is reported by the local API. Open the capability
            details below for this installation.
          </p>
        </div>
      )}
      <details>
        <summary>Runtime capabilities and model provenance</summary>
        <pre>{JSON.stringify(data, null, 2)}</pre>
      </details>
    </div>
  );
}

function GlobalExports({
  exports,
  titles,
  onAction,
  onOpen,
}: {
  exports: Export[];
  titles: Map<string, string>;
  onAction: (f: () => Promise<unknown>, s?: string) => Promise<void>;
  onOpen: (id: string) => void;
}) {
  const groups = new Map<string, Export[]>();
  for (const item of exports) {
    const key = String(item.project_id);
    groups.set(key, [...(groups.get(key) ?? []), item]);
  }
  return (
    <div className="page-content">
      <div className="page-heading">
        <div>
          <div className="eyebrow">Ready for your next step</div>
          <h1>
            Exports<span className="heading-dot">.</span>
          </h1>
          <p>Playable video, subtitles, and a complete handoff package.</p>
        </div>
        <Badge>
          {exports.length} {exports.length === 1 ? "export" : "exports"}
        </Badge>
      </div>
      {exports.length ? (
        [...groups].map(([id, items]) => {
          const title = String(
            items[0].project_title ??
              items[0].title ??
              titles.get(id) ??
              "Project " + id.slice(0, 8),
          );
          return (
            <section key={id} className="export-group">
              <div className="row between wrap">
                <h2>{title}</h2>
                <button className="text-button" onClick={() => onOpen(id)}>
                  Open project <Icon name="arrow" size={14} />
                </button>
              </div>
              <ExportCards
                project={{
                  id,
                  title,
                  exports: items,
                  source: { provenance: { permission_status: "unknown" } },
                }}
                onAction={onAction}
              />
            </section>
          );
        })
      ) : (
        <Empty icon="download" title="Your finished clips will live here.">
          Open a source, set your edit, and queue a render. Downloads appear
          only when the worker has finished.
        </Empty>
      )}
    </div>
  );
}

const PLAN_STATUS: Record<string, [string, string]> = {
  review_only: ["Review only", "success"],
  maintenance_required: ["Maintenance required before any cleanup", "warn"],
  blocked: ["Plan blocked", "danger"],
};
const sum = (record: Record<string, number> | undefined) =>
  Object.values(record ?? {}).reduce((a, b) => a + b, 0);

function StoragePage({ onError }: { onError: (s: string) => void }) {
  const [summary, setSummary] = useState<StorageSummary | null>(null),
    [summaryError, setSummaryError] = useState(""),
    [plan, setPlan] = useState<StoragePlan | null>(null),
    [planError, setPlanError] = useState(""),
    [planning, setPlanning] = useState(false),
    [verifying, setVerifying] = useState(false),
    [verifiedPlan, setVerifiedPlan] = useState<{
      fingerprint: string;
      result: PlanVerification;
    } | null>(null);
  const planRequests = useRef(createRequestScope());
  const verificationRequests = useRef(createRequestScope());
  const verification =
    verifiedPlan?.fingerprint === plan?.plan_fingerprint
      ? (verifiedPlan?.result ?? null)
      : null;
  const loadSummary = useCallback(async () => {
    try {
      setSummary(await api<StorageSummary>("/maintenance/storage"));
      setSummaryError("");
    } catch (e) {
      setSummaryError(message(e));
    }
  }, []);
  const loadPlan = useCallback(async () => {
    verificationRequests.current.invalidate();
    setVerifying(false);
    setVerifiedPlan(null);
    setPlanning(true);
    await planRequests.current.run(
      () => api<StoragePlan>("/maintenance/storage-plan"),
      {
        success: (next) => {
          setPlan(next);
          setPlanError("");
        },
        error: (error) => setPlanError(message(error)),
        settled: () => setPlanning(false),
      },
    );
  }, []);
  useEffect(() => {
    void loadSummary();
    void loadPlan();
    return () => {
      planRequests.current.invalidate();
      verificationRequests.current.invalidate();
    };
  }, [loadSummary, loadPlan]);
  const verify = async () => {
    if (!plan || planning) return;
    const submittedPlan = plan;
    setVerifying(true);
    setVerifiedPlan(null);
    await verificationRequests.current.run(
      () => post<PlanVerification>("/maintenance/verify-plan", submittedPlan),
      {
        success: (result) =>
          setVerifiedPlan({
            fingerprint: submittedPlan.plan_fingerprint,
            result,
          }),
        error: (error) => onError(message(error)),
        settled: () => setVerifying(false),
      },
    );
  };
  const disk = summary?.disk ?? null;
  const library = summary
    ? sum(summary.directories) +
      sum(summary.other_directories) +
      summary.database_bytes
    : 0;
  const inventory = plan?.inventory ?? {};
  const [statusLabel, statusTone] = plan
    ? (PLAN_STATUS[plan.status] ?? [plan.status, "muted"])
    : ["Planning…", "muted"];
  const rows: [string, number, string][] = summary
    ? [
        [
          "Projects",
          summary.directories.projects ?? 0,
          "originals, caches and exports",
        ],
        ["Models", summary.directories.models ?? 0, "pinned local model files"],
        [
          "Uploads",
          summary.directories.uploads ?? 0,
          "staged and resumable uploads",
        ],
        [
          "Logs",
          summary.directories.logs ?? 0,
          "job logs and worker heartbeats",
        ],
        ["Database", summary.database_bytes, "SQLite library and journal"],
        ...Object.entries(summary.other_directories).map(
          ([name, size]): [string, number, string] => [
            name,
            size,
            "other directory inside the library",
          ],
        ),
      ]
    : [];
  return (
    <div className="page-content">
      <div className="page-heading">
        <div>
          <div className="eyebrow">Nothing here deletes</div>
          <h1>
            Storage<span className="heading-dot">.</span>
          </h1>
          <p>
            Read-only inventory and an advisory retention plan for this library.
          </p>
        </div>
        <Badge>Read-only</Badge>
      </div>
      <div className="info-note">
        <Icon name="info" size={16} />
        <p>
          There is no delete, move or archive operation in the app. The plan
          below is an observation for operator review; originals, referenced
          revisions, exports, publications and active jobs are always protected.
        </p>
      </div>
      {summaryError && <ErrorBox error={summaryError} />}
      <div className="storage-grid">
        <article className="stat-card">
          <div className="section-kicker">
            <span>Disk</span>
          </div>
          {disk ? (
            <>
              <strong>{bytes(disk.free)} free</strong>
              <div
                className="meter"
                role="img"
                aria-label={`${bytes(disk.used)} used of ${bytes(disk.total)}`}
              >
                <span
                  style={{
                    width: `${Math.min(100, (disk.used / Math.max(1, disk.total)) * 100)}%`,
                  }}
                />
              </div>
              <small>
                {bytes(disk.used)} used of {bytes(disk.total)} on the volume
                holding the library
              </small>
            </>
          ) : (
            <strong>—</strong>
          )}
        </article>
        <article className="stat-card">
          <div className="section-kicker">
            <span>Library</span>
          </div>
          <strong>{summary ? bytes(library) : "—"}</strong>
          <small>
            {summary
              ? `${summary.file_count.toLocaleString()} files · hardlinks counted once${summary.symlinks_skipped ? ` · ${summary.symlinks_skipped} symlinks skipped` : ""}${summary.complete ? "" : " · scan incomplete"}`
              : "Measuring…"}
          </small>
          <code className="path">{summary?.data_directory ?? ""}</code>
        </article>
        <article className="stat-card">
          <div className="section-kicker">
            <span>Reclaimable estimate</span>
          </div>
          <strong>
            {plan ? bytes(plan.estimated_reclaimable_bytes) : "—"}
          </strong>
          <small>
            {plan
              ? `${plan.review_candidates.length} review candidate${plan.review_candidates.length === 1 ? "" : "s"} · ${plan.protected_paths.length} protected path${plan.protected_paths.length === 1 ? "" : "s"}`
              : "Waiting for the plan"}
          </small>
        </article>
      </div>
      {rows.length > 0 && (
        <table className="storage-table">
          <thead>
            <tr>
              <th>Area</th>
              <th>Holds</th>
              <th className="num">Size</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, size, what]) => (
              <tr key={name}>
                <td>{name}</td>
                <td className="subtle">{what}</td>
                <td className="num mono">{bytes(size)}</td>
              </tr>
            ))}
            {inventory.project_originals && (
              <tr className="sub">
                <td>↳ Originals</td>
                <td className="subtle">
                  {inventory.project_originals.file_count} immutable source
                  files
                </td>
                <td className="num mono">
                  {bytes(inventory.project_originals.unique_inode_bytes)}
                </td>
              </tr>
            )}
            {inventory.project_cache && (
              <tr className="sub">
                <td>↳ Caches</td>
                <td className="subtle">
                  {inventory.project_cache.file_count} proxy, analysis and
                  render stage files
                </td>
                <td className="num mono">
                  {bytes(inventory.project_cache.unique_inode_bytes)}
                </td>
              </tr>
            )}
            {inventory.project_exports && (
              <tr className="sub">
                <td>↳ Exports</td>
                <td className="subtle">
                  {inventory.project_exports.file_count} verified package files
                </td>
                <td className="num mono">
                  {bytes(inventory.project_exports.unique_inode_bytes)}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      )}
      <section className="plan-section">
        <div className="section-kicker section-title-row">
          <h2 className="section-title">Retention plan</h2>
          <span>{plan ? `Generated ${when(plan.generated_at)}` : ""}</span>
        </div>
        <div className="row wrap plan-toolbar">
          <Badge tone={statusTone}>{statusLabel}</Badge>
          <button
            className="button secondary small"
            disabled={planning}
            onClick={() => {
              void loadSummary();
              void loadPlan();
            }}
          >
            <Icon name="refresh" size={14} />
            {planning ? "Scanning…" : "Refresh plan"}
          </button>
          <button
            className="button secondary small"
            disabled={!plan || verifying || planning}
            onClick={() => void verify()}
          >
            <Icon name="check" size={14} />
            {verifying ? "Verifying…" : "Verify this plan"}
          </button>
          {plan && (
            <a
              className="text-link"
              href={
                "data:application/json;charset=utf-8," +
                encodeURIComponent(JSON.stringify(plan, null, 2))
              }
              download="storage-plan.json"
            >
              <Icon name="download" size={14} /> Download plan JSON
            </a>
          )}
        </div>
        {planError && <ErrorBox error={planError} />}
        {plan && (
          <>
            {plan.blockers.length > 0 && (
              <div className="plan-box danger">
                <strong>The plan could not be completed safely.</strong>
                <ul>
                  {plan.blockers.map((b) => (
                    <li key={b}>{b}</li>
                  ))}
                </ul>
                <p className="tiny">
                  A blocked plan reports nothing reclaimable. Symlinks and
                  special files inside the data directory stop the scan by
                  design; move them out of the library to plan again.
                </p>
              </div>
            )}
            {plan.maintenance_reasons.length > 0 && (
              <div className="plan-box warn">
                <strong>Before any future cleanup</strong>
                <ul>
                  {plan.maintenance_reasons.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              </div>
            )}
            {verification && (
              <div
                className={`plan-box verify-result ${verification.valid ? "ok" : "warn"}`}
                role="status"
              >
                <strong>
                  {verification.valid
                    ? "Plan still matches the library"
                    : "Plan is stale or could not be verified"}
                </strong>
                {verification.reasons.length > 0 && (
                  <ul>
                    {verification.reasons.map((r) => (
                      <li key={r}>{r}</li>
                    ))}
                  </ul>
                )}
                <p className="tiny subtle">{verification.meaning}</p>
              </div>
            )}
            <h3 className="plan-subtitle">
              Review candidates{" "}
              <span className="subtle">{plan.review_candidates.length}</span>
            </h3>
            {plan.review_candidates.length ? (
              <table className="storage-table">
                <thead>
                  <tr>
                    <th>Path</th>
                    <th>Why it is a candidate</th>
                    <th className="num">Files</th>
                    <th className="num">Size</th>
                  </tr>
                </thead>
                <tbody>
                  {plan.review_candidates.map((c) => (
                    <tr key={c.path}>
                      <td className="mono path">{c.path}</td>
                      <td className="subtle">
                        <Badge>{humanize(c.kind)}</Badge> {c.reason}
                      </td>
                      <td className="num mono">{c.file_count}</td>
                      <td className="num mono">{bytes(c.logical_bytes)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="subtle quiet">
                {plan.status === "blocked"
                  ? "Not evaluated: the scan was blocked."
                  : "No unreferenced caches or stale temporaries were found."}
              </p>
            )}
            {plan.retained_caches.length > 0 && (
              <details>
                <summary>
                  Caches kept despite being unreferenced (
                  {plan.retained_caches.length})
                </summary>
                <ul className="plan-list">
                  {plan.retained_caches.map((r) => (
                    <li key={r.path}>
                      <code className="path">{r.path}</code> — {r.reason}
                    </li>
                  ))}
                </ul>
              </details>
            )}
            <details>
              <summary>Protected paths ({plan.protected_paths.length})</summary>
              <ul className="plan-list">
                {plan.protected_paths.slice(0, 300).map((item) => (
                  <li key={item.path}>
                    <code className="path">{item.path}</code> —{" "}
                    {item.reasons.join("; ")}
                  </li>
                ))}
                {plan.protected_paths.length > 300 && (
                  <li className="subtle">
                    Showing 300 of {plan.protected_paths.length}; download the
                    plan for the full list.
                  </li>
                )}
              </ul>
            </details>
            <details>
              <summary>Limits of this plan</summary>
              <ul className="plan-list">
                {plan.limitations.map((l) => (
                  <li key={l}>{l}</li>
                ))}
              </ul>
              <p className="tiny subtle mono wrap-anywhere">
                plan {plan.plan_fingerprint} · snapshot{" "}
                {plan.snapshot_fingerprint ?? "none (blocked or incomplete)"}
              </p>
            </details>
          </>
        )}
        {!plan && !planError && (
          <p className="subtle quiet">Scanning the library read-only…</p>
        )}
      </section>
    </div>
  );
}

const AUDIT_PAGE = 100;
function AuditPage({
  titles,
  onOpen,
  onError,
}: {
  titles: Map<string, string>;
  onOpen: (id: string) => void;
  onError: (s: string) => void;
}) {
  const [entries, setEntries] = useState<AuditEntry[]>([]),
    [group, setGroup] = useState(""),
    [projectFilter, setProjectFilter] = useState(""),
    [loading, setLoading] = useState(true),
    [more, setMore] = useState(false);
  const requests = useRef(createRequestScope());
  const load = useCallback(
    async (before?: number) => {
      setLoading(true);
      if (!before) {
        setEntries([]);
        setMore(false);
      }
      const params = new URLSearchParams({ limit: String(AUDIT_PAGE) });
      if (group) params.set("event", group);
      if (projectFilter) params.set("project_id", projectFilter);
      if (before) params.set("before", String(before));
      await requests.current.run(
        () => api<AuditEntry[]>("/audit?" + params.toString()),
        {
          success: (rows) => {
            setEntries((prev) => (before ? [...prev, ...rows] : rows));
            setMore(rows.length === AUDIT_PAGE);
          },
          error: (error) => onError(message(error)),
          settled: () => setLoading(false),
        },
      );
    },
    [group, projectFilter, onError],
  );
  useEffect(() => {
    void load();
    return () => requests.current.invalidate();
  }, [load]);
  const projectOptions = [...titles.entries()].sort((a, b) =>
    a[1].localeCompare(b[1]),
  );
  return (
    <div className="page-content">
      <div className="page-heading">
        <div>
          <div className="eyebrow">Append-only</div>
          <h1>
            Audit trail<span className="heading-dot">.</span>
          </h1>
          <p>
            Every import, save, approval, job transition and recovery the local
            database recorded, newest first. Entries cannot be edited or deleted
            from the app.
          </p>
        </div>
        <Badge>Read-only</Badge>
      </div>
      <div className="audit-toolbar">
        <select
          aria-label="Filter audit by event type"
          value={group}
          onChange={(e) => setGroup(e.target.value)}
        >
          {AUDIT_GROUPS.map(([prefix, label]) => (
            <option key={prefix} value={prefix}>
              {label}
            </option>
          ))}
        </select>
        <select
          aria-label="Filter audit by project"
          value={projectFilter}
          onChange={(e) => setProjectFilter(e.target.value)}
        >
          <option value="">All projects</option>
          {projectOptions.map(([id, title]) => (
            <option key={id} value={id}>
              {title}
            </option>
          ))}
        </select>
        <button
          className="button secondary small"
          onClick={() => void load()}
          disabled={loading}
        >
          <Icon name="refresh" size={15} />
          Refresh
        </button>
        <span className="count" role="status">
          {loading && !entries.length
            ? "Loading…"
            : `${entries.length}${more ? "+" : ""} ${entries.length === 1 ? "entry" : "entries"}`}
        </span>
      </div>
      {!loading && !entries.length ? (
        <Empty icon="history" title="Nothing recorded yet.">
          Imports, saves, jobs and approvals appear here as they happen. Widen
          the filters to see more.
        </Empty>
      ) : (
        <table className="audit-table">
          <thead>
            <tr>
              <th>When</th>
              <th>Event</th>
              <th>Project</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => {
              const title = e.project_id ? titles.get(e.project_id) : null;
              const summary = auditSummary(e.event, e.detail ?? {}, seconds);
              return (
                <tr key={e.seq} data-seq={e.seq}>
                  <td className="time">{when(e.created_at)}</td>
                  <td className="event">
                    <Badge tone={auditTone(e.event)}>{e.event}</Badge>
                  </td>
                  <td>
                    {e.project_id ? (
                      <button
                        className="text-button"
                        onClick={() => onOpen(e.project_id!)}
                        title={e.project_id}
                      >
                        {title ?? shortId(e.project_id)}
                      </button>
                    ) : (
                      <span className="subtle">Workspace</span>
                    )}
                  </td>
                  <td className="details">
                    {summary || <span className="subtle">—</span>}
                    {Object.keys(e.detail ?? {}).length > 0 && (
                      <details>
                        <summary>Full record</summary>
                        <pre tabIndex={0}>
                          {JSON.stringify(e.detail, null, 2)}
                        </pre>
                      </details>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {more && (
        <div className="audit-more">
          <button
            className="button secondary"
            disabled={loading}
            onClick={() => void load(entries[entries.length - 1]?.seq)}
          >
            Show older entries
          </button>
        </div>
      )}
    </div>
  );
}
