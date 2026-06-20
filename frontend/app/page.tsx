"use client";
import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  getDatasets, createSession, runAutoML, getAutoMLStatus,
  runAutoMLForDataset, uploadDataset,
  Dataset, AutoMLStatus,
} from "@/lib/api";
import {
  Brain, Database, Loader2, Rows3, Columns3,
  Upload, Sparkles, CheckCircle2, AlertTriangle, ArrowRight,
} from "lucide-react";

export default function Home() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState<string | null>(null);
  const [runningAutoML, setRunningAutoML] = useState<string | null>(null);
  const [tab, setTab] = useState<"playground" | "automl">("playground");
  const [lastSession, setLastSession] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    getDatasets()
      .then(setDatasets)
      .catch(() => setError("Could not load datasets. Is the backend running on port 8000?"))
      .finally(() => setLoading(false));
    const last = localStorage.getItem("ml_last_session");
    if (last) setLastSession(last);
  }, []);

  async function start(datasetId: string) {
    setStarting(datasetId);
    setError(null);
    try {
      const session = await createSession(datasetId);
      localStorage.setItem("ml_last_session", session.id);
      router.push(`/playground/${session.id}`);
    } catch (e: any) {
      if (e.message?.includes("not found")) {
        localStorage.removeItem("ml_last_session");
        setLastSession(null);
      }
      setError(e.message || "Failed to start session.");
      setStarting(null);
    }
  }

  async function handleRunAutoMLForDataset(datasetId: string) {
    setRunningAutoML(datasetId);
    setError(null);
    try {
      const result = await runAutoMLForDataset(datasetId);
      // Store job_id then navigate to AutoML tab to show results
      localStorage.setItem("ml_automl_job", result.job_id);
      setTab("automl");
    } catch (e: any) {
      setError(e.message || "Failed to start AutoML.");
    } finally {
      setRunningAutoML(null);
    }
  }

  return (
    <main className="h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 flex flex-col overflow-hidden">
      <div className="max-w-5xl mx-auto px-6 w-full flex flex-col h-full py-8">

        {/* Header */}
        <div className="text-center mb-6 flex-shrink-0">
          <div className="flex items-center justify-center gap-3 mb-3">
            <div className="p-2.5 rounded-2xl bg-blue-500/10 border border-blue-500/20">
              <Brain className="w-7 h-7 text-blue-400" />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-white mb-2 tracking-tight">ML Playground</h1>
          <p className="text-slate-400 text-sm max-w-lg mx-auto">
            Walk through the complete ML lifecycle step by step. Every decision explained by AI.
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 px-4 py-3 mb-4 rounded-xl
            bg-red-500/10 border border-red-500/20 text-red-400 text-sm flex-shrink-0">
            <AlertTriangle className="w-4 h-4 flex-shrink-0" />
            {error}
          </div>
        )}

        {/* Resume */}
        {lastSession && (
          <div className="flex justify-center mb-4 flex-shrink-0">
            <button
              onClick={() => router.push(`/playground/${lastSession}`)}
              className="flex items-center gap-2 text-xs px-4 py-2 rounded-lg
                border border-slate-700 text-slate-400
                hover:text-white hover:border-slate-600 transition-colors">
              <ArrowRight className="w-3.5 h-3.5" />
              Resume last session
            </button>
          </div>
        )}

        {/* Tabs */}
        <div className="flex justify-center mb-5 flex-shrink-0">
          <div className="flex gap-1 bg-slate-800/60 p-1 rounded-xl border border-slate-700/50">
            {(["playground", "automl"] as const).map(t => (
              <button key={t} onClick={() => setTab(t)}
                className={`px-5 py-2 rounded-lg text-sm font-medium transition-all ${tab === t
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"}`}>
                {t === "playground" ? "🎮 Guided Playground" : "⚡ AutoML"}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {tab === "playground" && (
            <PlaygroundTab
              datasets={datasets}
              loading={loading}
              starting={starting}
              runningAutoML={runningAutoML}
              onStart={start}
              onRunAutoML={handleRunAutoMLForDataset}
            />
          )}
          {tab === "automl" && <AutoMLTab />}
        </div>

        <p className="text-center text-xs text-slate-600 mt-3 flex-shrink-0">
          13-step ML lifecycle · AI explanations · Python export · AutoML
        </p>
      </div>
    </main>
  );
}

// ─── PLAYGROUND TAB ───────────────────────────────────────────────────────────

function PlaygroundTab({ datasets, loading, starting, runningAutoML, onStart, onRunAutoML }: {
  datasets: Dataset[];
  loading: boolean;
  starting: string | null;
  runningAutoML: string | null;
  onStart: (id: string) => void;
  onRunAutoML: (id: string) => void;
}) {
  const [uploading, setUploading] = useState(false);
  const [uploadCols, setUploadCols] = useState<string[]>([]);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [targetCol, setTargetCol] = useState("");
  const [datasetName, setDatasetName] = useState("");
  const [showUploadForm, setShowUploadForm] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setUploadFile(f);
    setDatasetName(f.name.replace(/\.(csv|xlsx|xls)$/, ""));
    setUploadError(null);
    try {
      const text = await f.text();
      const cols = text.split("\n")[0].split(",").map(c => c.trim().replace(/"/g, ""));
      setUploadCols(cols);
      setTargetCol(cols[cols.length - 1]);
      setShowUploadForm(true);
    } catch { setUploadCols([]); }
  }

  async function handleUpload() {
    if (!uploadFile || !targetCol) return;
    setUploading(true);
    setUploadError(null);
    try {
      const fd = new FormData();
      fd.append("file", uploadFile);
      fd.append("target_col", targetCol);
      fd.append("dataset_name", datasetName || uploadFile.name);
      const result = await uploadDataset(fd);
      onStart(result.dataset_id);
    } catch (e: any) {
      setUploadError(e.message || "Upload failed.");
      setUploading(false);
    }
  }

  if (loading) return (
    <div className="flex justify-center py-16">
      <Loader2 className="w-7 h-7 animate-spin text-blue-400" />
    </div>
  );

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 h-full content-start overflow-y-auto pb-2">

      {/* Dataset cards */}
      {datasets.map(ds => (
        <div key={ds.id}
          className="bg-slate-800/60 border border-slate-700/60 hover:border-blue-500/50
            hover:bg-slate-800 transition-all duration-200 group rounded-2xl p-4
            cursor-pointer flex flex-col"
          onClick={() => onStart(ds.id)}>
          <div className="flex items-start justify-between mb-3">
            <div className="p-1.5 rounded-lg bg-slate-700/50 group-hover:bg-blue-500/10 transition-colors">
              <Database className="w-4 h-4 text-slate-400 group-hover:text-blue-400 transition-colors" />
            </div>
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium border ${ds.task === "classification"
                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                : "bg-amber-500/10 text-amber-400 border-amber-500/20"}`}>
              {ds.task}
            </span>
          </div>
          <h3 className="text-white font-semibold text-base mb-0.5">{ds.name}</h3>
          <p className="text-xs text-slate-500 font-mono mb-3">target: {ds.target}</p>
          <div className="flex items-center gap-3 text-xs text-slate-400 mb-4">
            <span className="flex items-center gap-1"><Rows3 className="w-3 h-3" />{ds.rows.toLocaleString()}</span>
            <span className="flex items-center gap-1"><Columns3 className="w-3 h-3" />{ds.cols} cols</span>
          </div>
          <div className="mt-auto flex flex-col gap-2">
            <div className="w-full flex items-center justify-center gap-2 py-2 rounded-lg
              bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-colors">
              {starting === ds.id
                ? <><Loader2 className="w-3.5 h-3.5 animate-spin" />Starting...</>
                : "Start Playground →"}
            </div>
            <button
              onClick={e => { e.stopPropagation(); onRunAutoML(ds.id); }}
              disabled={runningAutoML === ds.id}
              className="w-full flex items-center justify-center gap-2 py-2 rounded-lg
                bg-slate-700 hover:bg-slate-600 text-slate-300 text-sm font-medium
                transition-colors disabled:opacity-40">
              {runningAutoML === ds.id
                ? <><Loader2 className="w-3.5 h-3.5 animate-spin" />Running AutoML...</>
                : <><Sparkles className="w-3.5 h-3.5" />Run AutoML</>}
            </button>
          </div>
        </div>
      ))}

      {/* Upload card */}
      {!showUploadForm ? (
        <div
          onClick={() => fileRef.current?.click()}
          className="bg-slate-800/30 border-2 border-dashed border-slate-700
            hover:border-blue-500/50 hover:bg-slate-800/50 transition-all duration-200
            group rounded-2xl p-4 cursor-pointer flex flex-col items-center justify-center min-h-48">
          <div className="p-3 rounded-xl bg-slate-700/50 group-hover:bg-blue-500/10 transition-colors mb-3">
            <Upload className="w-6 h-6 text-slate-500 group-hover:text-blue-400 transition-colors" />
          </div>
          <h3 className="text-slate-300 font-semibold text-base mb-1">Upload Your Dataset</h3>
          <p className="text-xs text-slate-500 text-center">CSV or Excel · Max 50MB</p>
          <input ref={fileRef} type="file" accept=".csv,.xlsx,.xls"
            onChange={handleFileChange} className="hidden" />
        </div>
      ) : (
        <div className="bg-slate-800/60 border border-blue-500/30 rounded-2xl p-4 flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-white font-semibold text-sm">Configure Dataset</h3>
            <button
              onClick={() => { setShowUploadForm(false); setUploadFile(null); setUploadError(null); }}
              className="text-xs text-slate-500 hover:text-slate-300">
              Cancel
            </button>
          </div>
          {uploadError && (
            <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20
              rounded-lg px-3 py-2 mb-3">
              {uploadError}
            </div>
          )}
          <div className="space-y-3 flex-1">
            <div>
              <label className="text-xs text-slate-400 font-medium block mb-1">Dataset name</label>
              <input
                value={datasetName}
                onChange={e => setDatasetName(e.target.value)}
                className="w-full text-sm bg-slate-900 border border-slate-700 rounded-lg
                  px-3 py-2 text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 font-medium block mb-1">Target column</label>
              <select
                value={targetCol}
                onChange={e => setTargetCol(e.target.value)}
                className="w-full text-sm bg-slate-900 border border-slate-700 rounded-lg
                  px-3 py-2 text-white focus:outline-none focus:border-blue-500">
                {uploadCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          </div>
          <button
            onClick={handleUpload}
            disabled={uploading}
            className="mt-4 w-full flex items-center justify-center gap-2 py-2.5
              bg-blue-600 hover:bg-blue-500 disabled:opacity-40
              text-white text-sm font-semibold rounded-xl transition-colors">
            {uploading
              ? <><Loader2 className="w-4 h-4 animate-spin" />Uploading...</>
              : "Start Playground →"}
          </button>
        </div>
      )}
    </div>
  );
}

// ─── AUTOML TAB ───────────────────────────────────────────────────────────────

function AutoMLTab() {
  const [file, setFile] = useState<File | null>(null);
  const [targetCol, setTargetCol] = useState("");
  const [columns, setColumns] = useState<string[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<AutoMLStatus | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Resume any job from dataset card click
  useEffect(() => {
    const savedJob = localStorage.getItem("ml_automl_job");
    if (savedJob && !jobId) {
      setJobId(savedJob);
      setStatus({ status: "running", progress: 0, log: [] });
      localStorage.removeItem("ml_automl_job");
    }
  }, []);

  async function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setError(null);
    try {
      const text = await f.text();
      const cols = text.split("\n")[0].split(",").map(c => c.trim().replace(/"/g, ""));
      setColumns(cols);
      setTargetCol(cols[cols.length - 1]);
    } catch { setColumns([]); }
  }

  async function upload() {
    if (!file || !targetCol) return;
    setUploading(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("target_col", targetCol);
      const res = await runAutoML(fd);
      setJobId(res.job_id);
      setStatus({ status: "running", progress: 0, log: [] });
    } catch (e: any) {
      setError(e.message || "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  useEffect(() => {
    if (!jobId) return;
    pollRef.current = setInterval(async () => {
      try {
        const s = await getAutoMLStatus(jobId);
        setStatus(s);
        if (s.status === "done" || s.status === "error") {
          clearInterval(pollRef.current!);
        }
      } catch {
        clearInterval(pollRef.current!);
        setError("Lost connection to server.");
      }
    }, 1500);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [jobId]);

  function reset() {
    setJobId(null);
    setStatus(null);
    setFile(null);
    setColumns([]);
    setError(null);
    setTargetCol("");
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-xl mx-auto space-y-4">

        {error && (
          <div className="flex items-center gap-2 px-4 py-3 rounded-xl
            bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
            <AlertTriangle className="w-4 h-4 flex-shrink-0" />
            {error}
          </div>
        )}

        {/* Upload zone */}
        {!jobId && (
          <>
            <div
              onClick={() => fileRef.current?.click()}
              className="border-2 border-dashed border-slate-700 hover:border-blue-500/50
                rounded-2xl p-8 text-center cursor-pointer transition-colors group">
              <Upload className="w-8 h-8 text-slate-600 group-hover:text-blue-400
                mx-auto mb-3 transition-colors" />
              <p className="text-slate-300 font-medium mb-1">
                {file ? file.name : "Drop your CSV or Excel file here"}
              </p>
              <p className="text-xs text-slate-500">Max 50MB · CSV or Excel</p>
              <input ref={fileRef} type="file" accept=".csv,.xlsx,.xls"
                onChange={onFileChange} className="hidden" />
            </div>

            {columns.length > 0 && (
              <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-4">
                <label className="text-xs text-slate-400 font-medium block mb-2">
                  Target column
                </label>
                <select
                  value={targetCol}
                  onChange={e => setTargetCol(e.target.value)}
                  className="w-full text-sm bg-slate-900 border border-slate-700
                    rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-500">
                  {columns.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            )}

            <button
              onClick={upload}
              disabled={!file || !targetCol || uploading}
              className="w-full py-3 bg-blue-600 hover:bg-blue-500
                disabled:opacity-40 disabled:cursor-not-allowed
                text-white font-medium rounded-xl transition-colors
                flex items-center justify-center gap-2">
              {uploading
                ? <><Loader2 className="w-4 h-4 animate-spin" />Uploading...</>
                : <><Sparkles className="w-4 h-4" />Run AutoML</>}
            </button>

            <div className="bg-slate-800/40 border border-slate-700/40 rounded-xl p-4">
              <p className="text-xs text-slate-400 font-medium mb-2">What AutoML does:</p>
              <div className="space-y-1.5">
                {[
                  "Detects task type (classification / regression)",
                  "Drops columns with >50% missing values",
                  "Median imputation + StandardScaler",
                  "Label-encodes all categorical columns",
                  "Optuna hyperparameter search — 20 trials",
                  "Trains best Random Forest model",
                  "Returns metrics and feature importance",
                ].map((item, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs text-slate-500">
                    <span className="w-1 h-1 rounded-full bg-slate-600 flex-shrink-0" />
                    {item}
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {/* Progress */}
        {jobId && status?.status === "running" && (
          <div className="bg-slate-800/60 border border-slate-700/60 rounded-2xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
              <span className="text-sm font-medium text-white">AutoML running...</span>
              <span className="ml-auto text-sm font-mono text-blue-400">{status.progress}%</span>
            </div>
            <div className="h-2 bg-slate-700 rounded-full mb-4">
              <div
                className="h-2 bg-blue-500 rounded-full transition-all duration-500"
                style={{ width: `${status.progress}%` }}
              />
            </div>
            <div className="space-y-1">
              {status.log?.map((line, i) => (
                <p key={i} className="text-xs text-slate-400 font-mono">{line}</p>
              ))}
            </div>
          </div>
        )}

        {/* Error */}
        {status?.status === "error" && (
          <div className="bg-red-500/10 border border-red-500/20 rounded-2xl p-5">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-4 h-4 text-red-400" />
              <span className="text-sm font-medium text-red-400">AutoML failed</span>
            </div>
            <p className="text-xs text-red-400/70 mb-3">{status.error}</p>
            <button onClick={reset} className="text-xs text-slate-400 hover:text-white underline">
              Try again
            </button>
          </div>
        )}

        {/* Results */}
        {status?.status === "done" && (
          <div className="space-y-4">
            <div className="bg-slate-800/60 border border-slate-700/60 rounded-2xl p-5">
              <div className="flex items-center gap-2 mb-4">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-semibold text-white">AutoML complete</span>
                <span className={`ml-auto text-xs px-2 py-0.5 rounded-full ${status.task === "classification"
                    ? "bg-emerald-500/10 text-emerald-400"
                    : "bg-amber-500/10 text-amber-400"}`}>
                  {status.task}
                </span>
              </div>

              {/* Stats row */}
              <div className="grid grid-cols-3 gap-2 mb-4 text-center">
                {[
                  ["Rows", String(status.shape?.rows ?? "—")],
                  ["Features", String(status.n_features ?? "—")],
                  ["Trials", String(status.pipeline_summary?.n_trials ?? "—")],
                ].map(([l, v]) => (
                  <div key={l} className="bg-slate-900 rounded-xl p-3">
                    <div className="text-xs text-slate-500">{l}</div>
                    <div className="text-lg font-bold font-mono text-white mt-0.5">{v}</div>
                  </div>
                ))}
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-2 gap-2 mb-4">
                {Object.entries(status.metrics || {}).map(([k, v]) => (
                  <div key={k} className="bg-slate-900 rounded-xl p-3">
                    <div className="text-xs text-slate-500">{k.replace(/_/g, " ")}</div>
                    <div className="text-lg font-bold font-mono text-white mt-0.5">
                      {typeof v === "number" ? v.toFixed(4) : String(v)}
                    </div>
                  </div>
                ))}
              </div>

              {/* Pipeline summary */}
              {status.pipeline_summary && (
                <div className="bg-slate-900/60 rounded-xl p-3 mb-4">
                  <p className="text-xs text-slate-500 font-medium mb-2 uppercase tracking-wider">
                    Pipeline used
                  </p>
                  <div className="space-y-1.5">
                    {[
                      ["Missing values", status.pipeline_summary.imputation || "median"],
                      ["Scaling", status.pipeline_summary.scaling || "standard"],
                      ["Encoding", "label encoding"],
                      ["Model", `Random Forest (${status.pipeline_summary.n_trials} trials)`],
                      ["Best CV score", String(status.pipeline_summary.best_trial_score)],
                    ].map(([k, v]) => (
                      <div key={k} className="flex justify-between text-xs">
                        <span className="text-slate-500">{k}</span>
                        <span className="text-slate-300 font-mono">{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Feature importance */}
              {status.feature_importance && Object.keys(status.feature_importance).length > 0 && (
                <div>
                  <p className="text-xs text-slate-500 font-medium mb-2 uppercase tracking-wider">
                    Top features
                  </p>
                  <div className="space-y-2">
                    {Object.entries(status.feature_importance).slice(0, 8).map(([name, val], i) => {
                      const maxVal = Object.values(status.feature_importance!)[0] as number;
                      return (
                        <div key={name}>
                          <div className="flex justify-between mb-0.5">
                            <span className="text-xs text-slate-400 truncate pr-2">{name}</span>
                            <span className="text-xs font-mono text-slate-500">
                              {Number(val).toFixed(3)}
                            </span>
                          </div>
                          <div className="h-1.5 bg-slate-700 rounded-full">
                            <div className="h-1.5 rounded-full transition-all"
                              style={{
                                width: `${(Number(val) / maxVal) * 100}%`,
                                background: i === 0 ? "#3b82f6" : "#334155",
                              }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            <button onClick={reset}
              className="w-full py-2.5 bg-slate-800 hover:bg-slate-700
                text-slate-300 text-sm font-medium rounded-xl
                transition-colors border border-slate-700">
              Run another dataset
            </button>
          </div>
        )}
      </div>
    </div>
  );
}