"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
    getSession, getProfile, getDistributions, getCorrelation, getTargetAnalysis,
    getRuns, runStep, trainModel, tuneModel,
    Session, DatasetProfile, StepResponse, TrainResponse,
    EDADistributions, EDACorrelation, EDATargetAnalysis, RunRecord,
} from "@/lib/api";
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
    LineChart, Line, ScatterChart, Scatter, RadarChart, Radar,
    PolarGrid, PolarAngleAxis, AreaChart, Area,
} from "recharts";
import {
    Loader2, CheckCircle2, Circle, ChevronRight,
    Sparkles, AlertTriangle, Download, TrendingUp,
} from "lucide-react";

// ─── CONSTANTS ────────────────────────────────────────────────────────────────

const STEPS = [
    { id: 1, key: "profile", label: "Data profile" },
    { id: 2, key: "eda", label: "EDA" },
    { id: 3, key: "missing", label: "Missing values" },
    { id: 4, key: "outliers", label: "Outliers" },
    { id: 5, key: "features", label: "Feature engineering" },
    { id: 6, key: "encoding", label: "Encoding" },
    { id: 7, key: "selection", label: "Feature selection" },
    { id: 8, key: "scaling", label: "Scaling" },
    { id: 9, key: "train", label: "Train model" },
    { id: 10, key: "tune", label: "Hyperparameter tuning" },
    { id: 11, key: "explain", label: "Explainability" },
    { id: 12, key: "compare", label: "Experiment comparison" },
];

const TECHNIQUES: Record<string, { value: string; label: string; desc: string; params?: Record<string, any> }[]> = {
    missing: [
        { value: "mean", label: "Mean", desc: "Fill with column mean" },
        { value: "median", label: "Median", desc: "Fill with column median" },
        { value: "mode", label: "Mode", desc: "Fill with most frequent" },
        { value: "knn", label: "KNN", desc: "K-nearest neighbours", params: { n_neighbors: 5 } },
        { value: "mice", label: "MICE", desc: "Iterative imputation" },
        { value: "constant", label: "Constant", desc: "Fill with fixed value", params: { fill_value: 0 } },
        { value: "drop_rows", label: "Drop rows", desc: "Remove rows with nulls" },
        { value: "drop_cols", label: "Drop columns", desc: "Remove >50% missing cols", params: { threshold: 0.5 } },
    ],
    outliers: [
        { value: "iqr_cap", label: "IQR capping", desc: "Winsorize at 1.5×IQR" },
        { value: "zscore_remove", label: "Z-score", desc: "Remove rows beyond 3σ", params: { threshold: 3 } },
        { value: "log_transform", label: "Log transform", desc: "Compress skewed values" },
        { value: "keep", label: "Keep", desc: "Document only, no change" },
    ],
    features: [
        { value: "none", label: "None", desc: "Skip this step" },
        { value: "log_features", label: "Log features", desc: "log(x+1) for numerics" },
        { value: "polynomial", label: "Polynomial", desc: "Degree-2 interactions" },
        { value: "interaction", label: "Interactions", desc: "Pairwise products" },
        { value: "sqrt_features", label: "Square root", desc: "√x for numerics" },
        { value: "binning", label: "Binning", desc: "Discretise into bins", params: { n_bins: 5 } },
        { value: "ratio", label: "Ratios", desc: "Col A / Col B features" },
    ],
    encoding: [
        { value: "onehot", label: "One-hot", desc: "Binary indicator columns" },
        { value: "label", label: "Label", desc: "Integer codes (0,1,2…)" },
        { value: "ordinal", label: "Ordinal", desc: "Order-preserving integers" },
        { value: "frequency", label: "Frequency", desc: "Replace with value freq" },
        { value: "target", label: "Target mean", desc: "Replace with target mean" },
    ],
    selection: [
        { value: "none", label: "None", desc: "Keep all features" },
        { value: "variance_threshold", label: "Variance filter", desc: "Drop near-zero variance", params: { threshold: 0.01 } },
        { value: "correlation", label: "Correlation", desc: "Drop correlated >0.95", params: { threshold: 0.95 } },
        { value: "mutual_info", label: "Mutual info", desc: "Keep top-k by MI", params: { k: 10 } },
    ],
    scaling: [
        { value: "standard", label: "StandardScaler", desc: "Zero mean, unit variance" },
        { value: "minmax", label: "MinMax", desc: "Scale to [0, 1]" },
        { value: "robust", label: "Robust", desc: "IQR-based, outlier-safe" },
        { value: "maxabs", label: "MaxAbs", desc: "Scale by max absolute" },
        { value: "quantile", label: "Quantile", desc: "Uniform output distribution" },
        { value: "power", label: "PowerTransformer", desc: "Yeo-Johnson normalisation" },
        { value: "none", label: "None", desc: "Skip — tree models ok" },
    ],
    train: [
        { value: "random_forest", label: "Random Forest", desc: "Ensemble of decision trees" },
        { value: "xgboost", label: "XGBoost", desc: "Gradient boosted trees" },
        { value: "logistic_regression", label: "Logistic Regression", desc: "Linear classification" },
        { value: "lightgbm", label: "LightGBM", desc: "Fast gradient boosting" },
        { value: "decision_tree", label: "Decision Tree", desc: "Single tree, interpretable" },
        { value: "knn", label: "KNN", desc: "K-nearest neighbours" },
        { value: "naive_bayes", label: "Naive Bayes", desc: "Probabilistic classifier" },
        { value: "gradient_boosting", label: "Gradient Boosting", desc: "Sequential boosting" },
        { value: "ridge", label: "Ridge", desc: "L2 regularised regression" },
        { value: "lasso", label: "Lasso", desc: "L1 regularised regression" },
        { value: "elasticnet", label: "ElasticNet", desc: "L1+L2 combined" },
        { value: "svr", label: "SVR", desc: "Support vector regression" },
    ],
};

const STATS_KEYS = [
    "missing_before", "missing_after", "rows_before", "rows_after",
    "cols_before", "cols_after", "n_columns_scaled", "new_cols_created",
    "total_outliers_found", "new_features_created", "n_dropped",
];

const CHART_OPTIONS = [
    { value: "distribution", label: "Distribution" },
    { value: "missing", label: "Missing values" },
    { value: "correlation", label: "Correlations" },
    { value: "target", label: "Target analysis" },
];

const VIZ_TYPES = [
    { value: "bar", label: "Bar chart" },
    { value: "area", label: "Area chart" },
    { value: "line", label: "Line chart" },
    { value: "scatter", label: "Scatter plot" },
    { value: "radar", label: "Radar chart" },
];

const TOOLTIP_STYLE = {
    background: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: 8,
    fontSize: 11,
    color: "#cbd5e1",
};

// ─── MAIN PAGE ────────────────────────────────────────────────────────────────

export default function PlaygroundPage() {
    const { sessionId } = useParams<{ sessionId: string }>();
    const [session, setSession] = useState<Session | null>(null);
    const [profile, setProfile] = useState<DatasetProfile | null>(null);
    const [activeStep, setActiveStep] = useState(1);
    const [selected, setSelected] = useState<Record<string, string>>({});
    const [results, setResults] = useState<Record<string, any>>({});
    const [loading, setLoading] = useState<string | null>(null);
    const [distributions, setDist] = useState<EDADistributions | null>(null);
    const [correlation, setCorr] = useState<EDACorrelation | null>(null);
    const [targetData, setTarget] = useState<EDATargetAnalysis | null>(null);
    const [edaLoading, setEdaLoading] = useState(false);
    const [runs, setRuns] = useState<RunRecord[]>([]);

    useEffect(() => {
        if (!sessionId) return;
        getSession(sessionId).then(s => {
            setSession(s);
            return getProfile(s.dataset_id);
        }).then(setProfile);
    }, [sessionId]);

    useEffect(() => {
        if (!session || activeStep !== 2 || distributions) return;
        setEdaLoading(true);
        Promise.all([
            getDistributions(session.dataset_id),
            getCorrelation(session.dataset_id),
            getTargetAnalysis(session.dataset_id),
        ]).then(([d, c, t]) => { setDist(d); setCorr(c); setTarget(t); })
            .finally(() => setEdaLoading(false));
    }, [activeStep, session]);

    useEffect(() => {
        if (!session || activeStep !== 12) return;
        getRuns(session.id).then(setRuns);
    }, [activeStep, session]);

    async function apply(stepKey: string) {
        if (!session) return;
        setLoading(stepKey);
        try {
            if (stepKey === "train") {
                const model = selected["train"] || "random_forest";
                const res = await trainModel(session.id, model);
                setResults(r => ({ ...r, train: res }));
            } else if (stepKey === "tune") {
                const model = selected["train"] || "random_forest";
                const res = await tuneModel(session.id, model);
                setResults(r => ({ ...r, tune: res }));
            } else {
                const technique = selected[stepKey] || TECHNIQUES[stepKey][0].value;
                const techObj = TECHNIQUES[stepKey].find(t => t.value === technique);
                const res = await runStep(session.id, stepKey, technique, techObj?.params || {});
                setResults(r => ({ ...r, [stepKey]: res }));
            }
            if (stepKey !== "tune" && stepKey !== "compare") {
                setActiveStep(s => Math.min(s + 1, STEPS.length));
            }
        } catch (e) { console.error(e); }
        finally { setLoading(null); }
    }

    async function exportCode() {
        if (!session) return;
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/sessions/${session.id}/export`);
        const code = await res.text();
        const a = Object.assign(document.createElement("a"), {
            href: URL.createObjectURL(new Blob([code], { type: "text/plain" })),
            download: `ml_pipeline_${session.dataset_id}.py`,
        });
        a.click();
    }

    if (!session || !profile) return (
        <div className="min-h-screen bg-slate-950 flex items-center justify-center">
            <Loader2 className="w-7 h-7 animate-spin text-blue-400" />
        </div>
    );

    const stepResult = results[STEPS[activeStep - 1]?.key] as StepResponse | undefined;
    const trainResult = results["train"] as TrainResponse | undefined;
    const tuneResult = results["tune"] as TrainResponse | undefined;
    const regModels = ["ridge", "lasso", "elasticnet", "svr"];

    return (
        <div className="h-screen bg-slate-950 flex flex-col overflow-hidden">

            {/* TOP BAR */}
            <header className="h-11 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800
        border-b border-slate-800/80 flex items-center px-4 gap-3 flex-shrink-0">
                <span className="text-white font-semibold text-sm tracking-tight">ML Playground</span>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <div className="ml-auto flex items-center gap-3">
                    <span className="text-xs text-slate-500 capitalize font-medium">{session.dataset_id}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium border ${session.task_type === "classification"
                            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                            : "bg-amber-500/10 text-amber-400 border-amber-500/20"}`}>
                        {session.task_type}
                    </span>
                    <button onClick={exportCode}
                        className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg
              bg-blue-600/20 text-blue-300 border border-blue-500/30
              hover:bg-blue-600/30 transition-colors">
                        <Download className="w-3.5 h-3.5" /> Export .py
                    </button>
                </div>
            </header>

            <div className="flex flex-1 overflow-hidden">

                {/* LEFT SIDEBAR */}
                <aside className="w-52 bg-slate-900 border-r border-slate-800
          flex flex-col flex-shrink-0 overflow-y-auto">
                    <div className="px-3 pt-4 pb-2">
                        <p className="text-xs text-slate-600 uppercase tracking-widest mb-3 px-1">Lifecycle</p>
                        <nav className="space-y-0.5">
                            {STEPS.map(step => {
                                const done = ["profile", "eda"].includes(step.key)
                                    ? activeStep > step.id
                                    : !!results[step.key];
                                const active = activeStep === step.id;
                                return (
                                    <button key={step.id} onClick={() => setActiveStep(step.id)}
                                        className={`w-full flex items-center gap-2.5 px-2.5 py-2
                      rounded-lg text-xs transition-all text-left group ${active
                                                ? "bg-blue-600/20 text-blue-300 border border-blue-500/30"
                                                : done
                                                    ? "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
                                                    : "text-slate-600 hover:bg-slate-800/40 hover:text-slate-400"}`}>
                                        <span className="flex-shrink-0 w-4">
                                            {done
                                                ? <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                                                : active
                                                    ? <ChevronRight className="w-3.5 h-3.5 text-blue-400" />
                                                    : <Circle className="w-3.5 h-3.5" />}
                                        </span>
                                        <span className="flex-1 font-medium">{step.label}</span>
                                        <span className="text-slate-700 text-xs">{step.id}</span>
                                    </button>
                                );
                            })}
                        </nav>
                    </div>

                    {Object.keys(results).filter(k => !["train", "tune", "compare"].includes(k)).length > 0 && (
                        <div className="mx-3 mt-2 mb-4 p-3 rounded-xl bg-slate-800/40 border border-slate-700/40">
                            <p className="text-xs text-slate-600 mb-2 font-medium uppercase tracking-wider">Pipeline</p>
                            {Object.entries(results)
                                .filter(([k]) => !["train", "tune", "compare"].includes(k))
                                .map(([k, v]) => {
                                    const r = v as StepResponse;
                                    return (
                                        <div key={k} className="flex items-center gap-2 mb-1.5">
                                            <span className="w-1 h-1 rounded-full bg-emerald-400/60" />
                                            <span className="text-xs text-slate-400 truncate capitalize">{k.replace(/_/g, " ")}</span>
                                            <span className="text-xs text-slate-600 ml-auto truncate">{r.technique}</span>
                                        </div>
                                    );
                                })}
                        </div>
                    )}
                </aside>

                {/* CENTER */}
                <main className="flex-1 flex flex-col overflow-hidden">
                    <div className="px-5 pt-4 pb-3.5 border-b border-slate-800/60 flex-shrink-0">
                        <div className="flex items-center gap-2.5">
                            <span className="text-xs px-2 py-0.5 rounded-full
                bg-slate-800 text-slate-400 border border-slate-700 font-mono">
                                {activeStep} / {STEPS.length}
                            </span>
                            <h1 className="text-sm font-semibold text-white">
                                {STEPS[activeStep - 1]?.label}
                            </h1>
                        </div>
                    </div>

                    <div className="flex-1 overflow-y-auto p-5 space-y-4">

                        {activeStep === 1 && (
                            <ProfileView profile={profile} onNext={() => setActiveStep(2)} />
                        )}
                        {activeStep === 2 && (
                            <EDAView
                                profile={profile}
                                distributions={distributions}
                                correlation={correlation}
                                targetData={targetData}
                                loading={edaLoading}
                                onNext={() => setActiveStep(3)}
                            />
                        )}
                        {activeStep >= 3 && activeStep <= 8 && (() => {
                            const step = STEPS[activeStep - 1];
                            const techs = TECHNIQUES[step.key] || [];
                            const sel = selected[step.key] || techs[0]?.value;
                            return (
                                <StepView
                                    step={step}
                                    techniques={techs}
                                    selected={sel}
                                    onSelect={v => setSelected(s => ({ ...s, [step.key]: v }))}
                                    onApply={() => apply(step.key)}
                                    loading={loading === step.key}
                                    result={results[step.key] as StepResponse | undefined}
                                    profile={profile}
                                />
                            );
                        })()}
                        {activeStep === 9 && (
                            <TrainView
                                taskType={session.task_type}
                                selected={selected["train"] || "random_forest"}
                                onSelect={v => setSelected(s => ({ ...s, train: v }))}
                                onApply={() => apply("train")}
                                loading={loading === "train"}
                                result={trainResult}
                                regModels={regModels}
                                onNext={() => setActiveStep(10)}
                            />
                        )}
                        {activeStep === 10 && (
                            <TuneView
                                taskType={session.task_type}
                                selected={selected["train"] || "random_forest"}
                                onApply={() => apply("tune")}
                                loading={loading === "tune"}
                                result={tuneResult}
                                onNext={() => setActiveStep(11)}
                            />
                        )}
                        {activeStep === 11 && (
                            <ExplainView
                                trainResult={trainResult}
                                tuneResult={tuneResult}
                                taskType={session.task_type}
                                onNext={() => setActiveStep(12)}
                            />
                        )}
                        {activeStep === 12 && (
                            <CompareView
                                runs={runs}
                                taskType={session.task_type}
                            />
                        )}
                    </div>

                    <BottomChartBar
                        profile={profile}
                        distributions={distributions}
                        correlation={correlation}
                        targetData={targetData}
                        trainResult={trainResult || tuneResult}
                    />
                </main>

                {/* RIGHT PANEL */}
                <RightPanel
                    result={stepResult}
                    trainResult={trainResult || tuneResult}
                />
            </div>
        </div>
    );
}

// ─── PROFILE VIEW ─────────────────────────────────────────────────────────────

function ProfileView({ profile, onNext }: { profile: DatasetProfile; onNext: () => void }) {
    return (
        <div className="space-y-4 max-w-2xl">
            <div className="grid grid-cols-4 gap-3">
                {[
                    { label: "Rows", value: profile.shape.rows.toLocaleString() },
                    { label: "Columns", value: profile.shape.cols },
                    { label: "Missing cells", value: profile.missing_summary.total_missing_cells },
                    { label: "Duplicates", value: profile.duplicate_rows },
                ].map(item => (
                    <div key={item.label} className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-3.5">
                        <div className="text-xl font-bold text-white font-mono">{item.value}</div>
                        <div className="text-xs text-slate-500 mt-0.5">{item.label}</div>
                    </div>
                ))}
            </div>

            <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl overflow-hidden">
                <div className="px-4 py-2.5 border-b border-slate-700/50">
                    <span className="text-xs font-medium text-slate-400">Column overview</span>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                        <thead>
                            <tr className="text-slate-600 border-b border-slate-800/60 text-left">
                                {["Column", "Type", "Missing", "Unique", "Notes"].map(h => (
                                    <th key={h} className="px-4 py-2 font-medium">{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/40">
                            {profile.columns.map(col => (
                                <tr key={col.name} className="text-slate-300 hover:bg-slate-800/20">
                                    <td className="px-4 py-2 font-mono">
                                        {col.name}
                                        {col.is_target && (
                                            <span className="ml-1.5 px-1.5 py-0.5 rounded text-xs
                        bg-blue-500/10 text-blue-400 border border-blue-500/20">target</span>
                                        )}
                                    </td>
                                    <td className="px-4 py-2">
                                        <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${col.type === "numeric"
                                                ? "bg-purple-500/10 text-purple-400"
                                                : "bg-orange-500/10 text-orange-400"}`}>
                                            {col.type}
                                        </span>
                                    </td>
                                    <td className="px-4 py-2 font-mono">
                                        <span className={
                                            col.missing_pct > 20 ? "text-red-400"
                                                : col.missing_pct > 0 ? "text-amber-400"
                                                    : "text-emerald-400"}>
                                            {col.missing_pct}%
                                        </span>
                                    </td>
                                    <td className="px-4 py-2 text-slate-500 font-mono">{col.unique_count}</td>
                                    <td className="px-4 py-2 text-slate-600">
                                        {col.unique_count > 20 && col.type === "categorical" && "⚠ high cardinality"}
                                        {col.missing_pct > 50 && "⚠ >50% missing"}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            <button onClick={onNext}
                className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm
          rounded-lg font-medium transition-colors">
                Next: EDA →
            </button>
        </div>
    );
}

// ─── EDA VIEW ─────────────────────────────────────────────────────────────────

function EDAView({ profile, distributions, correlation, targetData, loading, onNext }: {
    profile: DatasetProfile;
    distributions: EDADistributions | null;
    correlation: EDACorrelation | null;
    targetData: EDATargetAnalysis | null;
    loading: boolean;
    onNext: () => void;
}) {
    const [activeChart, setActiveChart] = useState("distribution");
    const [selCol, setSelCol] = useState("");
    const [vizType, setVizType] = useState("bar");
    const col = selCol || Object.keys(distributions || {})[0] || "";

    if (loading) return (
        <div className="flex items-center gap-2 py-16 justify-center">
            <Loader2 className="w-5 h-5 animate-spin text-blue-400" />
            <span className="text-slate-400 text-sm">Loading EDA...</span>
        </div>
    );

    return (
        <div className="space-y-4 max-w-2xl">
            <div className="flex items-center gap-1.5 flex-wrap">
                {CHART_OPTIONS.map(ct => (
                    <button key={ct.value} onClick={() => setActiveChart(ct.value)}
                        className={`text-xs px-3 py-1.5 rounded-lg border font-medium transition-all ${activeChart === ct.value
                                ? "bg-blue-600/20 text-blue-300 border-blue-500/30"
                                : "text-slate-500 border-slate-700/50 hover:border-slate-600 hover:text-slate-300"}`}>
                        {ct.label}
                    </button>
                ))}
            </div>

            {activeChart === "distribution" && distributions && (
                <div className="space-y-3">
                    <div className="flex items-center gap-2">
                        <select value={col} onChange={e => setSelCol(e.target.value)}
                            className="text-xs bg-slate-800 border border-slate-700 rounded-lg
                px-3 py-1.5 text-slate-300 focus:outline-none focus:border-blue-500">
                            {Object.keys(distributions).map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                        {distributions[col]?.type === "numeric" && (
                            <select value={vizType} onChange={e => setVizType(e.target.value)}
                                className="text-xs bg-slate-800 border border-slate-700 rounded-lg
                  px-3 py-1.5 text-slate-300 focus:outline-none focus:border-blue-500">
                                {VIZ_TYPES.map(v => <option key={v.value} value={v.value}>{v.label}</option>)}
                            </select>
                        )}
                    </div>
                    {distributions[col] && (
                        <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
                            <p className="text-xs text-slate-500 mb-3 font-medium">{col} — {distributions[col].type}</p>
                            {distributions[col].type === "numeric" && distributions[col].histogram && (
                                <>
                                    <ChartRenderer
                                        vizType={vizType}
                                        data={distributions[col].histogram!.counts.map((c, i) => ({
                                            x: Number(distributions[col].histogram!.edges[i]).toFixed(1),
                                            v: c,
                                            name: Number(distributions[col].histogram!.edges[i]).toFixed(1),
                                        }))}
                                        height={160}
                                        color="#3b82f6"
                                    />
                                    {distributions[col].stats && (
                                        <div className="grid grid-cols-3 gap-2 mt-3">
                                            {Object.entries(distributions[col].stats!).map(([k, v]) => (
                                                <div key={k} className="bg-slate-900/50 rounded-lg p-2">
                                                    <div className="text-xs text-slate-600">{k}</div>
                                                    <div className="text-xs font-mono text-slate-300 mt-0.5">{Number(v).toFixed(3)}</div>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </>
                            )}
                            {distributions[col].type === "categorical" && distributions[col].bar && (
                                <ResponsiveContainer width="100%" height={160}>
                                    <BarChart data={distributions[col].bar!.labels.map((l, i) => ({
                                        label: l, count: distributions[col].bar!.counts[i]
                                    }))} layout="vertical">
                                        <XAxis type="number" tick={{ fill: "#475569", fontSize: 9 }} />
                                        <YAxis type="category" dataKey="label" tick={{ fill: "#475569", fontSize: 9 }} width={80} />
                                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                                        <Bar dataKey="count" fill="#3b82f6" radius={[0, 2, 2, 0]} opacity={0.8} />
                                    </BarChart>
                                </ResponsiveContainer>
                            )}
                        </div>
                    )}
                </div>
            )}

            {activeChart === "correlation" && correlation && (
                <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
                    <p className="text-xs text-slate-500 mb-3 font-medium">Correlation matrix</p>
                    <CorrelationHeatmap correlation={correlation} />
                </div>
            )}

            {activeChart === "target" && targetData && (
                <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
                    <p className="text-xs text-slate-500 mb-1 font-medium">
                        Target: <span className="text-white font-mono">{targetData.target}</span>
                    </p>
                    {targetData.is_imbalanced && (
                        <p className="text-xs text-amber-400 mb-3">
                            ⚠ Class imbalance — ratio {targetData.class_balance?.toFixed(2)}.
                            Consider SMOTE or class_weight="balanced".
                        </p>
                    )}
                    {targetData.distribution && (
                        <ResponsiveContainer width="100%" height={160}>
                            <BarChart data={targetData.distribution.labels.map((l, i) => ({
                                label: String(l), count: targetData.distribution!.counts[i]
                            }))}>
                                <XAxis dataKey="label" tick={{ fill: "#475569", fontSize: 11 }} />
                                <YAxis tick={{ fill: "#475569", fontSize: 11 }} />
                                <Tooltip contentStyle={TOOLTIP_STYLE} />
                                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                                    {targetData.distribution.labels.map((_, i) => (
                                        <Cell key={i} fill={["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b"][i % 4]} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    )}
                    {targetData.histogram && (
                        <ResponsiveContainer width="100%" height={160}>
                            <BarChart data={targetData.histogram.counts.map((c, i) => ({
                                x: Number(targetData.histogram!.edges[i]).toFixed(1), v: c
                            }))}>
                                <XAxis dataKey="x" tick={{ fill: "#475569", fontSize: 9 }} interval={4} />
                                <YAxis tick={{ fill: "#475569", fontSize: 9 }} />
                                <Tooltip contentStyle={TOOLTIP_STYLE} />
                                <Bar dataKey="v" fill="#f59e0b" radius={[2, 2, 0, 0]} opacity={0.8} />
                            </BarChart>
                        </ResponsiveContainer>
                    )}
                </div>
            )}

            {activeChart === "missing" && (
                <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
                    <p className="text-xs text-slate-500 mb-3 font-medium">Missing values per column</p>
                    <div className="space-y-2.5">
                        {profile.columns.map(col => (
                            <div key={col.name} className="flex items-center gap-3">
                                <span className="text-xs font-mono text-slate-500 w-24 truncate">{col.name}</span>
                                <div className="flex-1 bg-slate-700/50 rounded-full h-1.5">
                                    <div className="h-1.5 rounded-full transition-all" style={{
                                        width: `${Math.max(col.missing_pct, col.missing_pct > 0 ? 2 : 0)}%`,
                                        background: col.missing_pct > 20 ? "#ef4444"
                                            : col.missing_pct > 0 ? "#f59e0b" : "#10b981",
                                    }} />
                                </div>
                                <span className={`text-xs font-mono w-9 text-right ${col.missing_pct > 20 ? "text-red-400"
                                        : col.missing_pct > 0 ? "text-amber-400"
                                            : "text-slate-600"}`}>
                                    {col.missing_pct}%
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <button onClick={onNext}
                className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm
          rounded-lg font-medium transition-colors">
                Next: Missing values →
            </button>
        </div>
    );
}

// ─── CHART RENDERER ───────────────────────────────────────────────────────────

function ChartRenderer({ vizType, data, height, color }: {
    vizType: string;
    data: any[];
    height: number;
    color: string;
}) {
    return (
        <ResponsiveContainer width="100%" height={height}>
            {vizType === "bar" ? (
                <BarChart data={data}>
                    <XAxis dataKey="x" tick={{ fill: "#475569", fontSize: 9 }} interval={4} />
                    <YAxis tick={{ fill: "#475569", fontSize: 9 }} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Bar dataKey="v" fill={color} radius={[2, 2, 0, 0]} opacity={0.8} />
                </BarChart>
            ) : vizType === "area" ? (
                <AreaChart data={data}>
                    <defs>
                        <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                            <stop offset="95%" stopColor={color} stopOpacity={0.02} />
                        </linearGradient>
                    </defs>
                    <XAxis dataKey="x" tick={{ fill: "#475569", fontSize: 9 }} interval={4} />
                    <YAxis tick={{ fill: "#475569", fontSize: 9 }} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Area type="monotone" dataKey="v" stroke={color} fill="url(#areaGrad)" strokeWidth={2} />
                </AreaChart>
            ) : vizType === "line" ? (
                <LineChart data={data}>
                    <XAxis dataKey="x" tick={{ fill: "#475569", fontSize: 9 }} interval={4} />
                    <YAxis tick={{ fill: "#475569", fontSize: 9 }} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Line type="monotone" dataKey="v" stroke={color} strokeWidth={2} dot={false} />
                </LineChart>
            ) : vizType === "scatter" ? (
                <ScatterChart>
                    <XAxis dataKey="x" type="category" tick={{ fill: "#475569", fontSize: 9 }} interval={4} />
                    <YAxis dataKey="v" tick={{ fill: "#475569", fontSize: 9 }} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Scatter data={data} fill={color} opacity={0.7} />
                </ScatterChart>
            ) : vizType === "radar" ? (
                <RadarChart data={data.slice(0, 12)}>
                    <PolarGrid stroke="#1e293b" />
                    <PolarAngleAxis dataKey="x" tick={{ fill: "#475569", fontSize: 9 }} />
                    <Radar dataKey="v" stroke={color} fill={color} fillOpacity={0.2} strokeWidth={2} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                </RadarChart>
            ) : (
                <BarChart data={data}>
                    <XAxis dataKey="x" tick={{ fill: "#475569", fontSize: 9 }} interval={4} />
                    <YAxis tick={{ fill: "#475569", fontSize: 9 }} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Bar dataKey="v" fill={color} radius={[2, 2, 0, 0]} opacity={0.8} />
                </BarChart>
            )}
        </ResponsiveContainer>
    );
}

// ─── CORRELATION HEATMAP ──────────────────────────────────────────────────────

function CorrelationHeatmap({ correlation }: { correlation: EDACorrelation }) {
    const { columns, matrix } = correlation;
    const n = columns.length;
    const size = Math.min(Math.floor(420 / n), 38);

    function cellColor(v: number) {
        const abs = Math.abs(v);
        if (v > 0.7) return `rgba(59,130,246,${0.3 + abs * 0.7})`;
        if (v > 0.3) return `rgba(99,102,241,${0.2 + abs * 0.6})`;
        if (v > -0.3) return `rgba(51,65,85,0.4)`;
        if (v > -0.7) return `rgba(239,68,68,${0.2 + abs * 0.5})`;
        return `rgba(220,38,38,${0.3 + abs * 0.6})`;
    }

    return (
        <div className="overflow-x-auto">
            <div className="flex ml-16">
                {columns.map(c => (
                    <div key={c} style={{ width: size, fontSize: 8, color: "#475569", textAlign: "center" }}
                        className="overflow-hidden whitespace-nowrap pb-1">
                        {c.substring(0, 5)}
                    </div>
                ))}
            </div>
            {matrix.map((row, i) => (
                <div key={i} className="flex items-center">
                    <div style={{ width: 64, fontSize: 9, color: "#475569" }}
                        className="truncate pr-2 text-right flex-shrink-0">
                        {columns[i]}
                    </div>
                    {row.map((val, j) => (
                        <div key={j}
                            style={{ width: size, height: size, background: cellColor(val) }}
                            title={`${columns[i]} × ${columns[j]}: ${val.toFixed(2)}`}
                            className="border border-slate-900/50"
                        />
                    ))}
                </div>
            ))}
            <div className="flex items-center gap-3 mt-3 ml-16">
                {[
                    { color: "rgba(59,130,246,0.9)", label: "positive" },
                    { color: "rgba(51,65,85,0.4)", label: "neutral" },
                    { color: "rgba(220,38,38,0.8)", label: "negative" },
                ].map(({ color, label }) => (
                    <div key={label} className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded" style={{ background: color }} />
                        <span className="text-xs text-slate-500">{label}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

// ─── STEP VIEW ────────────────────────────────────────────────────────────────

function StepView({ step, techniques, selected, onSelect, onApply, loading, result, profile }: {
    step: { id: number; key: string; label: string };
    techniques: { value: string; label: string; desc: string }[];
    selected: string;
    onSelect: (v: string) => void;
    onApply: () => void;
    loading: boolean;
    result?: StepResponse;
    profile: DatasetProfile;
}) {
    return (
        <div className="space-y-4 max-w-2xl">
            <div className="grid grid-cols-2 gap-2">
                {techniques.map(t => (
                    <button key={t.value} onClick={() => onSelect(t.value)}
                        className={`text-left px-4 py-3 rounded-xl border transition-all ${selected === t.value
                                ? "bg-blue-600/15 border-blue-500/40"
                                : "bg-slate-800/40 border-slate-700/50 hover:border-slate-600 hover:bg-slate-800/60"}`}>
                        <div className={`text-xs font-semibold ${selected === t.value ? "text-blue-300" : "text-slate-200"}`}>
                            {t.label}
                        </div>
                        <div className="text-xs text-slate-500 mt-0.5">{t.desc}</div>
                    </button>
                ))}
            </div>

            {step.key === "missing" && (
                <BeforeAfterMissing profile={profile} technique={selected} />
            )}

            <button
                onClick={onApply}
                disabled={!!loading || !!result}
                className={`px-5 py-2.5 text-white text-sm rounded-lg font-medium
          transition-colors flex items-center gap-2
          ${result
                        ? "bg-emerald-700 cursor-not-allowed opacity-80"
                        : "bg-blue-600 hover:bg-blue-500 disabled:opacity-40"}`}>
                {loading
                    ? <><Loader2 className="w-4 h-4 animate-spin" />Applying...</>
                    : result
                        ? "✓ Applied"
                        : `Apply ${selected} →`}
            </button>
        </div>
    );
}

// ─── BEFORE / AFTER MISSING ───────────────────────────────────────────────────

function BeforeAfterMissing({ profile, technique }: { profile: DatasetProfile; technique: string }) {
    const missingCols = profile.columns.filter(c => c.missing_count > 0).slice(0, 2);
    if (!missingCols.length) return null;

    function afterLabel(col: typeof missingCols[0]) {
        if (technique === "drop_rows") return "dropped";
        if (technique === "drop_cols") return "removed";
        if (technique === "mean") return col.stats?.mean ? Number(col.stats.mean).toFixed(2) : "mean";
        if (technique === "median") return col.stats?.median ? Number(col.stats.median).toFixed(2) : "median";
        if (technique === "constant") return "0.0";
        if (technique === "knn") return "KNN est.";
        if (technique === "mice") return "MICE est.";
        return "filled";
    }

    return (
        <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4">
            <p className="text-xs text-slate-500 font-medium mb-3">Preview — what changes</p>
            <div className="space-y-4">
                {missingCols.map(col => (
                    <div key={col.name}>
                        <p className="text-xs font-mono text-slate-400 mb-2">
                            {col.name} <span className="text-slate-600 ml-1">{col.missing_count} nulls</span>
                        </p>
                        <div className="grid grid-cols-2 gap-3">
                            {[
                                { title: "Before", vals: ["29.0", "NULL", "38.0", "NULL"], isNull: [false, true, false, true], isAfter: false },
                                { title: "After", vals: ["29.0", afterLabel(col), "38.0", afterLabel(col)], isNull: [false, true, false, true], isAfter: true },
                            ].map(({ title, vals, isNull, isAfter }) => (
                                <div key={title}>
                                    <p className="text-xs text-slate-600 mb-1.5">{title}</p>
                                    <div className="space-y-1">
                                        {vals.map((v, i) => (
                                            <div key={i} className={`px-2.5 py-1.5 rounded-lg text-xs font-mono text-center ${isNull[i]
                                                    ? isAfter
                                                        ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                                                        : "bg-red-500/10 text-red-400 border border-red-500/20"
                                                    : "bg-slate-700/50 text-slate-300"}`}>
                                                {v}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

// ─── TRAIN VIEW ───────────────────────────────────────────────────────────────

function TrainView({ taskType, selected, onSelect, onApply, loading, result, regModels, onNext }: {
    taskType: string; selected: string;
    onSelect: (v: string) => void; onApply: () => void;
    loading: boolean; result?: TrainResponse;
    regModels: string[]; onNext: () => void;
}) {
    const models = TECHNIQUES.train.filter(m =>
        taskType === "regression"
            ? regModels.includes(m.value) || ["random_forest", "xgboost", "lightgbm", "knn"].includes(m.value)
            : !regModels.includes(m.value)
    );

    return (
        <div className="space-y-4 max-w-2xl">
            <div className="grid grid-cols-2 gap-2">
                {models.map(m => (
                    <button key={m.value} onClick={() => onSelect(m.value)}
                        className={`text-left px-4 py-3 rounded-xl border transition-all ${selected === m.value
                                ? "bg-blue-600/15 border-blue-500/40"
                                : "bg-slate-800/40 border-slate-700/50 hover:border-slate-600 hover:bg-slate-800/60"}`}>
                        <div className={`text-xs font-semibold ${selected === m.value ? "text-blue-300" : "text-slate-200"}`}>
                            {m.label}
                        </div>
                        <div className="text-xs text-slate-500 mt-0.5">{m.desc}</div>
                    </button>
                ))}
            </div>

            <button onClick={onApply} disabled={!!loading}
                className="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500
          disabled:opacity-40 text-white text-sm rounded-lg font-medium
          transition-colors flex items-center gap-2">
                {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Training...</> : `Train ${selected} →`}
            </button>

            {result && (
                <div className="space-y-3">
                    <div className="grid grid-cols-3 gap-2">
                        {Object.entries(result.metrics)
                            .filter(([k]) => k !== "confusion_matrix")
                            .map(([k, v]) => (
                                <div key={k} className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-3">
                                    <div className="text-xs text-slate-500">{k.replace(/_/g, " ")}</div>
                                    <div className="text-sm font-bold font-mono text-white mt-0.5">
                                        {typeof v === "number" ? v.toFixed(4) : String(v)}
                                    </div>
                                </div>
                            ))}
                    </div>
                    <div className="flex gap-4 text-xs text-slate-500">
                        <span>Train <span className="text-slate-300 font-mono">{result.train_size}</span></span>
                        <span>Test <span className="text-slate-300 font-mono">{result.test_size}</span></span>
                        <span>Features <span className="text-slate-300 font-mono">{result.n_features}</span></span>
                    </div>
                    <button onClick={onNext}
                        className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm
              rounded-lg font-medium transition-colors">
                        Next: Hyperparameter tuning →
                    </button>
                </div>
            )}
        </div>
    );
}

// ─── TUNE VIEW ────────────────────────────────────────────────────────────────

function TuneView({ taskType, selected, onApply, loading, result, onNext }: {
    taskType: string; selected: string;
    onApply: () => void; loading: boolean;
    result?: TrainResponse; onNext: () => void;
}) {
    const trials = result?.metrics?.trials as any[] | undefined;

    return (
        <div className="space-y-4 max-w-2xl">
            <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4">
                <p className="text-xs text-slate-400 font-medium mb-1">Optuna hyperparameter search</p>
                <p className="text-xs text-slate-500 leading-relaxed">
                    Runs 20 trials using Tree-structured Parzen Estimator (TPE) to find the best
                    hyperparameters for <span className="text-slate-300 font-mono">{selected}</span>.
                    Each trial uses 3-fold cross-validation. Best parameters are auto-applied.
                </p>
            </div>

            <div className="grid grid-cols-2 gap-2">
                {[
                    { label: "Sampler", value: "TPE (Bayesian)" },
                    { label: "Trials", value: "20" },
                    { label: "CV folds", value: "3" },
                    { label: "Timeout", value: "60 seconds" },
                ].map(item => (
                    <div key={item.label} className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-3">
                        <div className="text-xs text-slate-500">{item.label}</div>
                        <div className="text-sm font-mono text-white mt-0.5">{item.value}</div>
                    </div>
                ))}
            </div>

            <button onClick={onApply} disabled={!!loading}
                className="px-5 py-2.5 bg-violet-600 hover:bg-violet-500
          disabled:opacity-40 text-white text-sm rounded-lg font-medium
          transition-colors flex items-center gap-2">
                {loading
                    ? <><Loader2 className="w-4 h-4 animate-spin" />Searching ({selected})...</>
                    : `Run Optuna search on ${selected} →`}
            </button>

            {result && (
                <div className="space-y-3">
                    <div className="bg-violet-500/10 border border-violet-500/20 rounded-xl p-4">
                        <p className="text-xs text-violet-300 font-medium mb-2">Best result</p>
                        <div className="grid grid-cols-2 gap-2">
                            <div className="bg-slate-900/50 rounded-lg p-2.5">
                                <div className="text-xs text-slate-500">Best CV score</div>
                                <div className="text-lg font-bold font-mono text-white">
                                    {Number(result.metrics.best_trial_score).toFixed(4)}
                                </div>
                            </div>
                            <div className="bg-slate-900/50 rounded-lg p-2.5">
                                <div className="text-xs text-slate-500">Trials run</div>
                                <div className="text-lg font-bold font-mono text-white">
                                    {result.metrics.n_trials}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Trial history chart */}
                    {trials && trials.length > 0 && (
                        <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
                            <p className="text-xs text-slate-500 mb-3 font-medium">Trial scores over time</p>
                            <ResponsiveContainer width="100%" height={140}>
                                <LineChart data={trials}>
                                    <XAxis dataKey="number" tick={{ fill: "#475569", fontSize: 9 }} label={{ value: "Trial", fill: "#475569", fontSize: 9 }} />
                                    <YAxis tick={{ fill: "#475569", fontSize: 9 }} domain={["auto", "auto"]} />
                                    <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: any) => [Number(v).toFixed(4), "Score"]} />
                                    <Line type="monotone" dataKey="value" stroke="#8b5cf6" strokeWidth={2} dot={{ fill: "#8b5cf6", r: 3 }} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    )}

                    {/* Best params */}
                    <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
                        <p className="text-xs text-slate-500 mb-2 font-medium">Best parameters found</p>
                        <div className="space-y-1.5">
                            {Object.entries(result.metrics)
                                .filter(([k]) => !["best_trial_score", "n_trials", "trials", "confusion_matrix",
                                    "accuracy", "f1_weighted", "f1_macro", "roc_auc", "cv_mean", "cv_std", "rmse", "mae", "r2"].includes(k))
                                .map(([k, v]) => (
                                    <div key={k} className="flex justify-between">
                                        <span className="text-xs text-slate-500 font-mono">{k}</span>
                                        <span className="text-xs text-slate-300 font-mono">{String(v)}</span>
                                    </div>
                                ))}
                        </div>
                    </div>

                    <div className="grid grid-cols-3 gap-2">
                        {Object.entries(result.metrics)
                            .filter(([k]) => ["accuracy", "f1_weighted", "r2", "rmse", "cv_mean"].includes(k))
                            .map(([k, v]) => (
                                <div key={k} className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-3">
                                    <div className="text-xs text-slate-500">{k.replace(/_/g, " ")}</div>
                                    <div className="text-sm font-bold font-mono text-white mt-0.5">
                                        {typeof v === "number" ? v.toFixed(4) : String(v)}
                                    </div>
                                </div>
                            ))}
                    </div>

                    <button onClick={onNext}
                        className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm
              rounded-lg font-medium transition-colors">
                        Next: Explainability →
                    </button>
                </div>
            )}
        </div>
    );
}

// ─── EXPLAIN VIEW ─────────────────────────────────────────────────────────────

function ExplainView({ trainResult, tuneResult, taskType, onNext }: {
    trainResult?: TrainResponse;
    tuneResult?: TrainResponse;
    taskType: string;
    onNext: () => void;
}) {
    const result = tuneResult || trainResult;

    if (!result) return (
        <div className="max-w-2xl">
            <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-6 text-center">
                <p className="text-slate-400 text-sm">Complete Step 9 (Train model) first to see explainability.</p>
            </div>
        </div>
    );

    const fi = Object.entries(result.feature_importance);
    const maxFi = fi.length > 0 ? fi[0][1] : 1;

    const radarData = fi.slice(0, 8).map(([name, val]) => ({
        feature: name.length > 10 ? name.substring(0, 10) + "…" : name,
        importance: Number(val),
        fullName: name,
    }));

    return (
        <div className="space-y-4 max-w-2xl">

            {/* Model used */}
            <div className="flex items-center gap-3 text-xs text-slate-500">
                <span>Model: <span className="text-slate-300 font-mono">{result.model}</span></span>
                <span>Features: <span className="text-slate-300 font-mono">{result.n_features}</span></span>
                {tuneResult && <span className="text-violet-400">✓ Optuna-tuned</span>}
            </div>

            {/* Feature importance bar chart */}
            {fi.length > 0 && (
                <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
                    <p className="text-xs text-slate-500 mb-3 font-medium">Feature importance — bar chart</p>
                    <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={fi.slice(0, 12).map(([k, v]) => ({ name: k, value: Number(v) }))}
                            layout="vertical" margin={{ left: 10, right: 20 }}>
                            <XAxis type="number" tick={{ fill: "#475569", fontSize: 9 }} />
                            <YAxis type="category" dataKey="name" tick={{ fill: "#94a3b8", fontSize: 10 }} width={110} />
                            <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: any) => [Number(v).toFixed(4), "Importance"]} />
                            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                {fi.slice(0, 12).map((_, i) => (
                                    <Cell key={i} fill={i === 0 ? "#3b82f6" : i < 3 ? "#6366f1" : "#334155"} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            )}

            {/* Radar chart */}
            {radarData.length >= 3 && (
                <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
                    <p className="text-xs text-slate-500 mb-3 font-medium">Feature importance — radar view (top 8)</p>
                    <ResponsiveContainer width="100%" height={220}>
                        <RadarChart data={radarData}>
                            <PolarGrid stroke="#1e293b" />
                            <PolarAngleAxis dataKey="feature" tick={{ fill: "#64748b", fontSize: 10 }} />
                            <Radar dataKey="importance" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.15} strokeWidth={2} />
                            <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: any) => [Number(v).toFixed(4), "Importance"]} />
                        </RadarChart>
                    </ResponsiveContainer>
                </div>
            )}

            {/* Interpretation */}
            <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4">
                <p className="text-xs text-slate-400 font-medium mb-2 flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-blue-400" /> Interpretation
                </p>
                <div className="space-y-2">
                    {fi.slice(0, 3).map(([name, val], i) => (
                        <p key={name} className="text-xs text-slate-400 leading-relaxed">
                            {i === 0 && `The most important feature is `}
                            {i === 1 && `Second is `}
                            {i === 2 && `Third is `}
                            <span className="text-white font-mono">{name}</span>
                            {` (importance: ${Number(val).toFixed(4)})`}
                            {i === 0 && ` — this feature drives the model's predictions the most.`}
                            {i === 1 && `, contributing significantly to predictions.`}
                            {i === 2 && `.`}
                        </p>
                    ))}
                    <p className="text-xs text-slate-500 mt-2">
                        Note: These are permutation-based feature importances from{" "}
                        <span className="font-mono text-slate-400">{result.model}</span>.
                        For SHAP-level explanations, add the SHAP library and call
                        <span className="font-mono text-slate-400"> shap.TreeExplainer</span> on your fitted model.
                    </p>
                </div>
            </div>

            <button onClick={onNext}
                className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm
          rounded-lg font-medium transition-colors">
                Next: Experiment comparison →
            </button>
        </div>
    );
}

// ─── COMPARE VIEW ─────────────────────────────────────────────────────────────

function CompareView({ runs, taskType }: { runs: RunRecord[]; taskType: string }) {
    const [runA, setRunA] = useState<string>("");
    const [runB, setRunB] = useState<string>("");

    const selectedA = runs.find(r => r.id === runA) || runs[0];
    const selectedB = runs.find(r => r.id === runB) || runs[1];

    const metricKeys = taskType === "classification"
        ? ["accuracy", "f1_weighted", "f1_macro", "roc_auc", "cv_mean"]
        : ["rmse", "mae", "r2", "cv_mean"];

    if (runs.length < 1) return (
        <div className="max-w-2xl">
            <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-6 text-center">
                <p className="text-slate-400 text-sm mb-1">No runs yet.</p>
                <p className="text-xs text-slate-600">Complete Steps 9 and/or 10 to generate runs to compare.</p>
            </div>
        </div>
    );

    return (
        <div className="space-y-4 max-w-2xl">
            <p className="text-xs text-slate-500">
                Compare any two training runs. Each run captures the full pipeline state and model metrics.
            </p>

            {/* Run selectors */}
            <div className="grid grid-cols-2 gap-3">
                {[
                    { label: "Run A", val: runA, setter: setRunA, color: "blue" },
                    { label: "Run B", val: runB, setter: setRunB, color: "violet" },
                ].map(({ label, val, setter, color }) => (
                    <div key={label}>
                        <p className={`text-xs font-medium mb-1.5 text-${color}-400`}>{label}</p>
                        <select value={val} onChange={e => setter(e.target.value)}
                            className="w-full text-xs bg-slate-800 border border-slate-700 rounded-lg
                px-3 py-2 text-slate-300 focus:outline-none focus:border-blue-500">
                            <option value="">Latest run</option>
                            {runs.map(r => (
                                <option key={r.id} value={r.id}>
                                    {r.model_name} — {new Date(r.created_at).toLocaleTimeString()}
                                </option>
                            ))}
                        </select>
                    </div>
                ))}
            </div>

            {selectedA && selectedB && (
                <>
                    {/* Metric comparison table */}
                    <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl overflow-hidden">
                        <div className="grid grid-cols-3 px-4 py-2.5 border-b border-slate-700/50 text-xs font-medium">
                            <span className="text-slate-500">Metric</span>
                            <span className="text-blue-400">{selectedA.model_name}</span>
                            <span className="text-violet-400">{selectedB.model_name}</span>
                        </div>
                        <div className="divide-y divide-slate-800/40">
                            {metricKeys.map(k => {
                                const vA = selectedA.metrics[k] as number | undefined;
                                const vB = selectedB.metrics[k] as number | undefined;
                                if (vA === undefined && vB === undefined) return null;
                                const aWins = vA !== undefined && vB !== undefined
                                    ? (k === "rmse" || k === "mae" ? vA < vB : vA > vB)
                                    : false;
                                const bWins = vA !== undefined && vB !== undefined
                                    ? (k === "rmse" || k === "mae" ? vB < vA : vB > vA)
                                    : false;
                                return (
                                    <div key={k} className="grid grid-cols-3 px-4 py-2.5 text-xs">
                                        <span className="text-slate-500">{k.replace(/_/g, " ")}</span>
                                        <span className={`font-mono font-medium ${aWins ? "text-emerald-400" : "text-slate-300"}`}>
                                            {vA !== undefined ? vA.toFixed(4) : "—"}
                                            {aWins && " ✓"}
                                        </span>
                                        <span className={`font-mono font-medium ${bWins ? "text-emerald-400" : "text-slate-300"}`}>
                                            {vB !== undefined ? vB.toFixed(4) : "—"}
                                            {bWins && " ✓"}
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Metric bar comparison */}
                    {metricKeys.filter(k => selectedA.metrics[k] !== undefined).length > 0 && (
                        <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
                            <p className="text-xs text-slate-500 mb-3 font-medium">Visual comparison</p>
                            <ResponsiveContainer width="100%" height={180}>
                                <BarChart data={metricKeys
                                    .filter(k => selectedA.metrics[k] !== undefined || selectedB?.metrics[k] !== undefined)
                                    .map(k => ({
                                        metric: k.replace(/_/g, " "),
                                        [selectedA.model_name]: Number(selectedA.metrics[k] || 0),
                                        [selectedB?.model_name || "B"]: Number(selectedB?.metrics[k] || 0),
                                    }))}>
                                    <XAxis dataKey="metric" tick={{ fill: "#475569", fontSize: 9 }} />
                                    <YAxis tick={{ fill: "#475569", fontSize: 9 }} domain={[0, 1]} />
                                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                                    <Bar dataKey={selectedA.model_name} fill="#3b82f6" radius={[2, 2, 0, 0]} opacity={0.8} />
                                    <Bar dataKey={selectedB?.model_name || "B"} fill="#8b5cf6" radius={[2, 2, 0, 0]} opacity={0.8} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}

                    {/* Params comparison */}
                    <div className="grid grid-cols-2 gap-3">
                        {[selectedA, selectedB].map((run, idx) => (
                            <div key={idx} className={`bg-slate-800/40 border rounded-xl p-3 ${idx === 0 ? "border-blue-500/20" : "border-violet-500/20"}`}>
                                <p className={`text-xs font-medium mb-2 ${idx === 0 ? "text-blue-400" : "text-violet-400"}`}>
                                    {run.model_name} params
                                </p>
                                {Object.keys(run.params).length === 0 ? (
                                    <p className="text-xs text-slate-600">Default params</p>
                                ) : (
                                    Object.entries(run.params).map(([k, v]) => (
                                        <div key={k} className="flex justify-between text-xs mb-1">
                                            <span className="text-slate-500 font-mono">{k}</span>
                                            <span className="text-slate-300 font-mono">{String(v)}</span>
                                        </div>
                                    ))
                                )}
                            </div>
                        ))}
                    </div>
                </>
            )}

            {runs.length === 1 && (
                <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-3">
                    <p className="text-xs text-amber-400">
                        Only one run found. Go back to Step 9 and train with a different model to compare.
                    </p>
                </div>
            )}
        </div>
    );
}

// ─── BOTTOM CHART BAR ─────────────────────────────────────────────────────────

function BottomChartBar({ profile, distributions, correlation, targetData, trainResult }: {
    profile: DatasetProfile;
    distributions: EDADistributions | null;
    correlation: EDACorrelation | null;
    targetData: EDATargetAnalysis | null;
    trainResult?: TrainResponse;
}) {
    const [open, setOpen] = useState(false);
    const [chartType, setChartType] = useState("distribution");
    const [vizType, setVizType] = useState("bar");
    const [selCol, setSelCol] = useState("");
    const col = selCol || Object.keys(distributions || {})[0] || "";

    const options = [
        { value: "distribution", label: "Distribution" },
        { value: "missing", label: "Missing values" },
        { value: "correlation", label: "Correlations" },
        { value: "target", label: "Target" },
        ...(trainResult ? [{ value: "importance", label: "Feature importance" }] : []),
    ];

    const distData = distributions?.[col]?.histogram
        ? distributions[col].histogram!.counts.map((c, i) => ({
            x: Number(distributions![col].histogram!.edges[i]).toFixed(1), v: c, name: String(i)
        }))
        : [];

    return (
        <div className="border-t border-slate-800/60 bg-slate-950 flex-shrink-0">
            <div className="flex items-center gap-3 px-4 py-2">
                <button onClick={() => setOpen(o => !o)}
                    className="flex items-center gap-1.5 text-xs text-slate-500
            hover:text-slate-300 transition-colors font-medium uppercase tracking-wider">
                    <TrendingUp className="w-3.5 h-3.5" />
                    Charts
                    <span className="text-slate-700">{open ? "▲" : "▼"}</span>
                </button>

                {open && (
                    <>
                        <select value={chartType} onChange={e => setChartType(e.target.value)}
                            className="text-xs bg-slate-900 border border-slate-800 rounded-lg
                px-2.5 py-1 text-slate-400 focus:outline-none focus:border-slate-600">
                            {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                        </select>

                        {chartType === "distribution" && distributions && (
                            <>
                                <select value={col} onChange={e => setSelCol(e.target.value)}
                                    className="text-xs bg-slate-900 border border-slate-800 rounded-lg
                    px-2.5 py-1 text-slate-400 focus:outline-none focus:border-slate-600">
                                    {Object.keys(distributions).map(c => <option key={c} value={c}>{c}</option>)}
                                </select>
                                {distributions[col]?.type === "numeric" && (
                                    <select value={vizType} onChange={e => setVizType(e.target.value)}
                                        className="text-xs bg-slate-900 border border-slate-800 rounded-lg
                      px-2.5 py-1 text-slate-400 focus:outline-none focus:border-slate-600">
                                        {VIZ_TYPES.map(v => <option key={v.value} value={v.value}>{v.label}</option>)}
                                    </select>
                                )}
                            </>
                        )}
                    </>
                )}
            </div>

            {open && (
                <div className="px-4 pb-4">
                    <div className="bg-slate-900 border border-slate-800/60 rounded-xl p-3" style={{ height: 180 }}>

                        {chartType === "distribution" && distributions && distributions[col] && (
                            distributions[col].type === "numeric" ? (
                                <ChartRenderer vizType={vizType} data={distData} height={156} color="#3b82f6" />
                            ) : (
                                <ResponsiveContainer width="100%" height={156}>
                                    <BarChart data={distributions[col].bar!.labels.map((l, i) => ({
                                        label: l, count: distributions[col].bar!.counts[i]
                                    }))} layout="vertical">
                                        <XAxis type="number" tick={{ fill: "#334155", fontSize: 9 }} />
                                        <YAxis type="category" dataKey="label" tick={{ fill: "#334155", fontSize: 9 }} width={70} />
                                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                                        <Bar dataKey="count" fill="#3b82f6" radius={[0, 2, 2, 0]} opacity={0.8} />
                                    </BarChart>
                                </ResponsiveContainer>
                            )
                        )}

                        {chartType === "missing" && (
                            <div className="overflow-y-auto h-full space-y-2 pr-1">
                                {profile.columns.map(c => (
                                    <div key={c.name} className="flex items-center gap-2">
                                        <span className="text-xs font-mono text-slate-600 w-20 truncate">{c.name}</span>
                                        <div className="flex-1 bg-slate-800 rounded-full h-1.5">
                                            <div className="h-1.5 rounded-full" style={{
                                                width: `${Math.max(c.missing_pct, c.missing_pct > 0 ? 1 : 0)}%`,
                                                background: c.missing_pct > 20 ? "#ef4444"
                                                    : c.missing_pct > 0 ? "#f59e0b" : "#10b981",
                                            }} />
                                        </div>
                                        <span className="text-xs text-slate-600 font-mono w-8 text-right">
                                            {c.missing_pct}%
                                        </span>
                                    </div>
                                ))}
                            </div>
                        )}

                        {chartType === "correlation" && correlation && (
                            <div className="overflow-auto h-full">
                                <CorrelationHeatmap correlation={correlation} />
                            </div>
                        )}

                        {chartType === "target" && targetData?.distribution && (
                            <ResponsiveContainer width="100%" height={156}>
                                <BarChart data={targetData.distribution.labels.map((l, i) => ({
                                    label: String(l), count: targetData.distribution!.counts[i]
                                }))}>
                                    <XAxis dataKey="label" tick={{ fill: "#334155", fontSize: 10 }} />
                                    <YAxis tick={{ fill: "#334155", fontSize: 10 }} />
                                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                                    <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        )}

                        {chartType === "importance" && trainResult && (
                            <ResponsiveContainer width="100%" height={156}>
                                <BarChart layout="vertical"
                                    data={Object.entries(trainResult.feature_importance).slice(0, 8)
                                        .map(([k, v]) => ({ name: k, value: Number(v) }))}>
                                    <XAxis type="number" tick={{ fill: "#334155", fontSize: 9 }} />
                                    <YAxis type="category" dataKey="name"
                                        tick={{ fill: "#334155", fontSize: 9 }} width={90} />
                                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                                    <Bar dataKey="value" fill="#10b981" radius={[0, 4, 4, 0]}>
                                        {Object.keys(trainResult.feature_importance).slice(0, 8).map((_, i) => (
                                            <Cell key={i} fill={i === 0 ? "#10b981" : "#1e293b"} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

// ─── RIGHT PANEL ──────────────────────────────────────────────────────────────

function RightPanel({ result, trainResult }: {
    result?: StepResponse;
    trainResult?: TrainResponse;
}) {
    const fi = trainResult ? Object.entries(trainResult.feature_importance).slice(0, 8) : [];
    const maxFi = fi.length > 0 ? Number(fi[0][1]) : 1;

    return (
        <aside className="w-60 bg-slate-900 border-l border-slate-800
      flex flex-col overflow-y-auto flex-shrink-0">

            <div className="border-b border-slate-800">
                <div className="flex items-center gap-2 px-3 py-2.5">
                    <Sparkles className="w-3.5 h-3.5 text-blue-400" />
                    <span className="text-xs font-medium text-slate-300 uppercase tracking-wider">AI explanation</span>
                </div>
                <div className="px-3 pb-3.5">
                    {result?.ai_explanation ? (
                        <p className="text-xs text-slate-400 leading-relaxed">{result.ai_explanation}</p>
                    ) : (
                        <p className="text-xs text-slate-700 italic leading-relaxed">
                            Apply a technique to see the AI explain what changed and why.
                        </p>
                    )}
                </div>
            </div>

            {result?.warnings && result.warnings.length > 0 && (
                <div className="border-b border-slate-800 px-3 py-2.5 space-y-2">
                    {result.warnings.map((w, i) => (
                        <div key={i} className="flex gap-1.5 text-xs text-amber-400/80">
                            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                            <span>{w}</span>
                        </div>
                    ))}
                </div>
            )}

            {result?.stats && (
                <div className="border-b border-slate-800 px-3 py-2.5">
                    <p className="text-xs text-slate-600 uppercase tracking-wider mb-2 font-medium">Stats</p>
                    <div className="space-y-1.5">
                        {Object.entries(result.stats)
                            .filter(([k]) => STATS_KEYS.includes(k))
                            .map(([k, v]) => (
                                <div key={k} className="flex justify-between items-center">
                                    <span className="text-xs text-slate-600">{k.replace(/_/g, " ")}</span>
                                    <span className="text-xs font-mono text-slate-300">{String(v)}</span>
                                </div>
                            ))}
                    </div>
                </div>
            )}

            {trainResult && (
                <div className="border-b border-slate-800 px-3 py-2.5">
                    <p className="text-xs text-slate-600 uppercase tracking-wider mb-2 font-medium">Metrics</p>
                    <div className="space-y-1.5">
                        {Object.entries(trainResult.metrics)
                            .filter(([k]) => !["confusion_matrix", "trials", "best_trial_score", "n_trials"].includes(k))
                            .map(([k, v]) => (
                                <div key={k} className="flex justify-between items-center">
                                    <span className="text-xs text-slate-500">{k.replace(/_/g, " ")}</span>
                                    <span className="text-xs font-mono text-white font-semibold">
                                        {typeof v === "number" ? v.toFixed(4) : String(v)}
                                    </span>
                                </div>
                            ))}
                    </div>
                </div>
            )}

            {fi.length > 0 && (
                <div className="px-3 py-2.5">
                    <p className="text-xs text-slate-600 uppercase tracking-wider mb-2.5 font-medium">
                        Feature importance
                    </p>
                    <div className="space-y-2">
                        {fi.map(([name, val], i) => (
                            <div key={name}>
                                <div className="flex justify-between mb-0.5">
                                    <span className="text-xs text-slate-500 truncate pr-2">{name}</span>
                                    <span className="text-xs font-mono text-slate-500">{Number(val).toFixed(3)}</span>
                                </div>
                                <div className="h-1 bg-slate-800 rounded-full">
                                    <div className="h-1 rounded-full transition-all"
                                        style={{
                                            width: `${(Number(val) / maxFi) * 100}%`,
                                            background: i === 0 ? "#3b82f6" : "#1e293b",
                                        }} />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {!result && !trainResult && (
                <div className="flex-1 flex items-center justify-center px-4 py-8">
                    <p className="text-xs text-slate-700 text-center leading-relaxed">
                        Results and AI explanations appear here after each step.
                    </p>
                </div>
            )}
        </aside>
    );
}