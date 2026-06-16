"use client";
import { useEffect, useState, useRef } from "react";
import { useParams } from "next/navigation";
import {
    getSession, getProfile, getDistributions, getCorrelation, getTargetAnalysis,
    runStep, trainModel,
    Session, DatasetProfile, StepResponse, TrainResponse,
    EDADistributions, EDACorrelation, EDATargetAnalysis,
} from "@/lib/api";
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
    Cell, LineChart, Line, ScatterChart, Scatter,
} from "recharts";
import {
    Loader2, CheckCircle2, Circle, ChevronRight,
    Sparkles, AlertTriangle, Download, TrendingUp
} from "lucide-react";

// ─── CONSTANTS ────────────────────────────────────────────────────────────────

const STEPS = [
    { id: 1, key: "profile", label: "Data profile", action: null },
    { id: 2, key: "eda", label: "EDA", action: null },
    { id: 3, key: "missing", label: "Missing values", action: "missing" },
    { id: 4, key: "outliers", label: "Outliers", action: "outliers" },
    { id: 5, key: "features", label: "Feature engineering", action: "features" },
    { id: 6, key: "encoding", label: "Encoding", action: "encoding" },
    { id: 7, key: "selection", label: "Feature selection", action: "selection" },
    { id: 8, key: "scaling", label: "Scaling", action: "scaling" },
    { id: 9, key: "train", label: "Train model", action: "train" },
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
        { value: "elasticnet", label: "ElasticNet", desc: "L1 + L2 combined" },
        { value: "svr", label: "SVR", desc: "Support vector regression" },
    ],
};

const STATS_KEYS = [
    "missing_before", "missing_after", "rows_before", "rows_after",
    "cols_before", "cols_after", "n_columns_scaled", "new_cols_created",
    "total_outliers_found", "new_features_created", "n_dropped",
];

const CHART_TYPES = [
    { value: "distribution", label: "Distributions" },
    { value: "correlation", label: "Correlation heatmap" },
    { value: "target", label: "Target analysis" },
    { value: "missing", label: "Missing values" },
];

// ─── MAIN PAGE ────────────────────────────────────────────────────────────────

export default function PlaygroundPage() {
    const { sessionId } = useParams<{ sessionId: string }>();
    const [session, setSession] = useState<Session | null>(null);
    const [profile, setProfile] = useState<DatasetProfile | null>(null);
    const [activeStep, setActiveStep] = useState(1);
    const [selected, setSelected] = useState<Record<string, string>>({});
    const [results, setResults] = useState<Record<string, StepResponse | TrainResponse>>({});
    const [loading, setLoading] = useState<string | null>(null);
    const [activeChart, setActiveChart] = useState("distribution");

    // EDA data
    const [distributions, setDistributions] = useState<EDADistributions | null>(null);
    const [correlation, setCorrelation] = useState<EDACorrelation | null>(null);
    const [targetData, setTargetData] = useState<EDATargetAnalysis | null>(null);
    const [edaLoading, setEdaLoading] = useState(false);

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
        ]).then(([d, c, t]) => {
            setDistributions(d);
            setCorrelation(c);
            setTargetData(t);
        }).finally(() => setEdaLoading(false));
    }, [activeStep, session]);

    async function apply(stepKey: string) {
        if (!session) return;
        setLoading(stepKey);
        try {
            if (stepKey === "train") {
                const model = selected["train"] || "random_forest";
                const res = await trainModel(session.id, model);
                setResults(r => ({ ...r, train: res }));
            } else {
                const technique = selected[stepKey] || TECHNIQUES[stepKey][0].value;
                const techObj = TECHNIQUES[stepKey].find(t => t.value === technique);
                const params = techObj?.params || {};
                const res = await runStep(session.id, stepKey, technique, params);
                setResults(r => ({ ...r, [stepKey]: res }));
            }
            setActiveStep(s => Math.min(s + 1, STEPS.length));
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(null);
        }
    }

    async function exportCode() {
        if (!session) return;
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/sessions/${session.id}/export`);
        const code = await res.text();
        const blob = new Blob([code], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `ml_pipeline_${session.dataset_id}.py`;
        a.click();
        URL.revokeObjectURL(url);
    }

    if (!session || !profile) {
        return (
            <div className="min-h-screen bg-slate-950 flex items-center justify-center">
                <Loader2 className="w-8 h-8 animate-spin text-blue-400" />
            </div>
        );
    }

    const lastResult = results[STEPS[activeStep - 1]?.key] as StepResponse | undefined;
    const trainResult = results["train"] as TrainResponse | undefined;
    const regressionModels = ["ridge", "lasso", "elasticnet", "svr"];

    return (
        <div className="h-screen bg-slate-950 flex flex-col overflow-hidden">

            {/* ── Top bar ── */}
            <header className="h-11 bg-slate-900 border-b border-slate-800 flex items-center px-4 gap-3 flex-shrink-0">
                <span className="text-white font-medium text-sm">ML Playground</span>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-slate-500 text-xs">Live</span>
                <div className="ml-auto flex items-center gap-2">
                    <span className="text-xs text-slate-500 capitalize">{session.dataset_id}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${session.task_type === "classification"
                            ? "bg-emerald-500/10 text-emerald-400"
                            : "bg-amber-500/10 text-amber-400"}`}>
                        {session.task_type}
                    </span>
                    <button onClick={exportCode}
                        className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-blue-600/20 text-blue-300 border border-blue-500/30 hover:bg-blue-600/30 transition-colors">
                        <Download className="w-3.5 h-3.5" /> Export .py
                    </button>
                </div>
            </header>

            {/* ── Body ── */}
            <div className="flex flex-1 overflow-hidden">

                {/* ── LEFT — Causes ── */}
                <aside className="w-52 bg-slate-900 border-r border-slate-800 flex flex-col flex-shrink-0 overflow-y-auto">
                    <div className="px-3 pt-3 pb-2">
                        <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Lifecycle</div>
                        <nav className="space-y-0.5">
                            {STEPS.map(step => {
                                const done = step.key === "profile" || step.key === "eda"
                                    ? activeStep > step.id
                                    : !!results[step.key];
                                const active = activeStep === step.id;
                                return (
                                    <button key={step.id} onClick={() => setActiveStep(step.id)}
                                        className={`w-full flex items-center gap-2 px-2.5 py-2 rounded-lg text-xs transition-all text-left ${active
                                                ? "bg-blue-600/20 text-blue-300 border border-blue-500/30"
                                                : done
                                                    ? "text-slate-300 hover:bg-slate-800"
                                                    : "text-slate-500 hover:bg-slate-800/50"}`}>
                                        <span className="flex-shrink-0">
                                            {done
                                                ? <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                                                : active
                                                    ? <ChevronRight className="w-3.5 h-3.5" />
                                                    : <Circle className="w-3.5 h-3.5" />}
                                        </span>
                                        <span className="flex-1">{step.label}</span>
                                        <span className="text-slate-600 text-xs">{step.id}</span>
                                    </button>
                                );
                            })}
                        </nav>
                    </div>

                    {/* Pipeline state summary */}
                    {Object.keys(results).length > 0 && (
                        <div className="mx-3 mt-2 mb-3 p-2.5 rounded-lg bg-slate-800/50 border border-slate-700/50">
                            <div className="text-xs text-slate-500 mb-2">Pipeline so far</div>
                            {Object.entries(results).map(([k, v]) => {
                                const r = v as StepResponse;
                                return (
                                    <div key={k} className="flex items-center gap-1.5 mb-1">
                                        <span className="w-1 h-1 rounded-full bg-emerald-400 flex-shrink-0" />
                                        <span className="text-xs text-slate-400 capitalize">{k.replace(/_/g, " ")}</span>
                                        {r.technique && (
                                            <span className="text-xs text-slate-600 truncate">{r.technique}</span>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </aside>

                {/* ── CENTER — Action ── */}
                <main className="flex-1 flex flex-col overflow-hidden bg-slate-950">

                    {/* Step header */}
                    <div className="px-5 pt-4 pb-3 border-b border-slate-800 flex-shrink-0 bg-slate-900/50">
                        <div className="flex items-center gap-2.5">
                            <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
                                Step {activeStep}
                            </span>
                            <h1 className="text-sm font-medium text-white">
                                {STEPS[activeStep - 1]?.label}
                            </h1>
                        </div>
                    </div>

                    {/* Scrollable content */}
                    <div className="flex-1 overflow-y-auto p-5 space-y-4">

                        {/* ── STEP 1: Profile ── */}
                        {activeStep === 1 && (
                            <ProfileView profile={profile} onNext={() => setActiveStep(2)} />
                        )}

                        {/* ── STEP 2: EDA ── */}
                        {activeStep === 2 && (
                            <EDAView
                                profile={profile}
                                distributions={distributions}
                                correlation={correlation}
                                targetData={targetData}
                                loading={edaLoading}
                                activeChart={activeChart}
                                setActiveChart={setActiveChart}
                                onNext={() => setActiveStep(3)}
                            />
                        )}

                        {/* ── STEPS 3–8: Preprocessing ── */}
                        {activeStep >= 3 && activeStep <= 8 && (() => {
                            const step = STEPS[activeStep - 1];
                            const techniques = TECHNIQUES[step.key] || [];
                            const sel = selected[step.key] || techniques[0]?.value;
                            const result = results[step.key] as StepResponse | undefined;
                            return (
                                <StepView
                                    step={step}
                                    techniques={techniques}
                                    selected={sel}
                                    onSelect={v => setSelected(s => ({ ...s, [step.key]: v }))}
                                    onApply={() => apply(step.key)}
                                    loading={loading === step.key}
                                    result={result}
                                    profile={profile}
                                />
                            );
                        })()}

                        {/* ── STEP 9: Train ── */}
                        {activeStep === 9 && (
                            <TrainView
                                taskType={session.task_type}
                                selected={selected["train"] || "random_forest"}
                                onSelect={v => setSelected(s => ({ ...s, train: v }))}
                                onApply={() => apply("train")}
                                loading={loading === "train"}
                                result={trainResult}
                                regressionModels={regressionModels}
                            />
                        )}

                    </div>

                    {/* ── Bottom chart bar — always visible ── */}
                    <BottomChartBar
                        activeStep={activeStep}
                        profile={profile}
                        distributions={distributions}
                        correlation={correlation}
                        targetData={targetData}
                        results={results}
                        trainResult={trainResult}
                    />
                </main>

                {/* ── RIGHT — Effects ── */}
                <RightPanel
                    activeStep={activeStep}
                    result={lastResult}
                    trainResult={trainResult}
                    session={session}
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
                    { label: "Duplicate rows", value: profile.duplicate_rows },
                ].map(item => (
                    <div key={item.label} className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-3">
                        <div className="text-lg font-bold text-white">{item.value}</div>
                        <div className="text-xs text-slate-400 mt-0.5">{item.label}</div>
                    </div>
                ))}
            </div>

            <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl overflow-hidden">
                <div className="px-4 py-2.5 border-b border-slate-700/50">
                    <span className="text-xs font-medium text-slate-300">Column overview</span>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                        <thead>
                            <tr className="text-slate-500 border-b border-slate-700/50 text-left">
                                <th className="px-4 py-2 font-medium">Column</th>
                                <th className="px-4 py-2 font-medium">Type</th>
                                <th className="px-4 py-2 font-medium">Missing</th>
                                <th className="px-4 py-2 font-medium">Unique</th>
                                <th className="px-4 py-2 font-medium">Notes</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/50">
                            {profile.columns.map(col => (
                                <tr key={col.name} className="text-slate-300 hover:bg-slate-800/30">
                                    <td className="px-4 py-2 font-mono">
                                        {col.name}
                                        {col.is_target && (
                                            <span className="ml-1.5 text-xs px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400">target</span>
                                        )}
                                    </td>
                                    <td className="px-4 py-2">
                                        <span className={`px-1.5 py-0.5 rounded text-xs ${col.type === "numeric"
                                                ? "bg-purple-500/10 text-purple-400"
                                                : "bg-orange-500/10 text-orange-400"}`}>
                                            {col.type}
                                        </span>
                                    </td>
                                    <td className="px-4 py-2">
                                        <span className={
                                            col.missing_pct > 20 ? "text-red-400"
                                                : col.missing_pct > 0 ? "text-amber-400"
                                                    : "text-emerald-400"}>
                                            {col.missing_pct}%
                                        </span>
                                    </td>
                                    <td className="px-4 py-2 text-slate-400">{col.unique_count}</td>
                                    <td className="px-4 py-2 text-slate-500">
                                        {col.unique_count > 20 && col.type === "categorical" ? "⚠ high cardinality" : ""}
                                        {col.missing_pct > 50 ? "⚠ >50% missing" : ""}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            <button onClick={onNext}
                className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded-lg font-medium transition-colors">
                Next: EDA →
            </button>
        </div>
    );
}

// ─── EDA VIEW ─────────────────────────────────────────────────────────────────

function EDAView({ profile, distributions, correlation, targetData, loading,
    activeChart, setActiveChart, onNext }: {
        profile: DatasetProfile;
        distributions: EDADistributions | null;
        correlation: EDACorrelation | null;
        targetData: EDATargetAnalysis | null;
        loading: boolean;
        activeChart: string;
        setActiveChart: (v: string) => void;
        onNext: () => void;
    }) {
    const [selectedCol, setSelectedCol] = useState<string>("");
    const numericCols = profile.columns.filter(c => c.type === "numeric").map(c => c.name);
    const col = selectedCol || numericCols[0] || "";

    if (loading) {
        return (
            <div className="flex items-center justify-center py-20">
                <Loader2 className="w-6 h-6 animate-spin text-blue-400 mr-2" />
                <span className="text-slate-400 text-sm">Loading EDA data...</span>
            </div>
        );
    }

    return (
        <div className="space-y-4 max-w-2xl">
            {/* Chart type selector */}
            <div className="flex items-center gap-2 flex-wrap">
                {CHART_TYPES.map(ct => (
                    <button key={ct.value} onClick={() => setActiveChart(ct.value)}
                        className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${activeChart === ct.value
                                ? "bg-blue-600/20 text-blue-300 border-blue-500/30"
                                : "text-slate-400 border-slate-700/50 hover:border-slate-600"}`}>
                        {ct.label}
                    </button>
                ))}
            </div>

            {/* Distribution chart */}
            {activeChart === "distribution" && distributions && (
                <div className="space-y-3">
                    <div className="flex items-center gap-2">
                        <select value={col} onChange={e => setSelectedCol(e.target.value)}
                            className="text-xs bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-slate-300 focus:outline-none focus:border-blue-500">
                            {Object.keys(distributions).map(c => (
                                <option key={c} value={c}>{c}</option>
                            ))}
                        </select>
                    </div>
                    {distributions[col] && (
                        <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
                            <div className="text-xs text-slate-400 mb-3 font-medium">{col} — distribution</div>
                            {distributions[col].type === "numeric" && distributions[col].histogram && (
                                <>
                                    <ResponsiveContainer width="100%" height={160}>
                                        <BarChart data={distributions[col].histogram!.counts.map((c, i) => ({
                                            x: distributions[col].histogram!.edges[i].toFixed(1), v: c
                                        }))}>
                                            <XAxis dataKey="x" tick={{ fill: "#64748b", fontSize: 9 }} interval={4} />
                                            <YAxis tick={{ fill: "#64748b", fontSize: 9 }} />
                                            <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", fontSize: 11 }} />
                                            <Bar dataKey="v" fill="#3b82f6" radius={[2, 2, 0, 0]} opacity={0.8} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                    {distributions[col].stats && (
                                        <div className="grid grid-cols-3 gap-2 mt-3">
                                            {Object.entries(distributions[col].stats!).map(([k, v]) => (
                                                <div key={k} className="bg-slate-900/50 rounded-lg p-2">
                                                    <div className="text-xs text-slate-500">{k}</div>
                                                    <div className="text-sm font-mono text-white">{Number(v).toFixed(3)}</div>
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
                                        <XAxis type="number" tick={{ fill: "#64748b", fontSize: 9 }} />
                                        <YAxis type="category" dataKey="label" tick={{ fill: "#64748b", fontSize: 9 }} width={80} />
                                        <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", fontSize: 11 }} />
                                        <Bar dataKey="count" fill="#8b5cf6" radius={[0, 2, 2, 0]} opacity={0.8} />
                                    </BarChart>
                                </ResponsiveContainer>
                            )}
                        </div>
                    )}
                </div>
            )}

            {/* Correlation heatmap */}
            {activeChart === "correlation" && correlation && (
                <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
                    <div className="text-xs text-slate-400 mb-3 font-medium">Correlation matrix</div>
                    <CorrelationHeatmap correlation={correlation} />
                </div>
            )}

            {/* Target analysis */}
            {activeChart === "target" && targetData && (
                <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
                    <div className="text-xs text-slate-400 mb-1 font-medium">
                        Target: <span className="text-white font-mono">{targetData.target}</span>
                    </div>
                    {targetData.is_imbalanced && (
                        <div className="text-xs text-amber-400 mb-3">
                            ⚠ Class imbalance detected (ratio: {targetData.class_balance?.toFixed(2)}) — consider SMOTE or class weights
                        </div>
                    )}
                    {targetData.distribution && (
                        <ResponsiveContainer width="100%" height={160}>
                            <BarChart data={targetData.distribution.labels.map((l, i) => ({
                                label: String(l), count: targetData.distribution!.counts[i]
                            }))}>
                                <XAxis dataKey="label" tick={{ fill: "#64748b", fontSize: 11 }} />
                                <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
                                <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", fontSize: 11 }} />
                                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                                    {targetData.distribution.labels.map((_, i) => (
                                        <Cell key={i} fill={i === 0 ? "#3b82f6" : i === 1 ? "#8b5cf6" : "#10b981"} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    )}
                    {targetData.histogram && (
                        <ResponsiveContainer width="100%" height={160}>
                            <BarChart data={targetData.histogram.counts.map((c, i) => ({
                                x: targetData.histogram!.edges[i].toFixed(1), v: c
                            }))}>
                                <XAxis dataKey="x" tick={{ fill: "#64748b", fontSize: 9 }} interval={4} />
                                <YAxis tick={{ fill: "#64748b", fontSize: 9 }} />
                                <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", fontSize: 11 }} />
                                <Bar dataKey="v" fill="#f59e0b" radius={[2, 2, 0, 0]} opacity={0.8} />
                            </BarChart>
                        </ResponsiveContainer>
                    )}
                </div>
            )}

            {/* Missing values chart */}
            {activeChart === "missing" && (
                <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
                    <div className="text-xs text-slate-400 mb-3 font-medium">Missing values per column</div>
                    <div className="space-y-2">
                        {profile.columns.map(col => (
                            <div key={col.name} className="flex items-center gap-3">
                                <span className="text-xs font-mono text-slate-400 w-24 truncate">{col.name}</span>
                                <div className="flex-1 bg-slate-700/50 rounded-full h-2">
                                    <div className="h-2 rounded-full transition-all"
                                        style={{
                                            width: `${col.missing_pct}%`,
                                            background: col.missing_pct > 20 ? "#ef4444"
                                                : col.missing_pct > 0 ? "#f59e0b" : "#10b981"
                                        }} />
                                </div>
                                <span className={`text-xs w-10 text-right font-mono ${col.missing_pct > 20 ? "text-red-400"
                                        : col.missing_pct > 0 ? "text-amber-400"
                                            : "text-emerald-400"}`}>
                                    {col.missing_pct}%
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <button onClick={onNext}
                className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded-lg font-medium transition-colors">
                Next: Missing values →
            </button>
        </div>
    );
}

// ─── CORRELATION HEATMAP ──────────────────────────────────────────────────────

function CorrelationHeatmap({ correlation }: { correlation: EDACorrelation }) {
    const { columns, matrix } = correlation;
    const n = columns.length;
    const size = Math.min(Math.floor(480 / n), 40);

    function color(v: number) {
        if (v > 0.7) return "#3b82f6";
        if (v > 0.4) return "#6366f1";
        if (v > 0.1) return "#8b5cf6";
        if (v > -0.1) return "#374151";
        if (v > -0.4) return "#dc2626";
        return "#991b1b";
    }

    return (
        <div className="overflow-x-auto">
            <div className="flex">
                <div style={{ width: 60 }} />
                {columns.map(c => (
                    <div key={c} style={{ width: size, fontSize: 8, color: "#64748b", textAlign: "center" }}
                        className="overflow-hidden whitespace-nowrap">
                        {c.substring(0, 5)}
                    </div>
                ))}
            </div>
            {matrix.map((row, i) => (
                <div key={i} className="flex items-center">
                    <div style={{ width: 60, fontSize: 9, color: "#64748b" }} className="truncate pr-1 text-right">
                        {columns[i]}
                    </div>
                    {row.map((val, j) => (
                        <div key={j}
                            style={{ width: size, height: size, background: color(val), opacity: 0.3 + Math.abs(val) * 0.7 }}
                            title={`${columns[i]} × ${columns[j]}: ${val.toFixed(2)}`}
                        />
                    ))}
                </div>
            ))}
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
            {/* Technique grid */}
            <div className="grid grid-cols-2 gap-2">
                {techniques.map(t => (
                    <button key={t.value} onClick={() => onSelect(t.value)}
                        className={`text-left px-3.5 py-3 rounded-xl border transition-all ${selected === t.value
                                ? "bg-blue-600/15 border-blue-500/40 shadow-sm shadow-blue-500/10"
                                : "bg-slate-800/40 border-slate-700/50 hover:border-slate-600"}`}>
                        <div className={`text-xs font-medium ${selected === t.value ? "text-blue-300" : "text-slate-200"}`}>
                            {t.label}
                        </div>
                        <div className="text-xs text-slate-500 mt-0.5">{t.desc}</div>
                    </button>
                ))}
            </div>

            {/* Before / After visualiser */}
            {step.key === "missing" && (
                <BeforeAfterMissing profile={profile} technique={selected} />
            )}

            <button onClick={onApply} disabled={!!loading}
                className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm rounded-lg font-medium transition-colors flex items-center gap-2">
                {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Applying...</> : `Apply ${selected} →`}
            </button>
        </div>
    );
}

// ─── BEFORE / AFTER MISSING ───────────────────────────────────────────────────

function BeforeAfterMissing({ profile, technique }: { profile: DatasetProfile; technique: string }) {
    const missingCols = profile.columns.filter(c => c.missing_count > 0).slice(0, 3);
    if (missingCols.length === 0) return null;

    function getAfterLabel(col: typeof missingCols[0]) {
        if (technique === "drop_rows") return "— row dropped";
        if (technique === "drop_cols") return "— column dropped";
        if (technique === "mean") return col.stats?.mean ? `${Number(col.stats.mean).toFixed(2)} (mean)` : "filled";
        if (technique === "median") return col.stats?.median ? `${Number(col.stats.median).toFixed(2)} (median)` : "filled";
        if (technique === "constant") return "0.0 (constant)";
        if (technique === "knn") return "KNN estimate";
        if (technique === "mice") return "MICE estimate";
        return "filled";
    }

    return (
        <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4">
            <div className="text-xs text-slate-400 font-medium mb-3">Preview — what changes</div>
            <div className="space-y-3">
                {missingCols.map(col => (
                    <div key={col.name}>
                        <div className="text-xs font-mono text-slate-400 mb-1.5">{col.name}
                            <span className="ml-1.5 text-slate-500">({col.missing_count} missing)</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="flex-1 grid grid-cols-4 gap-1">
                                {[...Array(4)].map((_, i) => (
                                    <div key={i} className={`px-2 py-1.5 rounded text-xs font-mono text-center ${i % 2 === 1
                                            ? "bg-red-500/15 text-red-400 border border-red-500/20"
                                            : "bg-slate-700/50 text-slate-300"}`}>
                                        {i % 2 === 1 ? "NULL" : col.stats?.mean ? Number(col.stats.mean).toFixed(1) : "—"}
                                    </div>
                                ))}
                            </div>
                            <span className="text-slate-500 text-sm">→</span>
                            <div className="flex-1 grid grid-cols-4 gap-1">
                                {[...Array(4)].map((_, i) => (
                                    <div key={i} className={`px-2 py-1.5 rounded text-xs font-mono text-center ${i % 2 === 1
                                            ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/20"
                                            : "bg-slate-700/50 text-slate-300"}`}>
                                        {i % 2 === 1 ? getAfterLabel(col).split(" ")[0] : (col.stats?.mean ? Number(col.stats.mean).toFixed(1) : "—")}
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

// ─── BOTTOM CHART BAR ─────────────────────────────────────────────────────────

function BottomChartBar({ activeStep, profile, distributions, correlation,
    targetData, results, trainResult }: {
        activeStep: number;
        profile: DatasetProfile;
        distributions: EDADistributions | null;
        correlation: EDACorrelation | null;
        targetData: EDATargetAnalysis | null;
        results: Record<string, any>;
        trainResult?: TrainResponse;
    }) {
    const [open, setOpen] = useState(false);
    const [chartType, setChartType] = useState("distribution");
    const [selCol, setSelCol] = useState("");

    const numericCols = profile.columns.filter(c => c.type === "numeric").map(c => c.name);
    const col = selCol || numericCols[0] || "";

    const options = [
        { value: "distribution", label: "Distribution" },
        { value: "missing", label: "Missing values" },
        { value: "correlation", label: "Correlations" },
        { value: "target", label: "Target analysis" },
        ...(trainResult ? [{ value: "importance", label: "Feature importance" }] : []),
    ];

    return (
        <div className="border-t border-slate-800 bg-slate-900/80 flex-shrink-0">
            {/* Toggle bar */}
            <div className="flex items-center gap-3 px-4 py-2">
                <button onClick={() => setOpen(o => !o)}
                    className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors">
                    <TrendingUp className="w-3.5 h-3.5" />
                    Visualisations
                    <span className="text-slate-600">{open ? "▲" : "▼"}</span>
                </button>

                {open && (
                    <>
                        <select value={chartType} onChange={e => setChartType(e.target.value)}
                            className="text-xs bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1 text-slate-300 focus:outline-none focus:border-blue-500">
                            {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                        </select>

                        {chartType === "distribution" && (
                            <select value={col} onChange={e => setSelCol(e.target.value)}
                                className="text-xs bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1 text-slate-300 focus:outline-none focus:border-blue-500">
                                {Object.keys(distributions || {}).map(c => (
                                    <option key={c} value={c}>{c}</option>
                                ))}
                            </select>
                        )}
                    </>
                )}
            </div>

            {/* Chart content */}
            {open && (
                <div className="px-4 pb-3">
                    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-3" style={{ height: 180 }}>

                        {chartType === "distribution" && distributions && distributions[col] && (
                            <ResponsiveContainer width="100%" height="100%">
                                {distributions[col].type === "numeric" ? (
                                    <BarChart data={distributions[col].histogram!.counts.map((c, i) => ({
                                        x: distributions[col].histogram!.edges[i].toFixed(1), v: c
                                    }))}>
                                        <XAxis dataKey="x" tick={{ fill: "#64748b", fontSize: 9 }} interval={4} />
                                        <YAxis tick={{ fill: "#64748b", fontSize: 9 }} />
                                        <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", fontSize: 10 }} />
                                        <Bar dataKey="v" fill="#3b82f6" radius={[2, 2, 0, 0]} opacity={0.8} />
                                    </BarChart>
                                ) : (
                                    <BarChart data={distributions[col].bar!.labels.map((l, i) => ({
                                        label: l, count: distributions[col].bar!.counts[i]
                                    }))} layout="vertical">
                                        <XAxis type="number" tick={{ fill: "#64748b", fontSize: 9 }} />
                                        <YAxis type="category" dataKey="label" tick={{ fill: "#64748b", fontSize: 9 }} width={70} />
                                        <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", fontSize: 10 }} />
                                        <Bar dataKey="count" fill="#8b5cf6" radius={[0, 2, 2, 0]} opacity={0.8} />
                                    </BarChart>
                                )}
                            </ResponsiveContainer>
                        )}

                        {chartType === "missing" && (
                            <div className="overflow-y-auto h-full space-y-1.5 pr-1">
                                {profile.columns.map(c => (
                                    <div key={c.name} className="flex items-center gap-2">
                                        <span className="text-xs font-mono text-slate-500 w-20 truncate">{c.name}</span>
                                        <div className="flex-1 bg-slate-700/50 rounded-full h-1.5">
                                            <div className="h-1.5 rounded-full"
                                                style={{
                                                    width: `${c.missing_pct}%`,
                                                    background: c.missing_pct > 20 ? "#ef4444"
                                                        : c.missing_pct > 0 ? "#f59e0b" : "#10b981"
                                                }} />
                                        </div>
                                        <span className="text-xs text-slate-500 w-8 text-right">{c.missing_pct}%</span>
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
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={targetData.distribution.labels.map((l, i) => ({
                                    label: String(l), count: targetData.distribution!.counts[i]
                                }))}>
                                    <XAxis dataKey="label" tick={{ fill: "#64748b", fontSize: 10 }} />
                                    <YAxis tick={{ fill: "#64748b", fontSize: 10 }} />
                                    <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", fontSize: 10 }} />
                                    <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        )}

                        {chartType === "importance" && trainResult && (
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                    layout="vertical"
                                    data={Object.entries(trainResult.feature_importance).slice(0, 8).map(([k, v]) => ({ name: k, value: v }))}>
                                    <XAxis type="number" tick={{ fill: "#64748b", fontSize: 9 }} />
                                    <YAxis type="category" dataKey="name" tick={{ fill: "#64748b", fontSize: 9 }} width={90} />
                                    <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", fontSize: 10 }} />
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

// ─── TRAIN VIEW ───────────────────────────────────────────────────────────────

function TrainView({ taskType, selected, onSelect, onApply, loading, result, regressionModels }: {
    taskType: string;
    selected: string;
    onSelect: (v: string) => void;
    onApply: () => void;
    loading: boolean;
    result?: TrainResponse;
    regressionModels: string[];
}) {
    const models = TECHNIQUES.train.filter(m =>
        taskType === "regression"
            ? regressionModels.includes(m.value) || ["random_forest", "xgboost", "lightgbm", "knn"].includes(m.value)
            : !regressionModels.includes(m.value)
    );

    return (
        <div className="space-y-4 max-w-2xl">
            <div className="grid grid-cols-2 gap-2">
                {models.map(m => (
                    <button key={m.value} onClick={() => onSelect(m.value)}
                        className={`text-left px-3.5 py-3 rounded-xl border transition-all ${selected === m.value
                                ? "bg-emerald-600/15 border-emerald-500/40"
                                : "bg-slate-800/40 border-slate-700/50 hover:border-slate-600"}`}>
                        <div className={`text-xs font-medium ${selected === m.value ? "text-emerald-300" : "text-slate-200"}`}>
                            {m.label}
                        </div>
                        <div className="text-xs text-slate-500 mt-0.5">{m.desc}</div>
                    </button>
                ))}
            </div>

            <button onClick={onApply} disabled={!!loading}
                className="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-sm rounded-lg font-medium transition-colors flex items-center gap-2">
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
                    <div className="text-xs text-slate-500 flex gap-4">
                        <span>Train: <span className="text-slate-300">{result.train_size}</span></span>
                        <span>Test: <span className="text-slate-300">{result.test_size}</span></span>
                        <span>Features: <span className="text-slate-300">{result.n_features}</span></span>
                    </div>
                </div>
            )}
        </div>
    );
}

// ─── RIGHT PANEL ──────────────────────────────────────────────────────────────

function RightPanel({ activeStep, result, trainResult, session }: {
    activeStep: number;
    result?: StepResponse;
    trainResult?: TrainResponse;
    session: Session;
}) {
    const fi = trainResult
        ? Object.entries(trainResult.feature_importance).slice(0, 8)
        : [];

    return (
        <aside className="w-64 bg-slate-900 border-l border-slate-800 flex flex-col overflow-y-auto flex-shrink-0">

            {/* AI explanation */}
            <div className="border-b border-slate-800">
                <div className="flex items-center gap-2 px-3 py-2.5">
                    <Sparkles className="w-3.5 h-3.5 text-blue-400" />
                    <span className="text-xs font-medium text-slate-300">AI explanation</span>
                </div>
                <div className="px-3 pb-3">
                    {result?.ai_explanation ? (
                        <p className="text-xs text-slate-400 leading-relaxed">{result.ai_explanation}</p>
                    ) : (
                        <p className="text-xs text-slate-600 italic">
                            Apply a technique to see the AI explain what changed and why.
                        </p>
                    )}
                </div>
            </div>

            {/* Warnings */}
            {result?.warnings && result.warnings.length > 0 && (
                <div className="border-b border-slate-800 px-3 py-2.5 space-y-2">
                    {result.warnings.map((w, i) => (
                        <div key={i} className="flex gap-1.5 text-xs text-amber-400">
                            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                            <span>{w}</span>
                        </div>
                    ))}
                </div>
            )}

            {/* Step stats */}
            {result?.stats && (
                <div className="border-b border-slate-800 px-3 py-2.5">
                    <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Stats</div>
                    <div className="space-y-1.5">
                        {Object.entries(result.stats)
                            .filter(([k]) => STATS_KEYS.includes(k))
                            .map(([k, v]) => (
                                <div key={k} className="flex justify-between">
                                    <span className="text-xs text-slate-500">{k.replace(/_/g, " ")}</span>
                                    <span className="text-xs font-mono text-slate-300">{String(v)}</span>
                                </div>
                            ))}
                    </div>
                </div>
            )}

            {/* Train metrics in right panel */}
            {trainResult && (
                <div className="border-b border-slate-800 px-3 py-2.5">
                    <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Metrics</div>
                    <div className="space-y-1.5">
                        {Object.entries(trainResult.metrics)
                            .filter(([k]) => k !== "confusion_matrix")
                            .map(([k, v]) => (
                                <div key={k} className="flex justify-between">
                                    <span className="text-xs text-slate-500">{k.replace(/_/g, " ")}</span>
                                    <span className="text-xs font-mono text-white font-medium">
                                        {typeof v === "number" ? v.toFixed(4) : String(v)}
                                    </span>
                                </div>
                            ))}
                    </div>
                </div>
            )}

            {/* Feature importance bars */}
            {fi.length > 0 && (
                <div className="px-3 py-2.5">
                    <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Feature importance</div>
                    <div className="space-y-2">
                        {fi.map(([name, val], i) => (
                            <div key={name} className="space-y-0.5">
                                <div className="flex justify-between">
                                    <span className="text-xs text-slate-400 truncate">{name}</span>
                                    <span className="text-xs font-mono text-slate-400">{Number(val).toFixed(3)}</span>
                                </div>
                                <div className="h-1.5 bg-slate-700/50 rounded-full">
                                    <div className="h-1.5 rounded-full transition-all"
                                        style={{
                                            width: `${(val / fi[0][1]) * 100}%`,
                                            background: i === 0 ? "#3b82f6" : "#334155"
                                        }} />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Empty state */}
            {!result && !trainResult && (
                <div className="flex-1 flex items-center justify-center px-4">
                    <p className="text-xs text-slate-600 text-center leading-relaxed">
                        Results and AI explanations appear here after each step.
                    </p>
                </div>
            )}
        </aside>
    );
}