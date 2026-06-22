"use client";
import { useEffect, useState, useRef, useMemo } from "react";
import { useParams, useRouter } from "next/navigation";
    import {
        getSession, getProfile, getDistributions, getCorrelation, getTargetAnalysis,
        getRuns, runStep, trainModel, tuneModel, gridSearch, resetStep,
        Session, DatasetProfile, StepResponse, TrainResponse,
        EDADistributions, EDACorrelation, EDATargetAnalysis, RunRecord,
        getDatasets, AutoMLStatus,
    } from "@/lib/api";
    import MetricCard from "@/components/MetricCard";
    import FeatureImportance from "@/components/FeatureImportance";
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
    LineChart, Line, RadarChart, Radar, PolarGrid, PolarAngleAxis,
    AreaChart, Area, PieChart, Pie, Legend, ScatterChart, Scatter,
} from "recharts";
import {
    Loader2, CheckCircle2, Circle, ChevronRight,
    Sparkles, AlertTriangle, Download, TrendingUp,
    ArrowRight, RotateCcw, GitCompare, Share2, Home,
} from "lucide-react";

// ─── LABEL MAP ────────────────────────────────────────────────────────────────
// Every raw key that appears in the UI must go through this map.
// No underscores, no lowercase-only, no raw Python identifiers shown to users.

const LABELS: Record<string, string> = {
    // Metrics
    accuracy: "Accuracy",
    f1_weighted: "F1 Score (Weighted)",
    f1_macro: "F1 Score (Macro)",
    roc_auc: "ROC-AUC",
    cv_mean: "CV Mean Score",
    cv_std: "CV Std Dev",
    rmse: "RMSE",
    mae: "MAE",
    r2: "R² Score",
    best_trial_score: "Best Trial Score",
    n_trials: "Trials Run",
    best_cv_score: "Best CV Score",
    imbalance_warning: "Imbalance Warning",
    // Stats
    missing_before: "Missing Values (Before)",
    missing_after: "Missing Values (After)",
    rows_before: "Rows (Before)",
    rows_after: "Rows (After)",
    cols_before: "Columns (Before)",
    cols_after: "Columns (After)",
    n_columns_scaled: "Columns Scaled",
    new_cols_created: "New Columns Created",
    total_outliers_found: "Total Outliers Found",
    new_features_created: "New Features Created",
    n_dropped: "Features Dropped",
    components: "PCA Components",
    total_variance_retained: "Variance Retained (%)",
    // Tasks
    classification: "Classification",
    regression: "Regression",
};

function label(key: string): string {
    return LABELS[key] ?? key.split("_").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}

// ─── STEPS ────────────────────────────────────────────────────────────────────

const STEPS = [
    { id: 1, key: "profile", label: "Data Profile", action: null },
    { id: 2, key: "eda", label: "Exploratory Analysis", action: null },
    { id: 3, key: "missing", label: "Missing Values", action: "missing" },
    { id: 4, key: "outliers", label: "Outlier Treatment", action: "outliers" },
    { id: 5, key: "features", label: "Feature Engineering", action: "features" },
    { id: 6, key: "encoding", label: "Encoding", action: "encoding" },
    { id: 7, key: "selection", label: "Feature Selection", action: "selection" },
    { id: 8, key: "pca", label: "Dimensionality Reduction", action: "pca" },
    { id: 9, key: "scaling", label: "Scaling", action: "scaling" },
    { id: 10, key: "train", label: "Model Training", action: "train" },
    { id: 11, key: "tune", label: "Hyperparameter Tuning", action: "tune" },
    { id: 12, key: "explain", label: "Explainability", action: null },
    { id: 13, key: "compare", label: "Experiment Comparison", action: null },
    { id: 14, key: "predict", label: "Make a Prediction" },
];

// ─── TECHNIQUES ───────────────────────────────────────────────────────────────

const TECHNIQUES: Record<string, { value: string; label: string; desc: string; params?: Record<string, any> }[]> = {
    missing: [
        { value: "mean", label: "Mean Imputation", desc: "Replace nulls with column mean. Best for normally distributed data." },
        { value: "median", label: "Median Imputation", desc: "Replace nulls with median. Robust to outliers." },
        { value: "mode", label: "Mode Imputation", desc: "Replace nulls with the most frequent value." },
        { value: "knn", label: "KNN Imputation", desc: "Use K nearest neighbours to estimate missing values.", params: { n_neighbors: 5 } },
        { value: "mice", label: "MICE", desc: "Multivariate Imputation by Chained Equations. Most accurate, slowest." },
        { value: "random_sample", label: "Random Sample", desc: "Fill with random values drawn from observed distribution." },
        { value: "missing_indicator", label: "Missing Indicator", desc: "Add binary flag columns then fill with median. Preserves missingness signal." },
        { value: "constant", label: "Constant Fill", desc: "Replace all nulls with a fixed value (default 0).", params: { fill_value: 0 } },
        { value: "drop_rows", label: "Drop Rows", desc: "Remove all rows containing any null. Loses data." },
        { value: "drop_cols", label: "Drop Columns", desc: "Drop columns with >50% missing.", params: { threshold: 0.5 } },
    ],
    outliers: [
        { value: "iqr_cap", label: "IQR Capping (Winsorize)", desc: "Cap values at Q1 − 1.5×IQR and Q3 + 1.5×IQR. Non-destructive." },
        { value: "zscore_remove", label: "Z-Score Removal", desc: "Remove rows where any feature exceeds 3 standard deviations.", params: { threshold: 3 } },
        { value: "percentile_cap", label: "Percentile Capping", desc: "Cap at 1st and 99th percentile.", params: { lower: 1, upper: 99 } },
        { value: "log_transform", label: "Log Transform", desc: "Apply log(x+1) to compress extreme values and reduce skew." },
        { value: "keep", label: "Keep (Document Only)", desc: "Count outliers but apply no treatment. Use when outliers are valid." },
    ],
    features: [
        { value: "none", label: "Skip This Step", desc: "No feature engineering. Move directly to encoding." },
        { value: "polynomial", label: "Polynomial Features", desc: "Create degree-2 terms (x², x·y). Captures non-linear patterns." },
        { value: "interaction", label: "Interaction Terms", desc: "Pairwise products between numeric features." },
        { value: "log_features", label: "Log Transform", desc: "Create log(x+1) copies of numeric features." },
        { value: "reciprocal", label: "Reciprocal (1/x)", desc: "Create 1/x copies. Useful for rate features." },
        { value: "sqrt_features", label: "Square Root", desc: "Create √x copies. Mild compression of large values." },
        { value: "ratio", label: "Ratio Features", desc: "Create col_A / col_B ratios." },
        { value: "binning", label: "Binning", desc: "Discretise numeric columns into N bins.", params: { n_bins: 5 } },
        { value: "date_decompose", label: "Date Decomposition", desc: "Extract year, month, day, weekday from date columns." },
    ],
    encoding: [
        { value: "onehot", label: "One-Hot Encoding", desc: "Create binary columns per category. Best for nominal data with few categories." },
        { value: "label", label: "Label Encoding", desc: "Assign integer codes. Safe for trees, avoid for linear models." },
        { value: "ordinal", label: "Ordinal Encoding", desc: "Order-preserving integers. Use when categories have a natural order." },
        { value: "frequency", label: "Frequency Encoding", desc: "Replace category with its frequency in the dataset." },
        { value: "target", label: "Target Mean Encoding", desc: "Replace category with mean of target variable." },
    ],
    selection: [
        { value: "none", label: "Keep All Features", desc: "No feature selection. All columns passed to the model." },
        { value: "variance_threshold", label: "Variance Threshold", desc: "Drop features with near-zero variance (no information).", params: { threshold: 0.01 } },
        { value: "correlation", label: "Correlation Filter", desc: "Drop one of each pair of highly correlated features (>0.95).", params: { threshold: 0.95 } },
        { value: "mutual_info", label: "Mutual Information", desc: "Keep top-k features most statistically dependent on target.", params: { k: 10 } },
    ],
    pca: [
        { value: "none", label: "Skip PCA", desc: "No dimensionality reduction. Proceed with current feature set." },
        { value: "pca_auto", label: "PCA — 95% Variance", desc: "Keep minimum components that explain 95% of total variance." },
        { value: "pca_fixed", label: "PCA — Fixed Components", desc: "Reduce to a fixed number of principal components.", params: { n_components: 5 } },
    ],
    scaling: [
        { value: "standard", label: "Standard Scaler", desc: "Zero mean, unit variance (z-score). Most common choice." },
        { value: "minmax", label: "Min-Max Scaler", desc: "Scale to [0, 1] range. Sensitive to outliers." },
        { value: "robust", label: "Robust Scaler", desc: "Uses median and IQR. Best when outliers are present." },
        { value: "maxabs", label: "MaxAbs Scaler", desc: "Scale by maximum absolute value. Preserves zero entries." },
        { value: "quantile", label: "Quantile Transformer", desc: "Maps to uniform distribution. Reduces impact of outliers." },
        { value: "power", label: "Power Transformer", desc: "Yeo-Johnson transform. Makes features more Gaussian." },
        { value: "none", label: "No Scaling", desc: "Skip scaling. Correct for tree models (RF, XGBoost, LightGBM)." },
    ],
    train: [
        { value: "random_forest", label: "Random Forest", desc: "Ensemble of decision trees. Robust, interpretable feature importance." },
        { value: "xgboost", label: "XGBoost", desc: "Gradient boosted trees. Often best out-of-box performance." },
        { value: "logistic_regression", label: "Logistic Regression", desc: "Linear classification. Fast, interpretable, needs scaling." },
        { value: "lightgbm", label: "LightGBM", desc: "Fast gradient boosting. Efficient on large datasets." },
        { value: "catboost", label: "CatBoost", desc: "Handles categoricals natively. Strong with mixed data." },
        { value: "adaboost", label: "AdaBoost", desc: "Adaptive boosting. Focuses on misclassified samples." },
        { value: "gradient_boosting", label: "Gradient Boosting", desc: "Sequential boosting with decision trees." },
        { value: "decision_tree", label: "Decision Tree", desc: "Single tree. Highly interpretable, prone to overfitting." },
        { value: "knn", label: "K-Nearest Neighbours", desc: "Instance-based learning. No training phase." },
        { value: "naive_bayes", label: "Naive Bayes", desc: "Probabilistic classifier based on Bayes theorem." },
        { value: "svm", label: "Support Vector Machine", desc: "Maximum-margin classifier. Works well in high dimensions." },
        { value: "ridge", label: "Ridge Regression", desc: "L2 regularised linear regression." },
        { value: "lasso", label: "Lasso Regression", desc: "L1 regularised regression. Performs feature selection." },
        { value: "elasticnet", label: "ElasticNet", desc: "L1 + L2 combined regularisation." },
        { value: "svr", label: "Support Vector Regression", desc: "SVM for regression tasks." },
    ],
};

const STATS_KEYS = [
    "missing_before", "missing_after", "rows_before", "rows_after",
    "cols_before", "cols_after", "n_columns_scaled", "new_cols_created",
    "total_outliers_found", "new_features_created", "n_dropped",
    "components", "total_variance_retained",
];

// Helper function to normalize technique names for deduplication
// Backend may return either the value ("mean") or label ("Mean Imputation")
function getTechniqueValue(technique: string, TECHNIQUES: Record<string, { value: string; label: string }[]>): string {
    if (!technique) return "";
    
    const techniqueLower = technique.toLowerCase().trim();
    
    // Search through all technique categories
    for (const techniques of Object.values(TECHNIQUES)) {
        // Try to match by value (case-insensitive)
        const byValue = techniques.find(t => t.value.toLowerCase() === techniqueLower);
        if (byValue) return byValue.value;
        
        // Try to match by label (case-insensitive)
        const byLabel = techniques.find(t => t.label.toLowerCase() === techniqueLower);
        if (byLabel) return byLabel.value;
    }
    
    // Fallback: return as-is but normalized
    return techniqueLower;
}

const TOOLTIP_STYLE = {
    background: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: 8,
    fontSize: 11,
    color: "#cbd5e1",
};

// ─── MODEL PARAMS ─────────────────────────────────────────────────────────────

interface ParamDef {
    key: string;
    label: string;
    type: "int" | "float" | "select";
    min?: number;
    max?: number;
    step?: number;
    default: number | string;
    options?: string[];
    effect: string; // shown as behaviour preview
}

const MODEL_PARAMS: Record<string, ParamDef[]> = {
    random_forest: [
        { key: "n_estimators", label: "Number of Trees", type: "int", min: 10, max: 500, step: 10, default: 100, effect: "More trees → more stable predictions, slower training" },
        { key: "max_depth", label: "Max Tree Depth", type: "int", min: 1, max: 30, step: 1, default: 10, effect: "Deeper → captures more patterns, higher overfitting risk" },
        { key: "min_samples_split", label: "Min Samples to Split", type: "int", min: 2, max: 20, step: 1, default: 2, effect: "Higher → simpler trees, better generalisation" },
        { key: "min_samples_leaf", label: "Min Samples at Leaf", type: "int", min: 1, max: 20, step: 1, default: 1, effect: "Higher → smoother predictions, less variance" },
        { key: "max_features", label: "Features per Split", type: "select", default: "sqrt", options: ["sqrt", "log2", "auto"], effect: "sqrt is default for classification trees" },
    ],
    xgboost: [
        { key: "n_estimators", label: "Boosting Rounds", type: "int", min: 10, max: 500, step: 10, default: 100, effect: "More rounds → lower bias, risk of overfitting" },
        { key: "max_depth", label: "Max Tree Depth", type: "int", min: 1, max: 15, step: 1, default: 6, effect: "Shallower trees generalise better with boosting" },
        { key: "learning_rate", label: "Learning Rate (eta)", type: "float", min: 0.01, max: 0.5, step: 0.01, default: 0.1, effect: "Lower → needs more rounds but generalises better" },
        { key: "subsample", label: "Row Subsample Ratio", type: "float", min: 0.5, max: 1.0, step: 0.05, default: 1.0, effect: "Lower → reduces overfitting through randomness" },
        { key: "colsample_bytree", label: "Column Sample Ratio", type: "float", min: 0.3, max: 1.0, step: 0.05, default: 1.0, effect: "Lower → feature randomisation, reduces overfitting" },
        { key: "reg_lambda", label: "L2 Regularisation", type: "float", min: 0.0, max: 10.0, step: 0.1, default: 1.0, effect: "Higher → stronger regularisation, simpler model" },
    ],
    lightgbm: [
        { key: "n_estimators", label: "Boosting Rounds", type: "int", min: 10, max: 500, step: 10, default: 100, effect: "More rounds → better fit, potential overfitting" },
        { key: "learning_rate", label: "Learning Rate", type: "float", min: 0.01, max: 0.5, step: 0.01, default: 0.1, effect: "Lower → slower but more generalised" },
        { key: "num_leaves", label: "Max Leaves", type: "int", min: 10, max: 200, step: 5, default: 31, effect: "More leaves → more complex model" },
        { key: "min_child_samples", label: "Min Samples per Leaf", type: "int", min: 1, max: 50, step: 1, default: 20, effect: "Higher → prevents overfitting on small datasets" },
        { key: "reg_alpha", label: "L1 Regularisation", type: "float", min: 0.0, max: 5.0, step: 0.1, default: 0.0, effect: "Higher → sparsity in feature weights" },
    ],
    catboost: [
        { key: "iterations", label: "Iterations", type: "int", min: 10, max: 500, step: 10, default: 100, effect: "More iterations → better fit" },
        { key: "learning_rate", label: "Learning Rate", type: "float", min: 0.01, max: 0.5, step: 0.01, default: 0.1, effect: "Lower → more robust learning" },
        { key: "depth", label: "Tree Depth", type: "int", min: 1, max: 10, step: 1, default: 6, effect: "Deeper → more complex, slower" },
        { key: "l2_leaf_reg", label: "L2 Regularisation", type: "float", min: 0.0, max: 10.0, step: 0.5, default: 3.0, effect: "Higher → prevents overfitting" },
    ],
    adaboost: [
        { key: "n_estimators", label: "Estimators", type: "int", min: 10, max: 300, step: 10, default: 50, effect: "More estimators → better accuracy, slower" },
        { key: "learning_rate", label: "Learning Rate", type: "float", min: 0.01, max: 2.0, step: 0.01, default: 1.0, effect: "Lower → requires more estimators to converge" },
    ],
    logistic_regression: [
        { key: "C", label: "Inverse Regularisation (C)", type: "float", min: 0.001, max: 100.0, step: 0.1, default: 1.0, effect: "Lower C → stronger regularisation, simpler model" },
        { key: "max_iter", label: "Max Iterations", type: "int", min: 100, max: 5000, step: 100, default: 1000, effect: "Increase if model fails to converge" },
    ],
    decision_tree: [
        { key: "max_depth", label: "Max Depth", type: "int", min: 1, max: 30, step: 1, default: 5, effect: "Deeper → more complex, prone to overfitting" },
        { key: "min_samples_split", label: "Min Samples to Split", type: "int", min: 2, max: 50, step: 1, default: 2, effect: "Higher → simpler tree, better generalisation" },
        { key: "min_samples_leaf", label: "Min Samples at Leaf", type: "int", min: 1, max: 20, step: 1, default: 1, effect: "Higher → smoother decision boundary" },
        { key: "criterion", label: "Split Criterion", type: "select", default: "gini", options: ["gini", "entropy"], effect: "Gini is faster, entropy can be more accurate" },
    ],
    knn: [
        { key: "n_neighbors", label: "Number of Neighbours (K)", type: "int", min: 1, max: 50, step: 1, default: 5, effect: "Higher K → smoother boundary, more bias" },
        { key: "weights", label: "Weighting", type: "select", min: 0, max: 0, step: 0, default: "uniform", options: ["uniform", "distance"], effect: "Distance weighting gives closer neighbours more influence" },
        { key: "p", label: "Distance Metric (p)", type: "int", min: 1, max: 2, step: 1, default: 2, effect: "p=1 is Manhattan, p=2 is Euclidean distance" },
    ],
    gradient_boosting: [
        { key: "n_estimators", label: "Boosting Rounds", type: "int", min: 10, max: 500, step: 10, default: 100, effect: "More rounds → better fit, slower training" },
        { key: "max_depth", label: "Max Tree Depth", type: "int", min: 1, max: 10, step: 1, default: 3, effect: "Shallow trees work best for boosting (3-5)" },
        { key: "learning_rate", label: "Learning Rate", type: "float", min: 0.01, max: 0.5, step: 0.01, default: 0.1, effect: "Lower → needs more rounds, generalises better" },
        { key: "subsample", label: "Subsample Ratio", type: "float", min: 0.5, max: 1.0, step: 0.05, default: 1.0, effect: "Lower → stochastic boosting, reduces overfitting" },
    ],
    naive_bayes: [],
    svm: [
        { key: "C", label: "Regularisation (C)", type: "float", min: 0.01, max: 100.0, step: 0.1, default: 1.0, effect: "Lower C → wider margin, more misclassifications allowed" },
        { key: "kernel", label: "Kernel", type: "select", min: 0, max: 0, step: 0, default: "rbf", options: ["rbf", "linear", "poly", "sigmoid"], effect: "RBF works well for most non-linear problems" },
    ],
    ridge: [
        { key: "alpha", label: "Alpha (L2 strength)", type: "float", min: 0.01, max: 100.0, step: 0.1, default: 1.0, effect: "Higher → stronger shrinkage of coefficients" },
    ],
    lasso: [
        { key: "alpha", label: "Alpha (L1 strength)", type: "float", min: 0.001, max: 10.0, step: 0.01, default: 1.0, effect: "Higher → more features shrunk to exactly zero" },
    ],
    elasticnet: [
        { key: "alpha", label: "Alpha", type: "float", min: 0.001, max: 10.0, step: 0.01, default: 1.0, effect: "Controls overall regularisation strength" },
        { key: "l1_ratio", label: "L1 Ratio", type: "float", min: 0.0, max: 1.0, step: 0.05, default: 0.5, effect: "0 = pure Ridge, 1 = pure Lasso, 0.5 = balanced" },
    ],
    svr: [
        { key: "C", label: "Regularisation (C)", type: "float", min: 0.01, max: 100.0, step: 0.1, default: 1.0, effect: "Lower → smoother regression line, more tolerance" },
        { key: "epsilon", label: "Epsilon (tube width)", type: "float", min: 0.0, max: 1.0, step: 0.01, default: 0.1, effect: "Larger → more tolerance for prediction errors" },
    ],
};

// ─── MAIN PAGE ────────────────────────────────────────────────────────────────

export default function PlaygroundPage() {
    const { sessionId } = useParams<{ sessionId: string }>();
    const router = useRouter();

    const [session, setSession] = useState<Session | null>(null);
    const [profile, setProfile] = useState<DatasetProfile | null>(null);
    const [activeStep, setActiveStep] = useState(1);
    const [selected, setSelected] = useState<Record<string, string>>({});
    // Each step can have multiple results — index 0 is latest, 1 is previous
    const [results, setResults] = useState<Record<string, StepResponse[]>>({});
    const [trainResults, setTrainResults] = useState<TrainResponse[]>([]);
    const [tuneResult, setTuneResult] = useState<TrainResponse | null>(null);
    const [loading, setLoading] = useState<string | null>(null);
    const [distributions, setDist] = useState<EDADistributions | null>(null);
    const [correlation, setCorr] = useState<EDACorrelation | null>(null);
    const [targetData, setTarget] = useState<EDATargetAnalysis | null>(null);
    const [edaLoading, setEdaLoading] = useState(false);
    const [runs, setRuns] = useState<RunRecord[]>([]);
    const [chartOpen, setChartOpen] = useState(false);
    const [chartType, setChartType] = useState("distribution");
    const [chartCol, setChartCol] = useState("");

    const [trainParams, setTrainParams] = useState<Record<string, any>>({});

    const [datasetName, setDatasetName] = useState<string>("");
    const [saving, setSaving] = useState(false);
    const [sharing, setSharing] = useState(false);
    const [shareUrl, setShareUrl] = useState<string | null>(null);
    const [showSaveDialog, setShowSaveDialog] = useState(false);
    const [saveName, setSaveName] = useState("");
    const [saveConfirmed, setSaveConfirmed] = useState(false);
    const [shareConfirmed, setShareConfirmed] = useState(false);

    // Simple ref to track latest results for display
    const resultsRef = useRef<Record<string, StepResponse[]>>({});
    resultsRef.current = results;
    // CRITICAL: Synchronous guard against rapid double-clicks
    const applyingRef = useRef(false);
    
    // AutoML polling
    const [automlJobId, setAutomlJobId] = useState<string | null>(null);
    const [automlStatus, setAutomlStatus] = useState<AutoMLStatus | null>(null);
    const automlPollRef = useRef<NodeJS.Timeout | null>(null);
    const [applyingAutoml, setApplyingAutoml] = useState(false);

    useEffect(() => {
        if (!sessionId) return;
        localStorage.setItem("ml_last_session", sessionId);
        getSession(sessionId).then(s => {
            setSession(s);
            // Get display name from datasets list
            getDatasets().then(datasets => {
                const ds = datasets.find(d => d.id === s.dataset_id);
                setDatasetName(ds?.name || s.dataset_id.replace("upload_", "Upload: ").replace(/_/g, " "));
            });
            return getProfile(s.dataset_id);
        }).then(setProfile);
        
        // Check for AutoML job
        const jobId = localStorage.getItem("ml_last_automl_job");
        if (jobId) {
            setAutomlJobId(jobId);
        }
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
        if (!session || activeStep !== 13) return;
        getRuns(session.id).then(setRuns);
    }, [activeStep, session]);
    
    // Poll AutoML status and apply when done
    useEffect(() => {
        if (!automlJobId || !session) return;
        
        const pollStatus = async () => {
            try {
                const { getAutoMLStatus, applyAutoML } = await import("@/lib/api");
                const status = await getAutoMLStatus(automlJobId);
                setAutomlStatus(status);
                
                if (status.status === "done" && !applyingAutoml) {
                    setApplyingAutoml(true);
                    try {
                        const trainResult = await applyAutoML(session.id, automlJobId);
                        localStorage.removeItem("ml_last_automl_job");
                        setAutomlJobId(null);
                        // Set the training result so PredictView has a model to use
                        setTrainResults([trainResult]);
                        // Mark tuning as complete too (use train result as tune result)
                        setTuneResult(trainResult);
                        // Navigate to final step
                        setActiveStep(14);
                    } catch (e: any) {
                        console.error("Failed to apply AutoML:", e.message);
                        alert("AutoML completed but failed to apply: " + e.message);
                    } finally {
                        setApplyingAutoml(false);
                    }
                } else if (status.status === "error") {
                    alert("AutoML failed: " + (status.error || "Unknown error"));
                    localStorage.removeItem("ml_last_automl_job");
                    setAutomlJobId(null);
                }
            } catch (e: any) {
                console.error("AutoML poll error:", e.message);
            }
        };
        
        pollStatus();
        automlPollRef.current = setInterval(pollStatus, 2000);
        
        return () => {
            if (automlPollRef.current) {
                clearInterval(automlPollRef.current);
            }
        };
    }, [automlJobId, session, applyingAutoml]);

    async function apply(stepKey: string) {
        // SYNCHRONOUS guard — prevents ANY concurrent calls regardless of React state timing
        if (applyingRef.current) return;
        applyingRef.current = true;
        
        if (!session) {
            applyingRef.current = false;
            return;
        }
        setLoading(stepKey);
        try {
            if (stepKey === "train") {
                const model = selected["train"] || "random_forest";
                const res = await trainModel(session.id, model, trainParams);
                setTrainResults(prev => [res, ...prev]);
            } else if (stepKey === "tune") {
                const model = selected["tune_model"] || "random_forest";
                const method = selected["tune_method"] || "optuna";
                const res = method === "gridsearch"
                    ? await gridSearch(session.id)
                    : await tuneModel(session.id, model);
                setTuneResult(res);
            } else {
                // Get the currently selected technique for this step
                const currentTechniques = TECHNIQUES[stepKey] || [];
                const currentSelection = selected[stepKey];
                const tech = currentSelection || currentTechniques[0]?.value;
                const techObj = currentTechniques.find(t => t.value === tech);
                const params = { ...(techObj?.params || {}) };

                // Backend now handles dedup: returns cached result if same technique was used
                const res = await runStep(session.id, stepKey, tech, params);
                const appliedTechnique = res.technique || tech;
                setSelected(s => ({ ...s, [stepKey]: appliedTechnique }));
                
                // Add to results list (backend ensures no duplicate techniques)
                setResults(prev => {
                    const existing = prev[stepKey] || [];
                    // Only add if not already in list (backend dedup + frontend safety)
                    const alreadyExists = existing.some(r => r.technique === appliedTechnique);
                    if (alreadyExists) return prev;
                    return { ...prev, [stepKey]: [res, ...existing] };
                });
                // Stay on same step - don't auto-advance
            }
        } catch (e: any) { console.error(e.message); }
        finally { 
            setLoading(null);
            applyingRef.current = false; 
        }
    }

    // Reset current step and all subsequent steps
    async function handleResetStep(stepKey: string) {
        if (!session) return;
        if (!confirm("Reset this step and all subsequent steps? This will clear all results from this point forward.")) return;
        
        setLoading(stepKey);
        try {
            // Map step keys to backend step names
            const stepNameMap: Record<string, string> = {
                missing: "missing_values",
                outliers: "outliers",
                features: "feature_engineering",
                encoding: "encoding",
                selection: "feature_selection",
                scaling: "scaling",
                pca: "pca",
            };
            
            const backendStepName = stepNameMap[stepKey];
            if (!backendStepName) return;
            
            await resetStep(session.id, backendStepName);
            
            // Clear local results for this step and all subsequent steps
            const stepOrder = ["missing", "outliers", "features", "encoding", "selection", "scaling", "pca", "train", "tune"];
            const currentIdx = stepOrder.indexOf(stepKey);
            
            setResults(prev => {
                const next = { ...prev };
                for (let i = currentIdx; i < stepOrder.length; i++) {
                    delete next[stepOrder[i]];
                }
                return next;
            });
            
            // Clear train/tune results if resetting those steps
            if (stepKey === "train") setTrainResults([]);
            if (stepKey === "tune") setTuneResult(null);
            
            // Reset selection for this step
            setSelected(prev => {
                const next = { ...prev };
                delete next[stepKey];
                return next;
            });
            
        } catch (e: any) {
            console.error("Reset failed:", e.message);
            alert("Reset failed: " + e.message);
        } finally {
            setLoading(null);
        }
    }

    function advance() {
        setActiveStep(s => Math.min(s + 1, STEPS.length));
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

    async function handleSave() {
        if (!session) return;
        setSaving(true);
        try {
            const { saveSession } = await import("@/lib/api");
            await saveSession(session.id, saveName || datasetName);
            setShowSaveDialog(false);
            setSaveName("");
            alert("Session saved successfully!");
        } catch (e: any) {
            console.error("Save failed:", e.message);
            alert("Save failed: " + e.message);
        } finally {
            setSaving(false);
        }
    }

    async function handleShare() {
        if (!session) return;
        setSharing(true);
        try {
            const { shareSession } = await import("@/lib/api");
            const result = await shareSession(session.id);
            const url = window.location.origin + result.share_url;
            setShareUrl(url);
            await navigator.clipboard.writeText(url);
            alert("Share link copied to clipboard!");
        } catch (e: any) {
            console.error("Share failed:", e.message);
            alert("Share failed: " + e.message);
        } finally {
            setSharing(false);
        }
    }

    if (!session || !profile) return (
        <div className="min-h-screen bg-slate-950 flex items-center justify-center">
            <Loader2 className="w-6 h-6 animate-spin text-blue-400" />
        </div>
    );

    const currentStep = STEPS[activeStep - 1];
    const stepResultsList = results[currentStep?.key] || [];
    const latestResult = stepResultsList[0];
    const prevResult = stepResultsList[1];
    const latestTrain = trainResults[0];
    const finalModel = tuneResult || latestTrain;
    const regModels = ["ridge", "lasso", "elasticnet", "svr"];
    
    // Show AutoML progress if running
    const showAutoMLProgress = automlJobId && automlStatus?.status === "running";

    return (
        <div className="h-screen bg-slate-950 flex flex-col overflow-hidden">

            {/* Top bar */}
            <header className="h-10 bg-slate-900 border-b border-slate-800 flex items-center px-3 gap-2 flex-shrink-0">
                <button onClick={() => router.push("/")} className="text-xs text-slate-500 hover:text-white transition-colors font-medium flex items-center gap-1">
                    <Home className="w-3.5 h-3.5" /> Home
                </button>
                {showAutoMLProgress && (
                    <>
                        <div className="w-px h-3.5 bg-slate-800" />
                        <div className="flex items-center gap-2">
                            <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-400" />
                            <span className="text-xs text-blue-400 font-medium">AutoML running... {automlStatus?.progress}%</span>
                        </div>
                    </>
                )}
                <div className="w-px h-3.5 bg-slate-800" />
                <span className="text-xs font-semibold text-white">ML Playground</span>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                <div className="ml-auto flex items-center gap-2.5">
                    <span className="text-xs text-slate-400 font-medium">{datasetName}</span>
                    <span className={`text-xs px-2 py-0.5 rounded font-medium ${session.task_type === "classification"
                            ? "bg-emerald-500/10 text-emerald-400"
                            : "bg-amber-500/10 text-amber-400"}`}>
                        {session.task_type === "classification" ? "Classification" : "Regression"}
                    </span>

                    {/* Save button */}
                    <button
                        onClick={() => {
                            navigator.clipboard.writeText(window.location.href).then(() => {
                                setSaveConfirmed(true);
                                setTimeout(() => setSaveConfirmed(false), 2000);
                            });
                        }}
                        className="text-xs px-2.5 py-1 rounded-md bg-slate-800 text-slate-400
                        border border-slate-700 hover:text-white hover:border-slate-600
                        transition-colors font-medium flex items-center gap-1">
                        {saveConfirmed
                            ? <><CheckCircle2 className="w-3 h-3 text-emerald-400" />Saved!</>
                            : <><Download className="w-3 h-3" />Save</>}
                    </button>

                    {/* Share button */}
                    <button
                        onClick={() => {
                            navigator.clipboard.writeText(window.location.href).then(() => {
                                setShareConfirmed(true);
                                setTimeout(() => setShareConfirmed(false), 2000);
                            });
                        }}
                        className="text-xs px-2.5 py-1 rounded-md bg-blue-600 text-white
                        border border-blue-500 hover:bg-blue-500 transition-colors
                        font-medium flex items-center gap-1">
                        {shareConfirmed
                            ? <><CheckCircle2 className="w-3 h-3" />Copied!</>
                            : <><Share2 className="w-3 h-3" />Share</>}
                    </button>

                    <button onClick={exportCode}
                        className="text-xs px-2.5 py-1 rounded-md bg-slate-800 text-slate-400 border border-slate-700 hover:text-white hover:border-slate-600 transition-colors font-medium flex items-center gap-1">
                        <Download className="w-3 h-3" /> Export Code
                    </button>
                </div>
            </header>

            <div className="flex flex-1 overflow-hidden">

                {/* Left sidebar */}
                <aside className="w-48 bg-slate-900 border-r border-slate-800 flex flex-col flex-shrink-0">
                    <div className="px-3 pt-4 pb-3 flex-1 overflow-y-auto">
                        <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4">
                            Lifecycle
                        </p>
                        <nav className="space-y-0.5">
                    {STEPS.map(step => {
                                // AutoML complete: all steps are done
                                const autoMLComplete = activeStep === 14 && (trainResults.length > 0 || !!tuneResult);
                                const readOnly = ["profile", "eda"].includes(step.key);
                                const done = autoMLComplete ? true
                                    : readOnly
                                        ? activeStep > step.id
                                        : step.key === "train"
                                            ? trainResults.length > 0
                                            : step.key === "tune"
                                                ? !!tuneResult
                                                : (results[step.key]?.length ?? 0) > 0;
                                const active = activeStep === step.id;

                                return (
                                    <button key={step.id}
                                        onClick={() => setActiveStep(step.id)}
                                        className={`w-full flex items-center gap-2 px-2.5 py-2 rounded-md text-xs font-medium transition-all text-left ${active
                                                ? "bg-blue-500/10 text-blue-300 border border-blue-500/20"
                                                : done
                                                    ? "text-slate-400 hover:bg-slate-800"
                                                    : "text-slate-600 hover:bg-slate-800/40"
                                            }`}>
                                        <span className="flex-shrink-0">
                                            {done
                                                ? <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                                                : active
                                                    ? <ChevronRight className="w-4 h-4 text-blue-400" />
                                                    : <Circle className="w-4 h-4 text-slate-700" />}
                                        </span>
                                        <span className="flex-1 leading-tight">{step.label}</span>
                                        <span className="text-slate-700 text-xs font-normal">{step.id}</span>
                                    </button>
                                );
                            })}
                        </nav>
                    </div>

                    {/* Pipeline summary */}
                    {Object.keys(results).length > 0 && (
                        <div className="mx-3 mb-4 p-3 rounded-xl bg-slate-800/40 border border-slate-700/40">
                            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2.5">
                                Pipeline
                            </p>
                            {Object.entries(results).map(([k, v]) => {
                                const latest = v[0];
                                return (
                                    <div key={k} className="flex items-center justify-between mb-1.5">
                                        <div className="flex items-center gap-1.5">
                                            <span className="w-1 h-1 rounded-full bg-emerald-400 flex-shrink-0" />
                                            <span className="text-xs text-slate-400">{label(k)}</span>
                                        </div>
                                        <span className="text-xs text-slate-600 truncate ml-2 max-w-20">
                                            {latest?.technique ? TECHNIQUES[k]?.find(t => t.value === latest.technique)?.label || latest.technique : ""}
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </aside>

                {/* Center */}
                <main className="flex-1 flex flex-col overflow-hidden min-w-0">
                    
                    {/* Step header */}
                    <div className="px-5 py-3 border-b border-slate-800 flex-shrink-0">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2.5">
                                <span className="text-xs font-mono text-slate-500">{activeStep}/{STEPS.length}</span>
                                <span className="text-slate-700">·</span>
                                <h1 className="text-sm font-semibold text-white">{currentStep?.label}</h1>
                            </div>
                            {activeStep < STEPS.length && (
                                <button onClick={advance}
                                    className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-300 transition-colors font-medium">
                                    Skip <ArrowRight className="w-3 h-3" />
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Main content */}
                    <div className="flex-1 overflow-y-auto p-4">

                        {activeStep === 1 && (
                            <ProfileView profile={profile} onNext={advance} />
                        )}

                        {activeStep === 2 && (
                            <EDAView
                                profile={profile}
                                distributions={distributions}
                                correlation={correlation}
                                targetData={targetData}
                                loading={edaLoading}
                                onNext={advance}
                                datasetId={session.dataset_id}
                            />
                        )}

                        {activeStep >= 3 && activeStep <= 9 && currentStep?.action && (
                            <StepView
                                step={currentStep}
                                techniques={TECHNIQUES[currentStep.key] || []}
                                selected={selected[currentStep.key] || (TECHNIQUES[currentStep.key] || [])[0]?.value || ""}
                                onSelect={v => setSelected(s => ({ ...s, [currentStep.key]: v }))}
                                onApply={() => apply(currentStep.key)}
                                loading={loading === currentStep.key}
                                latestResult={results[currentStep.key]?.[0]}
                                allResults={results[currentStep.key] || []}
                                profile={profile}
                                onNext={advance}
                                onChooseAlternative={(tech) => {
                                    setSelected(s => ({ ...s, [currentStep.key]: tech }));
                                    handleResetStep(currentStep.key);
                                }}
                                distributions={distributions}
                            />
                        )}

                        {activeStep === 10 && (
                            <TrainView
                                taskType={session.task_type}
                                selected={selected["train"] || "random_forest"}
                                onSelect={v => {
                                    setSelected(s => ({ ...s, train: v }));
                                    // Reset params when model changes
                                    const defaults: Record<string, any> = {};
                                    (MODEL_PARAMS[v] || []).forEach(p => { defaults[p.key] = p.default; });
                                    setTrainParams(defaults);
                                }}
                                onParamChange={(key, val) => setTrainParams(prev => ({ ...prev, [key]: val }))}
                                onApply={() => apply("train")}
                                loading={loading === "train"}
                                results={trainResults}
                                regModels={regModels}
                                onNext={advance}
                                currentParams={trainParams}
                            />
                        )}

                        {activeStep === 11 && (
                            <TuneView
                                taskType={session.task_type}
                                selectedModel={selected["tune_model"] || "random_forest"}
                                selectedMethod={selected["tune_method"] || "optuna"}
                                onSelectModel={v => setSelected(s => ({ ...s, tune_model: v }))}
                                onSelectMethod={v => setSelected(s => ({ ...s, tune_method: v }))}
                                onApply={() => apply("tune")}
                                loading={loading === "tune"}
                                result={tuneResult}
                                onNext={advance}
                            />
                        )}

                        {activeStep === 12 && (
                            <ExplainView
                                trainResult={tuneResult || latestTrain}
                                taskType={session.task_type}
                                onNext={advance}
                            />
                        )}

                        {activeStep === 13 && (
                            <CompareView
                                runs={runs}
                                taskType={session.task_type}
                            />
                        )}

                        {activeStep === 14 && <PredictView profile={profile} session={session} finalModel={finalModel} />}
                    </div>

                    {/* Bottom chart bar */}
                    <BottomChartBar
                        open={chartOpen}
                        setOpen={setChartOpen}
                        chartType={chartType}
                        setChartType={setChartType}
                        chartCol={chartCol}
                        setChartCol={setChartCol}
                        profile={profile}
                        distributions={distributions}
                        correlation={correlation}
                        targetData={targetData}
                        trainResult={tuneResult || latestTrain}
                    />
                </main>

                {/* Right panel */}
                <RightPanel
                    latestResult={latestResult}
                    trainResult={tuneResult || latestTrain}
                />
            </div>

            {/* Save Dialog */}
            <SaveDialog
                show={showSaveDialog}
                onClose={() => setShowSaveDialog(false)}
                onSave={handleSave}
                saving={saving}
                name={saveName}
                onNameChange={setSaveName}
            />
        </div>
    );
}

// ─── PROFILE VIEW ─────────────────────────────────────────────────────────────

function ProfileView({ profile, onNext }: { profile: DatasetProfile; onNext: () => void }) {
    return (
        <div className="max-w-3xl space-y-5">
            <div className="grid grid-cols-4 gap-3">
                {[
                    { label: "Rows", value: profile.shape.rows.toLocaleString() },
                    { label: "Columns", value: profile.shape.cols },
                    { label: "Missing Cells", value: profile.missing_summary.total_missing_cells },
                    { label: "Duplicates", value: profile.duplicate_rows },
                ].map(item => (
                    <div key={item.label} className="bg-slate-900 border border-slate-800 rounded-lg p-3">
                        <div className="text-lg font-bold text-white font-mono">{item.value}</div>
                        <div className="text-xs text-slate-500 mt-0.5 font-medium">{item.label}</div>
                    </div>
                ))}
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                <div className="px-5 py-3 border-b border-slate-800">
                    <span className="text-sm font-semibold text-white">Column Overview</span>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-slate-800">
                                {["Column", "Data Type", "Missing %", "Unique", "Notes"].map(h => (
                                    <th key={h} className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 tracking-wide border-b border-slate-800">{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {profile.columns.map(col => (
                                <tr key={col.name} className="border-b border-slate-800/50 hover:bg-slate-800/20 transition-colors">
                                    <td className="px-5 py-3 font-mono text-sm text-slate-300">
                                        {col.name}
                                        {col.is_target && (
                                            <span className="ml-2 text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-sans">target</span>
                                        )}
                                    </td>
                                    <td className="px-5 py-3">
                                        <span className={`text-xs px-2 py-1 rounded-lg font-medium ${col.type === "numeric"
                                                ? "bg-purple-500/10 text-purple-400"
                                                : "bg-orange-500/10 text-orange-400"}`}>
                                            {col.type === "numeric" ? "Numeric" : "Categorical"}
                                        </span>
                                    </td>
                                    <td className="px-5 py-3 font-mono text-sm">
                                        <span className={
                                            col.missing_pct > 20 ? "text-red-400 font-semibold"
                                                : col.missing_pct > 0 ? "text-amber-400"
                                                    : "text-emerald-400"}>
                                            {col.missing_pct}%
                                        </span>
                                    </td>
                                    <td className="px-5 py-3 text-slate-400 font-mono text-sm">{col.unique_count}</td>
                                    <td className="px-5 py-3 text-xs text-slate-500">
                                        {col.unique_count > 20 && col.type === "categorical" && "High cardinality"}
                                        {col.missing_pct > 50 && "Over 50% missing"}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            <button onClick={onNext}
                className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-500
          text-white text-sm font-semibold rounded-xl transition-colors">
                Begin Analysis <ArrowRight className="w-4 h-4" />
            </button>
        </div>
    );
}

// ─── EDA VIEW ─────────────────────────────────────────────────────────────────

function EDAView({ profile, distributions, correlation, targetData, loading, onNext, datasetId }: {
    profile: DatasetProfile;
    distributions: EDADistributions | null;
    correlation: EDACorrelation | null;
    targetData: EDATargetAnalysis | null;
    loading: boolean;
    onNext: () => void;
    datasetId: string;
}) {
    const [activeChart, setActiveChart] = useState("distribution");
    const [selCol, setSelCol] = useState("");
    const col = selCol || Object.keys(distributions || {})[0] || "";

    const chartTabs = [
        { value: "distribution", label: "Distributions" },
        { value: "correlation", label: "Correlation" },
        { value: "target", label: "Target Analysis" },
        { value: "missing", label: "Missing Values" },
        { value: "scatter3d", label: "3D Scatter" },
    ];

    if (loading) return (
        <div className="flex items-center gap-3 py-20 justify-center">
            <Loader2 className="w-5 h-5 animate-spin text-blue-400" />
            <span className="text-slate-400">Loading analysis...</span>
        </div>
    );

    return (
        <div className="max-w-3xl space-y-5">
            {/* Chart type tabs */}
            <div className="flex gap-2 border-b border-slate-800 pb-0">
                {chartTabs.map(ct => (
                    <button key={ct.value} onClick={() => setActiveChart(ct.value)}
                        className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors -mb-px ${activeChart === ct.value
                                ? "border-blue-500 text-blue-400"
                                : "border-transparent text-slate-500 hover:text-slate-300"}`}>
                        {ct.label}
                    </button>
                ))}
            </div>

            {/* Distribution */}
            {activeChart === "distribution" && distributions && (
                <div className="space-y-4">
                    <select value={col} onChange={e => setSelCol(e.target.value)}
                        className="text-sm bg-slate-900 border border-slate-700 rounded-xl
              px-4 py-2 text-slate-300 focus:outline-none focus:border-blue-500 font-medium">
                        {Object.keys(distributions).map(c => (
                            <option key={c} value={c}>{c}</option>
                        ))}
                    </select>
                    {distributions[col] && (
                        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                            <p className="text-sm font-semibold text-white mb-4">
                                {col} <span className="text-slate-500 font-normal text-xs ml-1">({distributions[col].type})</span>
                            </p>
                            {distributions[col].type === "numeric" && distributions[col].histogram && (
                                <>
                                    <ResponsiveContainer width="100%" height={180}>
                                        <BarChart data={distributions[col].histogram!.counts.map((c, i) => ({
                                            x: Number(distributions[col].histogram!.edges[i]).toFixed(1), v: c
                                        }))}>
                                            <XAxis dataKey="x" tick={{ fill: "#64748b", fontSize: 11 }} interval={4} />
                                            <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
                                            <Tooltip contentStyle={TOOLTIP_STYLE} />
                                            <Bar dataKey="v" fill="#3b82f6" radius={[3, 3, 0, 0]} opacity={0.85} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                    {distributions[col].stats && (
                                        <div className="grid grid-cols-3 gap-3 mt-4">
                                            {Object.entries(distributions[col].stats!).map(([k, v]) => (
                                                <div key={k} className="bg-slate-800 rounded-lg p-3">
                                                    <div className="text-xs text-slate-500 font-medium">{label(k)}</div>
                                                    <div className="text-sm font-mono text-white mt-1">{Number(v).toFixed(3)}</div>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </>
                            )}
                            {distributions[col].type === "categorical" && distributions[col].bar && (
                                <ResponsiveContainer width="100%" height={180}>
                                    <BarChart data={distributions[col].bar!.labels.map((l, i) => ({
                                        label: l, count: distributions[col].bar!.counts[i]
                                    }))} layout="vertical">
                                        <XAxis type="number" tick={{ fill: "#64748b", fontSize: 11 }} />
                                        <YAxis type="category" dataKey="label" tick={{ fill: "#64748b", fontSize: 11 }} width={90} />
                                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                                        <Bar dataKey="count" fill="#8b5cf6" radius={[0, 3, 3, 0]} opacity={0.85} />
                                    </BarChart>
                                </ResponsiveContainer>
                            )}
                        </div>
                    )}
                </div>
            )}

            {/* Correlation */}
            {activeChart === "correlation" && correlation && (
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                    <p className="text-sm font-semibold text-white mb-4">Correlation Matrix</p>
                    <CorrelationHeatmap correlation={correlation} />
                </div>
            )}

            {/* Target */}
            {activeChart === "target" && targetData && (
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                    <p className="text-sm font-semibold text-white mb-1">
                        Target Column: <span className="text-blue-400 font-mono">{targetData.target}</span>
                    </p>
                    {targetData.is_imbalanced && (
                        <div className="flex items-center gap-2 text-xs text-amber-400 mb-4 bg-amber-500/5 border border-amber-500/20 rounded-lg px-3 py-2">
                            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
                            Class imbalance detected (ratio: {targetData.class_balance?.toFixed(2)}). Consider using class_weight="balanced".
                        </div>
                    )}
                    {targetData.distribution && (
                        <ResponsiveContainer width="100%" height={180}>
                            <BarChart data={targetData.distribution.labels.map((l, i) => ({
                                label: String(l), count: targetData.distribution!.counts[i]
                            }))}>
                                <XAxis dataKey="label" tick={{ fill: "#64748b", fontSize: 12 }} />
                                <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
                                <Tooltip contentStyle={TOOLTIP_STYLE} />
                                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                                    {targetData.distribution.labels.map((_, i) => (
                                        <Cell key={i} fill={["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b"][i % 4]} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    )}
                </div>
            )}


            {activeChart === "scatter3d" && (
                <Scatter3DView
                    profile={profile}
                    sessionDatasetId={datasetId}
                />
            )}

            {/* Missing */}
            {activeChart === "missing" && (
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                    <p className="text-sm font-semibold text-white mb-4">Missing Values by Column</p>
                    <div className="space-y-3">
                        {profile.columns.map(col => (
                            <div key={col.name} className="flex items-center gap-4">
                                <span className="text-sm font-mono text-slate-400 w-28 truncate">{col.name}</span>
                                <div className="flex-1 bg-slate-800 rounded-full h-2">
                                    <div className="h-2 rounded-full transition-all" style={{
                                        width: `${Math.max(col.missing_pct, col.missing_pct > 0 ? 1 : 0)}%`,
                                        background: col.missing_pct > 20 ? "#ef4444"
                                            : col.missing_pct > 0 ? "#f59e0b" : "#10b981",
                                    }} />
                                </div>
                                <span className={`text-xs font-mono w-10 text-right font-semibold ${col.missing_pct > 20 ? "text-red-400"
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
                className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-500
          text-white text-sm font-semibold rounded-xl transition-colors">
                Start Preprocessing <ArrowRight className="w-4 h-4" />
            </button>
        </div>
    );
}

// ─── CORRELATION HEATMAP ──────────────────────────────────────────────────────

function CorrelationHeatmap({ correlation }: { correlation: EDACorrelation }) {
    const { columns, matrix } = correlation;
    const n = columns.length;
    const size = Math.min(Math.floor(460 / n), 40);

    function cellColor(v: number) {
        const abs = Math.abs(v);
        if (v > 0.7) return `rgba(59,130,246,${0.3 + abs * 0.7})`;
        if (v > 0.3) return `rgba(99,102,241,${0.2 + abs * 0.5})`;
        if (v > -0.3) return `rgba(51,65,85,0.3)`;
        if (v > -0.7) return `rgba(239,68,68,${0.2 + abs * 0.5})`;
        return `rgba(220,38,38,${0.3 + abs * 0.6})`;
    }

    return (
        <div className="overflow-x-auto">
            <div className="flex" style={{ marginLeft: 72 }}>
                {columns.map(c => (
                    <div key={c} style={{ width: size, fontSize: 9, color: "#64748b", textAlign: "center" }}
                        className="overflow-hidden whitespace-nowrap">
                        {c.substring(0, 5)}
                    </div>
                ))}
            </div>
            {matrix.map((row, i) => (
                <div key={i} className="flex items-center">
                    <div style={{ width: 72, fontSize: 10, color: "#64748b" }} className="truncate pr-2 text-right flex-shrink-0">
                        {columns[i]}
                    </div>
                    {row.map((val, j) => (
                        <div key={j}
                            style={{ width: size, height: size, background: cellColor(val) }}
                            title={`${columns[i]} × ${columns[j]}: ${val.toFixed(2)}`}
                            className="border border-slate-900/30"
                        />
                    ))}
                </div>
            ))}
            <div className="flex items-center gap-4 mt-3" style={{ marginLeft: 72 }}>
                {[
                    { color: "rgba(59,130,246,0.9)", label: "Strong positive" },
                    { color: "rgba(51,65,85,0.5)", label: "No correlation" },
                    { color: "rgba(220,38,38,0.8)", label: "Strong negative" },
                ].map(({ color, label: lbl }) => (
                    <div key={lbl} className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded" style={{ background: color }} />
                        <span className="text-xs text-slate-500">{lbl}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function Scatter3DView({ profile, sessionDatasetId }: {
    profile: DatasetProfile;
    sessionDatasetId: string;
}) {
    const numericCols = profile.columns
        .filter(c => c.type === "numeric" && !c.is_target)
        .map(c => c.name);

    const [xCol, setXCol] = useState(numericCols[0] || "");
    const [yCol, setYCol] = useState(numericCols[1] || "");
    const [zCol, setZCol] = useState(numericCols[2] || "");
    const [colorCol, setColorCol] = useState("");
    const [sampleData, setSampleData] = useState<Record<string, number>[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (!sessionDatasetId) return;
        setLoading(true);
        import("@/lib/api").then(({ getDatasetSample }) =>
            getDatasetSample(sessionDatasetId, 300)
        ).then(res => {
            setSampleData(res.rows);
        }).catch(console.error)
            .finally(() => setLoading(false));
    }, [sessionDatasetId]);

    if (numericCols.length < 2) return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 text-center">
            <p className="text-slate-500 text-sm">Need at least 2 numeric columns for scatter plot.</p>
        </div>
    );

    // Normalize Z for bubble size (4px to 20px)
    const zValues = sampleData.map(r => Number(r[zCol] ?? 0)).filter(v => !isNaN(v));
    const zMin = Math.min(...zValues);
    const zMax = Math.max(...zValues);
    const zRange = zMax - zMin || 1;

    const colorScale = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#ec4899"];
    const colorValues = colorCol
        ? sampleData.map(r => Number(r[colorCol] ?? 0)).filter(v => !isNaN(v))
        : [];
    const colorMin = Math.min(...colorValues);
    const colorRange = (Math.max(...colorValues) - colorMin) || 1;

    const plotData = sampleData
        .filter(r => r[xCol] !== undefined && r[yCol] !== undefined)
        .map(r => ({
            x: Number(r[xCol]),
            y: Number(r[yCol]),
            z: zCol ? Number(r[zCol] ?? 0) : null,
            c: colorCol ? Number(r[colorCol] ?? 0) : null,
        }));

    return (
        <div className="space-y-4">
            {/* Column selectors */}
            <div className="grid grid-cols-4 gap-3">
                {[
                    { label: "X Axis", val: xCol, set: setXCol, opts: numericCols, req: true },
                    { label: "Y Axis", val: yCol, set: setYCol, opts: numericCols, req: true },
                    { label: "Size (Z)", val: zCol, set: setZCol, opts: numericCols, req: false },
                    { label: "Colour", val: colorCol, set: setColorCol, opts: numericCols, req: false },
                ].map(({ label: lbl, val, set, opts, req }) => (
                    <div key={lbl}>
                        <label className="text-xs text-slate-500 font-medium block mb-1">{lbl}</label>
                        <select
                            value={val}
                            onChange={e => set(e.target.value)}
                            className="w-full text-xs bg-slate-900 border border-slate-800 rounded-lg
                px-2.5 py-1.5 text-slate-300 focus:outline-none focus:border-blue-500">
                            {!req && <option value="">None</option>}
                            {opts.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                    </div>
                ))}
            </div>

            {/* Chart */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
                {loading ? (
                    <div className="flex items-center justify-center h-64 gap-2 text-slate-500">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span className="text-sm">Loading real data...</span>
                    </div>
                ) : (
                    <>
                        <div className="flex items-center justify-between mb-3">
                            <p className="text-xs font-semibold text-slate-400">
                                {xCol} vs {yCol}
                                {zCol && ` · size = ${zCol}`}
                                {colorCol && ` · colour = ${colorCol}`}
                            </p>
                            <span className="text-xs text-slate-600">{plotData.length} real data points</span>
                        </div>

                        <ResponsiveContainer width="100%" height={320}>
                            <ScatterChart margin={{ top: 10, right: 20, bottom: 40, left: 20 }}>
                                <XAxis
                                    dataKey="x"
                                    type="number"
                                    name={xCol}
                                    tick={{ fill: "#64748b", fontSize: 10 }}
                                    label={{ value: xCol, position: "insideBottom", offset: -10, fill: "#64748b", fontSize: 11 }}
                                />
                                <YAxis
                                    dataKey="y"
                                    type="number"
                                    name={yCol}
                                    tick={{ fill: "#64748b", fontSize: 10 }}
                                    label={{ value: yCol, angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }}
                                />
                                <Tooltip
                                    contentStyle={TOOLTIP_STYLE}
                                    formatter={(v: any, name: any) => [Number(v).toFixed(3), String(name)]}
                                    cursor={{ strokeDasharray: "3 3" }}
                                />
                                <Scatter data={plotData} shape={(props: any) => {
                                    const { cx, cy, payload } = props;
                                    // Proper normalised bubble size
                                    const r = zCol && payload.z !== null
                                        ? 4 + ((payload.z - zMin) / zRange) * 16
                                        : 6;
                                    // Proper normalised colour
                                    const fill = colorCol && payload.c !== null
                                        ? colorScale[Math.floor(((payload.c - colorMin) / colorRange) * (colorScale.length - 1))]
                                        : "#3b82f6";
                                    return (
                                        <circle
                                            cx={cx} cy={cy} r={r}
                                            fill={fill}
                                            fillOpacity={0.6}
                                            stroke={fill}
                                            strokeWidth={1}
                                        />
                                    );
                                }} />
                            </ScatterChart>
                        </ResponsiveContainer>

                        <div className="flex items-center justify-center gap-6 mt-2">
                            <div className="flex items-center gap-1.5">
                                <div className="w-2.5 h-2.5 rounded-full bg-blue-500 opacity-60" />
                                <span className="text-xs text-slate-600">Real sample rows — not synthetic</span>
                            </div>
                            {zCol && (
                                <span className="text-xs text-slate-600">
                                    Bubble size ∝ {zCol} ({zMin.toFixed(1)} – {zMax.toFixed(1)})
                                </span>
                            )}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}

// ─── STEP VIEW ────────────────────────────────────────────────────────────────

function StepView({ step, techniques, selected, onSelect, onApply, loading,
    latestResult, prevResult, profile, onNext, allResults = [], onChooseAlternative, distributions }: {
        step: { id: number; key: string; label: string };
        techniques: { value: string; label: string; desc: string }[];
        selected: string;
        onSelect: (v: string) => void;
        onApply: () => void;
        loading: boolean;
        latestResult?: StepResponse;
        prevResult?: StepResponse;
        profile: DatasetProfile;
        onNext: () => void;
        allResults?: StepResponse[];
        onChooseAlternative?: (tech: string) => void;
        distributions?: EDADistributions | null;
    }) {
    const selectedTech = techniques.find(t => t.value === selected);

    return (
        <div className="max-w-4xl space-y-5">

            {/* Technique grid */}
            <div className="grid grid-cols-2 gap-2">
                {techniques.map(t => (
                    <button key={t.value} onClick={() => onSelect(t.value)}
                        className={`text-left px-3.5 py-3 rounded-lg border transition-all ${selected === t.value
                                ? "bg-blue-500/8 border-blue-500/50"
                                : "bg-slate-900 border-slate-800 hover:border-slate-700"}`}>
                        <div className={`text-xs font-semibold ${selected === t.value ? "text-blue-300" : "text-slate-200"}`}>
                            {t.label}
                        </div>
                        <div className="text-xs text-slate-500 mt-0.5 leading-relaxed">{t.desc}</div>
                    </button>
                ))}
            </div>

            {/* Live previews — show effect of selected technique before applying */}
            {step.key === "missing" && (
                <BeforeAfterMissing profile={profile} technique={selected} />
            )}
            {step.key === "outliers" && (
                <BeforeAfterOutliers profile={profile} technique={selected} />
            )}
            {step.key === "features" && (
                <BeforeAfterFeatures profile={profile} technique={selected} />
            )}
            {step.key === "encoding" && (
                <BeforeAfterEncoding profile={profile} technique={selected} />
            )}
            {step.key === "selection" && (
                <BeforeAfterSelection profile={profile} technique={selected} />
            )}
            {step.key === "pca" && (
                <BeforeAfterPCA profile={profile} technique={selected} />
            )}
            {step.key === "scaling" && (
                <BeforeAfterScaling profile={profile} technique={selected} />
            )}

            {/* Action buttons */}
            <div className="flex items-center gap-3">
                <button onClick={onApply} disabled={!!loading}
                    className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-500
            disabled:opacity-40 text-white text-sm font-semibold rounded-xl transition-colors">
                    {loading
                        ? <><Loader2 className="w-4 h-4 animate-spin" />Applying...</>
                        : `Apply ${selectedTech?.label ?? selected}`}
                </button>
                {latestResult && (
                    <button onClick={onNext}
                        className="flex items-center gap-2 px-4 py-2.5 text-emerald-400
              hover:text-emerald-300 border border-emerald-500/30 hover:border-emerald-500/50
              text-sm font-medium rounded-xl transition-colors">
                        Next Step <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                )}
            </div>

            {/* Results — show all alternatives in grid */}
            {allResults.length > 0 && (
                <div className="space-y-3">
                    <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                        All Alternatives ({allResults.length})
                    </p>
                    <div className={`grid gap-3 ${allResults.length === 1 ? "grid-cols-1" : allResults.length === 2 ? "grid-cols-2" : "grid-cols-3"}`}>
                        {allResults.slice(0, 9).map((result, idx) => (
                            <ResultCard 
                                key={idx} 
                                result={result} 
                                label={idx === 0 ? "Current" : `Alternative ${idx}`} 
                                highlight={idx === 0}
                                onChooseAlternative={() => {
                                    if (idx > 0 && onChooseAlternative) {
                                        onChooseAlternative(result.technique);
                                    }
                                }}
                            />
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

// ─── SAVE DIALOG MODAL ────────────────────────────────────────────────────────

function SaveDialog({ show, onClose, onSave, saving, name, onNameChange }: {
    show: boolean;
    onClose: () => void;
    onSave: () => void;
    saving: boolean;
    name: string;
    onNameChange: (v: string) => void;
}) {
    if (!show) return null;
    return (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
            <div className="bg-slate-900 border border-slate-700 rounded-2xl p-6 w-full max-w-md shadow-2xl" onClick={e => e.stopPropagation()}>
                <h3 className="text-lg font-semibold text-white mb-2">Save Session</h3>
                <p className="text-xs text-slate-500 mb-4">Give your ML pipeline a name to save it for later.</p>
                <input
                    type="text"
                    value={name}
                    onChange={e => onNameChange(e.target.value)}
                    placeholder="My ML Pipeline"
                    className="w-full text-sm bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-500 mb-4"
                    autoFocus
                />
                <div className="flex items-center gap-3 justify-end">
                    <button onClick={onClose} className="px-4 py-2 text-xs text-slate-400 hover:text-white transition-colors font-medium">Cancel</button>
                    <button onClick={onSave} disabled={saving || !name.trim()} className="px-5 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white text-sm font-semibold rounded-lg transition-colors">
                        {saving ? <><Loader2 className="w-3.5 h-3.5 animate-spin inline mr-1" />Saving...</> : "Save Session"}
                    </button>
                </div>
            </div>
        </div>
    );
}

// ─── RESULT CARD ──────────────────────────────────────────────────────────────

function ResultCard({ result, label: cardLabel, highlight, onChooseAlternative }: {
    result: StepResponse;
    label: string;
    highlight?: boolean;
    onChooseAlternative?: () => void;
}) {
    const techLabel = Object.values(TECHNIQUES).flat().find(t => t.value === result.technique)?.label || result.technique;

    return (
        <div className={`rounded-xl border ${highlight ? "border-blue-500/30 bg-blue-500/5" : "border-slate-800 bg-slate-900"}`}>
            <div className={`px-4 py-3 border-b flex items-center justify-between ${highlight ? "border-blue-500/20" : "border-slate-800"}`}>
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{cardLabel}</span>
                <span className={`text-xs px-2.5 py-1 rounded-lg font-medium ${highlight ? "bg-blue-500/10 text-blue-300" : "bg-slate-800 text-slate-400"}`}>
                    {techLabel}
                </span>
            </div>

            {/* AI Explanation */}
            {result.ai_explanation && (
                <div className="px-4 py-3 border-b border-slate-800/50">
                    <div className="flex items-center gap-1.5 mb-2">
                        <Sparkles className="w-3.5 h-3.5 text-blue-400" />
                        <span className="text-xs font-semibold text-slate-400">AI Explanation</span>
                    </div>
                    <p className="text-sm text-slate-300 leading-relaxed">{result.ai_explanation}</p>
                </div>
            )}

            {/* AI Recommendation */}
            {result.ai_recommendation && (
                <div className="px-4 py-3 border-b border-slate-800/50 bg-blue-500/5">
                    <div className="flex items-center gap-1.5 mb-2">
                        <ArrowRight className="w-3.5 h-3.5 text-blue-400" />
                        <span className="text-xs font-semibold text-blue-400">Recommendation</span>
                    </div>
                    <p className="text-sm text-slate-300 leading-relaxed">{result.ai_recommendation}</p>
                </div>
            )}

            {/* Warnings */}
            {result.warnings?.length > 0 && (
                <div className="px-4 py-3 border-b border-slate-800/50 space-y-1.5">
                    {result.warnings.map((w, i) => (
                        <div key={i} className="flex gap-2 text-xs text-amber-400/90">
                            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                            <span>{w}</span>
                        </div>
                    ))}
                </div>
            )}

            {/* Stats */}
            <div className="px-4 py-3">
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2.5">Statistics</p>
                <div className="grid grid-cols-2 gap-2">
                    {Object.entries(result.stats)
                        .filter(([k]) => STATS_KEYS.includes(k))
                        .map(([k, v]) => (
                            <div key={k} className="bg-slate-800/60 rounded-lg p-2.5">
                                <div className="text-xs text-slate-500">{label(k)}</div>
                                <div className="text-sm font-mono text-white font-semibold mt-0.5">{String(v)}</div>
                            </div>
                        ))}
                </div>
            </div>
            
            {/* Choose Alternative button */}
            {!highlight && onChooseAlternative && (
                <div className="px-4 py-3 border-t border-slate-800/50">
                    <button 
                        onClick={onChooseAlternative}
                        className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-medium rounded-lg transition-colors"
                    >
                        <RotateCcw className="w-3.5 h-3.5" />
                        Choose This Alternative
                    </button>
                </div>
            )}
        </div>
    );
}

// ─── BEFORE / AFTER MISSING ───────────────────────────────────────────────────

function BeforeAfterMissing({ profile, technique }: { profile: DatasetProfile; technique: string }) {
    const missingCols = profile.columns.filter(c => c.missing_count > 0).slice(0, 2);
    if (!missingCols.length) return null;

    const techLabel = TECHNIQUES.missing.find(t => t.value === technique)?.label || technique;

    function afterLabel(col: typeof missingCols[0]) {
        if (technique === "drop_rows") return "Row removed";
        if (technique === "drop_cols") return "Column dropped";
        if (technique === "mean") return col.stats?.mean ? `${Number(col.stats.mean).toFixed(2)} (mean)` : "filled";
        if (technique === "median") return col.stats?.median ? `${Number(col.stats.median).toFixed(2)} (median)` : "filled";
        if (technique === "constant") return "0 (constant)";
        if (technique === "knn") return "KNN estimate";
        if (technique === "mice") return "MICE estimate";
        if (technique === "random_sample") return "Random sample";
        if (technique === "missing_indicator") return "0 (median) + flag";
        return "filled";
    }

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <p className="text-sm font-semibold text-white mb-1">
                Preview — Effect of {techLabel}
            </p>
            <p className="text-xs text-slate-500 mb-4">Showing columns with missing values</p>
            <div className="space-y-5">
                {missingCols.map(col => (
                    <div key={col.name}>
                        <p className="text-sm font-mono text-slate-300 mb-2">
                            {col.name}
                            <span className="text-slate-500 font-sans text-xs ml-2">{col.missing_count} missing values</span>
                        </p>
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <p className="text-xs font-semibold text-slate-500 mb-2 uppercase">Before</p>
                                <div className="space-y-1.5">
                                    {["29.0", "NULL", "38.0", "NULL"].map((v, i) => (
                                        <div key={i} className={`px-3 py-2 rounded-lg text-sm font-mono ${v === "NULL"
                                                ? "bg-red-500/10 text-red-400 border border-red-500/20"
                                                : "bg-slate-800 text-slate-300"}`}>
                                            {v}
                                        </div>
                                    ))}
                                </div>
                            </div>
                            <div>
                                <p className="text-xs font-semibold text-slate-500 mb-2 uppercase">After</p>
                                <div className="space-y-1.5">
                                    {["29.0", afterLabel(col), "38.0", afterLabel(col)].map((v, i) => (
                                        <div key={i} className={`px-3 py-2 rounded-lg text-sm font-mono ${i % 2 === 1
                                                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                                                : "bg-slate-800 text-slate-300"}`}>
                                            {v}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

// ─── BEFORE / AFTER OUTLIERS ──────────────────────────────────────────────────

function BeforeAfterOutliers({ profile, technique }: { profile: DatasetProfile; technique: string }) {
    const numericCols = profile.columns.filter(c => c.type === "numeric" && c.stats).slice(0, 2);
    if (!numericCols.length) return null;

    const techLabel = TECHNIQUES.outliers.find(t => t.value === technique)?.label || technique;

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <p className="text-sm font-semibold text-white mb-1">Preview — Effect of {techLabel}</p>
            <p className="text-xs text-slate-500 mb-4">Showing numeric columns that may contain outliers</p>
            <div className="space-y-4">
                {numericCols.map(col => {
                    const mean = col.stats?.mean ?? 0;
                    const std = col.stats?.std ?? 1;
                    const outlierVal = (mean + 3 * std).toFixed(2);
                    const cappedVal = (mean + 1.5 * std).toFixed(2);
                    return (
                        <div key={col.name}>
                            <p className="text-sm font-mono text-slate-300 mb-2">
                                {col.name}
                                <span className="text-slate-500 font-sans text-xs ml-2">μ={mean.toFixed(1)}, σ={std.toFixed(1)}</span>
                            </p>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <p className="text-xs font-semibold text-slate-500 mb-2 uppercase">Before (extreme value)</p>
                                    <div className="space-y-1.5">
                                        <div className="px-3 py-2 rounded-lg text-sm font-mono bg-slate-800 text-slate-300">{mean.toFixed(2)}</div>
                                        <div className="px-3 py-2 rounded-lg text-sm font-mono bg-red-500/10 text-red-400 border border-red-500/20">{outlierVal} ← outlier</div>
                                        <div className="px-3 py-2 rounded-lg text-sm font-mono bg-slate-800 text-slate-300">{mean.toFixed(2)}</div>
                                    </div>
                                </div>
                                <div>
                                    <p className="text-xs font-semibold text-slate-500 mb-2 uppercase">After treatment</p>
                                    <div className="space-y-1.5">
                                        <div className="px-3 py-2 rounded-lg text-sm font-mono bg-slate-800 text-slate-300">{mean.toFixed(2)}</div>
                                        <div className="px-3 py-2 rounded-lg text-sm font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">{cappedVal} ← capped</div>
                                        <div className="px-3 py-2 rounded-lg text-sm font-mono bg-slate-800 text-slate-300">{mean.toFixed(2)}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
            {technique === "keep" && (
                <div className="mt-3 text-xs text-amber-400 bg-amber-500/5 border border-amber-500/20 rounded-lg px-3 py-2">
                    No transformation applied. Outliers will be documented in statistics only.
                </div>
            )}
        </div>
    );
}

// ─── BEFORE / AFTER ENCODING ──────────────────────────────────────────────────

function BeforeAfterEncoding({ profile, technique }: { profile: DatasetProfile; technique: string }) {
    const catCols = profile.columns.filter(c => c.type === "categorical").slice(0, 2);
    if (!catCols.length) return null;

    const techLabel = TECHNIQUES.encoding.find(t => t.value === technique)?.label || technique;
    const sampleVals = catCols.map(c => c.unique_count > 0 ? ["A", "B", "C"].slice(0, Math.min(3, c.unique_count)) : ["A", "B", "C"]);

    function afterLabel(tech: string, val: string, idx: number) {
        if (tech === "onehot") return `[${val}=1]`;
        if (tech === "label") return String(idx);
        if (tech === "ordinal") return String(idx + 1);
        if (tech === "frequency") return "0.35";
        if (tech === "target") return "0.72";
        return val;
    }

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <p className="text-sm font-semibold text-white mb-1">Preview — Effect of {techLabel}</p>
            <p className="text-xs text-slate-500 mb-4">Showing categorical columns</p>
            <div className="space-y-4">
                {catCols.map((col, i) => (
                    <div key={col.name}>
                        <p className="text-sm font-mono text-slate-300 mb-2">
                            {col.name}
                            <span className="text-slate-500 font-sans text-xs ml-2">{col.unique_count} categories</span>
                        </p>
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <p className="text-xs font-semibold text-slate-500 mb-2 uppercase">Before (raw values)</p>
                                <div className="space-y-1.5">
                                    {sampleVals[i].map((v, j) => (
                                        <div key={j} className="px-3 py-2 rounded-lg text-sm font-mono bg-slate-800 text-slate-300">{v}</div>
                                    ))}
                                </div>
                            </div>
                            <div>
                                <p className="text-xs font-semibold text-slate-500 mb-2 uppercase">After ({techLabel})</p>
                                <div className="space-y-1.5">
                                    {sampleVals[i].map((v, j) => (
                                        <div key={j} className="px-3 py-2 rounded-lg text-sm font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                                            {afterLabel(technique, v, j)}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
            {technique === "onehot" && (
                <div className="mt-3 text-xs text-blue-400 bg-blue-500/5 border border-blue-500/20 rounded-lg px-3 py-2">
                    One-Hot Encoding will create {catCols.reduce((sum, c) => sum + c.unique_count, 0)} new binary columns.
                </div>
            )}
        </div>
    );
}

// ─── BEFORE / AFTER FEATURE SELECTION ─────────────────────────────────────────

function BeforeAfterSelection({ profile, technique }: { profile: DatasetProfile; technique: string }) {
    const allCols = profile.columns.filter(c => !c.is_target);
    const nAll = allCols.length;
    const techLabel = TECHNIQUES.selection.find(t => t.value === technique)?.label || technique;

    let nKept = nAll;
    if (technique === "variance_threshold") nKept = Math.max(1, Math.floor(nAll * 0.8));
    if (technique === "correlation") nKept = Math.max(1, Math.floor(nAll * 0.7));
    if (technique === "mutual_info") nKept = Math.min(10, nAll);
    const nDropped = nAll - nKept;

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <p className="text-sm font-semibold text-white mb-1">Preview — Effect of {techLabel}</p>
            <p className="text-xs text-slate-500 mb-4">Feature selection impact</p>
            <div className="grid grid-cols-3 gap-3">
                <div className="bg-slate-800 rounded-lg p-3 text-center">
                    <div className="text-lg font-bold font-mono text-white">{nAll}</div>
                    <div className="text-xs text-slate-500">Total Features</div>
                </div>
                <div className="bg-emerald-500/10 rounded-lg p-3 text-center border border-emerald-500/20">
                    <div className="text-lg font-bold font-mono text-emerald-400">{nKept}</div>
                    <div className="text-xs text-emerald-400/70">Will Keep</div>
                </div>
                <div className="bg-red-500/10 rounded-lg p-3 text-center border border-red-500/20">
                    <div className="text-lg font-bold font-mono text-red-400">{nDropped}</div>
                    <div className="text-xs text-red-400/70">Will Drop</div>
                </div>
            </div>
            {technique === "none" && (
                <div className="mt-3 text-xs text-slate-400 bg-slate-800/50 rounded-lg px-3 py-2">
                    All {nAll} features will be passed to the model.
                </div>
            )}
        </div>
    );
}

// ─── BEFORE / AFTER PCA ────────────────────────────────────────────────────────

function BeforeAfterPCA({ profile, technique }: { profile: DatasetProfile; technique: string }) {
    const numericCols = profile.columns.filter(c => c.type === "numeric" && !c.is_target);
    const nFeatures = numericCols.length;
    const techLabel = TECHNIQUES.pca.find(t => t.value === technique)?.label || technique;

    let nComponents = nFeatures;
    if (technique === "pca_auto") nComponents = Math.max(2, Math.floor(nFeatures * 0.7));
    if (technique === "pca_fixed") nComponents = Math.min(5, nFeatures);
    const varianceRetained = technique === "none" ? 100 : Math.round((nComponents / nFeatures) * 95);

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <p className="text-sm font-semibold text-white mb-1">Preview — Effect of {techLabel}</p>
            <p className="text-xs text-slate-500 mb-4">Dimensionality reduction impact</p>
            <div className="grid grid-cols-3 gap-3">
                <div className="bg-slate-800 rounded-lg p-3 text-center">
                    <div className="text-lg font-bold font-mono text-white">{nFeatures}</div>
                    <div className="text-xs text-slate-500">Current Features</div>
                </div>
                <div className="bg-violet-500/10 rounded-lg p-3 text-center border border-violet-500/20">
                    <div className="text-lg font-bold font-mono text-violet-400">{nComponents}</div>
                    <div className="text-xs text-violet-400/70">PCA Components</div>
                </div>
                <div className="bg-blue-500/10 rounded-lg p-3 text-center border border-blue-500/20">
                    <div className="text-lg font-bold font-mono text-blue-400">{varianceRetained}%</div>
                    <div className="text-xs text-blue-400/70">Variance Retained</div>
                </div>
            </div>
            {technique === "none" && (
                <div className="mt-3 text-xs text-slate-400 bg-slate-800/50 rounded-lg px-3 py-2">
                    No dimensionality reduction. All {nFeatures} features retained.
                </div>
            )}
        </div>
    );
}

// ─── BEFORE / AFTER SCALING ────────────────────────────────────────────────────

function BeforeAfterScaling({ profile, technique }: { profile: DatasetProfile; technique: string }) {
    const numericCols = profile.columns.filter(c => c.type === "numeric" && c.stats).slice(0, 2);
    if (!numericCols.length) return null;

    const techLabel = TECHNIQUES.scaling.find(t => t.value === technique)?.label || technique;

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <p className="text-sm font-semibold text-white mb-1">Preview — Effect of {techLabel}</p>
            <p className="text-xs text-slate-500 mb-4">Value ranges before and after scaling</p>
            <div className="space-y-4">
                {numericCols.map(col => {
                    const mean = col.stats?.mean ?? 0;
                    const std = col.stats?.std ?? 1;
                    const min = col.stats?.min ?? 0;
                    const max = col.stats?.max ?? 1;
                    let afterMin = min, afterMax = max, afterMean = mean;
                    if (technique === "standard") { afterMean = 0; afterMin = min - mean; afterMax = max - mean; }
                    if (technique === "minmax") { afterMin = 0; afterMax = 1; afterMean = ((mean - min) / (max - min)); }
                    if (technique === "robust") { afterMean = 0; }
                    if (technique === "none") { afterMin = min; afterMax = max; afterMean = mean; }
                    return (
                        <div key={col.name}>
                            <p className="text-sm font-mono text-slate-300 mb-2">{col.name}</p>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <p className="text-xs font-semibold text-slate-500 mb-2 uppercase">Before</p>
                                    <div className="space-y-1.5">
                                        <div className="px-3 py-2 rounded-lg text-sm font-mono bg-slate-800 text-slate-300">
                                            range: [{min.toFixed(2)}, {max.toFixed(2)}]
                                        </div>
                                        <div className="px-3 py-2 rounded-lg text-sm font-mono bg-slate-800 text-slate-300">
                                            μ={mean.toFixed(2)}, σ={std.toFixed(2)}
                                        </div>
                                    </div>
                                </div>
                                <div>
                                    <p className="text-xs font-semibold text-slate-500 mb-2 uppercase">After</p>
                                    <div className="space-y-1.5">
                                        <div className="px-3 py-2 rounded-lg text-sm font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                                            range: [{afterMin.toFixed(2)}, {afterMax.toFixed(2)}]
                                        </div>
                                        <div className="px-3 py-2 rounded-lg text-sm font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                                            μ={afterMean.toFixed(2)}, σ≈1
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
            {technique === "none" && (
                <div className="mt-3 text-xs text-slate-400 bg-slate-800/50 rounded-lg px-3 py-2">
                    No scaling applied. Tree models (RF, XGBoost) don't require scaling.
                </div>
            )}
        </div>
    );
}

// ─── BEFORE / AFTER FEATURE ENGINEERING ────────────────────────────────────────

function BeforeAfterFeatures({ profile, technique }: { profile: DatasetProfile; technique: string }) {
    const numericCols = profile.columns.filter(c => c.type === "numeric" && !c.is_target);
    const nCurrent = numericCols.length;
    const techLabel = TECHNIQUES.features.find(t => t.value === technique)?.label || technique;

    let nNew = 0;
    if (technique === "polynomial") nNew = Math.floor(nCurrent * 1.5);
    else if (technique === "interaction") nNew = Math.floor(nCurrent * (nCurrent - 1) / 2);
    else if (technique === "log_features") nNew = nCurrent;
    else if (technique === "sqrt_features") nNew = nCurrent;
    else if (technique === "reciprocal") nNew = nCurrent;
    else if (technique === "ratio") nNew = Math.floor(nCurrent / 2);
    else if (technique === "binning") nNew = nCurrent * 5;
    else nNew = 0;

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <p className="text-sm font-semibold text-white mb-1">Preview — Effect of {techLabel}</p>
            <p className="text-xs text-slate-500 mb-4">Feature generation impact</p>
            <div className="grid grid-cols-3 gap-3">
                <div className="bg-slate-800 rounded-lg p-3 text-center">
                    <div className="text-lg font-bold font-mono text-white">{nCurrent}</div>
                    <div className="text-xs text-slate-500">Current Features</div>
                </div>
                <div className="bg-blue-500/10 rounded-lg p-3 text-center border border-blue-500/20">
                    <div className="text-lg font-bold font-mono text-blue-400">+{nNew}</div>
                    <div className="text-xs text-blue-400/70">New Features</div>
                </div>
                <div className="bg-violet-500/10 rounded-lg p-3 text-center border border-violet-500/20">
                    <div className="text-lg font-bold font-mono text-violet-400">{nCurrent + nNew}</div>
                    <div className="text-xs text-violet-400/70">Total After</div>
                </div>
            </div>
            {technique === "none" && (
                <div className="mt-3 text-xs text-slate-400 bg-slate-800/50 rounded-lg px-3 py-2">
                    No feature engineering. Proceeding with {nCurrent} existing features.
                </div>
            )}
        </div>
    );
}

// ─── TRAIN VIEW ───────────────────────────────────────────────────────────────

function TrainView({ taskType, selected, onSelect, onParamChange, onApply, loading,
    results, regModels, onNext, currentParams }: {
        taskType: string;
        selected: string;
        onSelect: (v: string) => void;
        onParamChange: (key: string, val: any) => void;
        onApply: () => void;
        loading: boolean;
        results: TrainResponse[];
        regModels: string[];
        onNext: () => void;
        currentParams: Record<string, any>;
    }) {
    const models = TECHNIQUES.train.filter(m =>
        taskType === "regression"
            ? regModels.includes(m.value) || ["random_forest", "xgboost", "lightgbm", "knn"].includes(m.value)
            : !regModels.includes(m.value)
    );

    const paramDefs = MODEL_PARAMS[selected] || [];
    const [hoveredParam, setHoveredParam] = useState<string | null>(null);
    const latest = results[0];
    const prev = results[1];

    return (
        <div className="max-w-4xl space-y-5">

            {/* Model selector */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {models.map(m => (
                    <button key={m.value} onClick={() => onSelect(m.value)}
                        className={`text-left px-4 py-3.5 rounded-xl border transition-all ${selected === m.value
                                ? "bg-blue-600/10 border-blue-500/40 ring-1 ring-blue-500/20"
                                : "bg-slate-900 border-slate-800 hover:border-slate-700 hover:bg-slate-800/60"}`}>
                        <div className={`text-sm font-semibold ${selected === m.value ? "text-blue-300" : "text-slate-200"}`}>
                            {m.label}
                        </div>
                        <div className="text-xs text-slate-500 mt-1">{m.desc}</div>
                    </button>
                ))}
            </div>

            {/* Parameter panel */}
            {paramDefs.length > 0 && (
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                    <div className="flex items-center justify-between mb-4">
                        <p className="text-sm font-semibold text-white">Hyperparameters</p>
                        <span className="text-xs text-slate-500">Hover a parameter to see its effect</span>
                    </div>
                    <div className="space-y-5">
                        {paramDefs.map(param => {
                            const val = currentParams[param.key] ?? param.default;
                            return (
                                <div key={param.key}
                                    onMouseEnter={() => setHoveredParam(param.key)}
                                    onMouseLeave={() => setHoveredParam(null)}>
                                    <div className="flex items-center justify-between mb-2">
                                        <label className="text-sm font-medium text-slate-300">{param.label}</label>
                                        <div className="flex items-center gap-3">
                                            {hoveredParam === param.key && (
                                                <span className="text-xs text-blue-400 italic">{param.effect}</span>
                                            )}
                                            <span className="text-sm font-mono font-bold text-white min-w-12 text-right">
                                                {val}
                                            </span>
                                        </div>
                                    </div>
                                    {param.type === "select" ? (
                                        <select
                                            value={String(val)}
                                            onChange={e => onParamChange(param.key, e.target.value)}
                                            className="w-full text-sm bg-slate-800 border border-slate-700 rounded-lg
                        px-3 py-2 text-white focus:outline-none focus:border-blue-500">
                                            {param.options?.map(o => <option key={o} value={o}>{o}</option>)}
                                        </select>
                                    ) : (
                                        <input
                                            type="range"
                                            min={param.min}
                                            max={param.max}
                                            step={param.step}
                                            value={Number(val)}
                                            onChange={e => onParamChange(
                                                param.key,
                                                param.type === "int" ? parseInt(e.target.value) : parseFloat(e.target.value)
                                            )}
                                            className="w-full h-2 bg-slate-700 rounded-full appearance-none cursor-pointer accent-blue-500"
                                        />
                                    )}
                                    <div className="flex justify-between mt-1">
                                        <span className="text-xs text-slate-600">{param.min}</span>
                                        <span className="text-xs text-slate-600">{param.max}</span>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Train button */}
            <div className="flex items-center gap-3">
                <button onClick={onApply} disabled={!!loading}
                    className="flex items-center gap-2 px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500
            disabled:opacity-40 text-white text-sm font-semibold rounded-xl transition-colors">
                    {loading
                        ? <><Loader2 className="w-4 h-4 animate-spin" />Training...</>
                        : `Train ${TECHNIQUES.train.find(m => m.value === selected)?.label ?? selected}`}
                </button>
                {latest && (
                    <button onClick={onNext}
                        className="flex items-center gap-2 px-4 py-2.5 text-emerald-400
              hover:text-emerald-300 border border-emerald-500/30
              text-sm font-medium rounded-xl transition-colors ml-auto">
                        Next: Tuning <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                )}
            </div>

            {/* Side by side results */}
            {latest && (
                <div className={`grid gap-4 ${prev ? "grid-cols-2" : "grid-cols-1"}`}>
                    <TrainResultCard result={latest} label="Latest Run" highlight />
                    {prev && <TrainResultCard result={prev} label="Previous Run" />}
                </div>
            )}
        </div>
    );
}

function TrainResultCard({ result, label: cardLabel, highlight }: {
    result: TrainResponse; label: string; highlight?: boolean;
}) {
    const cm = result.metrics.confusion_matrix as number[][] | undefined;

    return (
        <div className={`rounded-xl border ${highlight ? "border-emerald-500/30 bg-emerald-500/5" : "border-slate-800 bg-slate-900"}`}>
            <div className={`px-4 py-3 border-b flex items-center justify-between ${highlight ? "border-emerald-500/20" : "border-slate-800"}`}>
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{cardLabel}</span>
                <span className={`text-xs px-2.5 py-1 rounded-lg font-medium ${highlight ? "bg-emerald-500/10 text-emerald-300" : "bg-slate-800 text-slate-400"}`}>
                    {TECHNIQUES.train.find(m => m.value === result.model_name)?.label ?? result.model_name}
                </span>
            </div>
            <div className="p-4 space-y-4">
                {/* Primary Metrics */}
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    {Object.entries(result.metrics)
                        .filter(([k]) => !["confusion_matrix", "trials", "imbalance_warning", "best_trial_score", "n_trials", "best_cv_score"].includes(k))
                        .slice(0, 6)
                        .map(([k, v]) => (
                            <MetricCard
                                key={k}
                                label={label(k)}
                                value={typeof v === "number" ? v : Number(v)}
                                size="sm"
                            />
                        ))}
                </div>

                {/* Confusion Matrix */}
                {cm && (
                    <div className="mt-3">
                        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Confusion Matrix</p>
                        <div className="grid grid-cols-2 gap-1 max-w-[200px]">
                            {cm.map((row, i) => (
                                row.map((val, j) => (
                                    <div key={`${i}-${j}`} className={`rounded-lg p-3 text-center ${
                                        i === j
                                            ? val > 0
                                                ? "bg-emerald-500/15 border border-emerald-500/30"
                                                : "bg-red-500/10 border border-red-500/20"
                                            : "bg-slate-800/60 border border-slate-700/50"
                                    }`}>
                                        <div className={`text-lg font-bold font-mono ${
                                            i === j ? "text-emerald-400" : "text-red-400"
                                        }`}>{val}</div>
                                        <div className="text-[10px] text-slate-500">
                                            {i === j ? "Correct" : "Error"}
                                        </div>
                                    </div>
                                ))
                            ))}
                        </div>
                        <div className="flex gap-4 mt-2 text-[10px] text-slate-500">
                            <span>Rows: Actual</span>
                            <span>Cols: Predicted</span>
                        </div>
                    </div>
                )}

                {result.metrics.imbalance_warning && (
                    <div className="flex gap-2 mt-3 text-xs text-amber-400 bg-amber-500/5 border border-amber-500/20 rounded-lg p-2.5">
                        <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
                        {result.metrics.imbalance_warning}
                    </div>
                )}
                <div className="flex gap-4 mt-3 text-xs text-slate-500">
                    <span>Train: <span className="text-slate-300 font-mono">{result.train_size}</span></span>
                    <span>Test: <span className="text-slate-300 font-mono">{result.test_size}</span></span>
                    <span>Features: <span className="text-slate-300 font-mono">{result.n_features}</span></span>
                </div>
            </div>
        </div>
    );
}

// ─── TUNE VIEW ────────────────────────────────────────────────────────────────

function TuneView({ taskType, selectedModel, selectedMethod, onSelectModel,
    onSelectMethod, onApply, loading, result, onNext }: {
        taskType: string;
        selectedModel: string;
        selectedMethod: string;
        onSelectModel: (v: string) => void;
        onSelectMethod: (v: string) => void;
        onApply: () => void;
        loading: boolean;
        result: TrainResponse | null;
        onNext: () => void;
    }) {
    const trials = result?.metrics?.trials as any[] | undefined;

    return (
        <div className="max-w-3xl space-y-5">
            <div className="grid grid-cols-2 gap-5">
                <div>
                    <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Search Method</p>
                    <div className="space-y-2">
                        {[
                            { value: "optuna", label: "Bayesian Optimisation (Optuna)", desc: "Intelligent search that learns from each trial. Runs 60 iterations to find the best settings." },
                            { value: "gridsearch", label: "Grid Search (Exhaustive)", desc: "Tests every possible combination systematically. Thorough but takes longer." },
                        ].map(m => (
                            <button key={m.value} onClick={() => onSelectMethod(m.value)}
                                className={`w-full text-left px-4 py-3 rounded-xl border transition-all ${selectedMethod === m.value
                                        ? "bg-violet-600/10 border-violet-500/40"
                                        : "bg-slate-900 border-slate-800 hover:border-slate-700"}`}>
                                <div className={`text-sm font-semibold ${selectedMethod === m.value ? "text-violet-300" : "text-slate-200"}`}>{m.label}</div>
                                <div className="text-xs text-slate-500 mt-1">{m.desc}</div>
                            </button>
                        ))}
                    </div>
                </div>

                <div>
                    <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Model to Tune</p>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                        {TECHNIQUES.train.map(m => (
                            <button key={m.value} onClick={() => onSelectModel(m.value)}
                                className={`w-full text-left px-4 py-2.5 rounded-xl border transition-all ${selectedModel === m.value
                                        ? "bg-slate-700/70 border-slate-500/70"
                                        : "bg-slate-900 border-slate-800 hover:border-slate-700"}`}>
                                <div className={`text-sm font-semibold ${selectedModel === m.value ? "text-white" : "text-slate-300"}`}>{m.label}</div>
                            </button>
                        ))}
                    </div>
                    <p className="text-xs text-slate-500 mt-2">
                        Training model: <span className="text-slate-300 font-mono">{selectedModel}</span>
                    </p>
                </div>
            </div>

            <div className="flex items-center gap-3">
                <button onClick={onApply} disabled={!!loading}
                    className="flex items-center gap-2 px-5 py-2.5 bg-violet-600 hover:bg-violet-500
            disabled:opacity-40 text-white text-sm font-semibold rounded-xl transition-colors">
                    {loading
                        ? <><Loader2 className="w-4 h-4 animate-spin" />Searching...</>
                        : `Run ${selectedMethod === "gridsearch" ? "Grid Search" : "Bayesian Search"}`}
                </button>
                {result && (
                    <button onClick={onNext}
                        className="flex items-center gap-2 px-4 py-2.5 text-violet-400
              hover:text-violet-300 border border-violet-500/30
              text-sm font-medium rounded-xl transition-colors ml-auto">
                        Next: Explainability <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                )}
            </div>

            {result && (
                <div className="space-y-4">
                    <div className="bg-violet-500/5 border border-violet-500/20 rounded-xl p-4">
                        <p className="text-sm font-semibold text-violet-300 mb-3">Best Result Found</p>
                        <div className="grid grid-cols-2 gap-3">
                            {[
                                { k: "Best CV Score", v: result.metrics.best_trial_score ?? result.metrics.best_cv_score },
                                { k: "Trials Run", v: result.metrics.n_trials ?? "N/A" },
                            ].map(({ k, v }) => (
                                <MetricCard
                                    key={k}
                                    label={k}
                                    value={typeof v === "number" ? v : Number(v ?? 0)}
                                    size="md"
                                />
                            ))}
                        </div>
                    </div>

                    {/* Trial chart */}
                    {trials && trials.length > 0 && (
                        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
                            <p className="text-sm font-semibold text-white mb-4">Trial Score Progression</p>
                            <ResponsiveContainer width="100%" height={150}>
                                <LineChart data={trials}>
                                    <XAxis dataKey="number" tick={{ fill: "#64748b", fontSize: 11 }} label={{ value: "Trial", position: "insideBottomRight", fill: "#64748b", fontSize: 11 }} />
                                    <YAxis tick={{ fill: "#64748b", fontSize: 11 }} domain={["auto", "auto"]} />
                                    <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: any) => [Number(v).toFixed(4), "Score"]} />
                                    <Line type="monotone" dataKey="value" stroke="#8b5cf6" strokeWidth={2} dot={{ fill: "#8b5cf6", r: 3 }} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    )}

                    {/* Best params */}
                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
                        <p className="text-sm font-semibold text-white mb-3">Best Parameters Found</p>
                        <div className="grid grid-cols-2 gap-2">
                            {Object.entries(result.metrics)
                                .filter(([k]) => !["best_trial_score", "n_trials", "trials", "confusion_matrix",
                                    "accuracy", "f1_weighted", "f1_macro", "roc_auc", "cv_mean", "cv_std",
                                    "rmse", "mae", "r2", "best_cv_score", "imbalance_warning"].includes(k))
                                .map(([k, v]) => (
                                    <div key={k} className="bg-slate-800/60 rounded-lg p-2.5">
                                        <div className="text-xs text-slate-500">{label(k)}</div>
                                        <div className="text-sm font-mono text-white font-semibold mt-0.5">{String(v)}</div>
                                    </div>
                                ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

// ─── EXPLAIN VIEW ─────────────────────────────────────────────────────────────

function ExplainView({ trainResult, taskType, onNext }: {
    trainResult?: TrainResponse;
    taskType: string;
    onNext: () => void;
}) {
    if (!trainResult) return (
        <div className="max-w-2xl">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 text-center">
                <p className="text-slate-400">Complete Model Training (Step 10) first.</p>
            </div>
        </div>
    );

    const fi = Object.entries(trainResult.feature_importance);
    const maxFi = fi.length > 0 ? Number(fi[0][1]) : 1;
    const radarData = fi.slice(0, 8).map(([name, val]) => ({
        feature: name.length > 10 ? name.substring(0, 10) + "…" : name,
        importance: Number(val),
    }));

    return (
        <div className="max-w-3xl space-y-5">
            <div className="flex items-center gap-3 text-sm text-slate-400">
                <span>Model: <span className="text-white font-semibold">{TECHNIQUES.train.find(m => m.value === trainResult.model_name)?.label ?? trainResult.model_name}</span></span>
                <span className="text-slate-600">·</span>
                <span>{trainResult.n_features} features</span>
            </div>

            {/* Bar chart */}
            {fi.length > 0 && (
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                    <p className="text-sm font-semibold text-white mb-4">Feature Importance</p>
                    <ResponsiveContainer width="100%" height={220}>
                        <BarChart data={fi.slice(0, 12).map(([k, v]) => ({ name: k, value: Number(v) }))}
                            layout="vertical" margin={{ left: 10, right: 20 }}>
                            <XAxis type="number" tick={{ fill: "#64748b", fontSize: 11 }} />
                            <YAxis type="category" dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} width={120} />
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

            {/* Radar */}
            {radarData.length >= 3 && (
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                    <p className="text-sm font-semibold text-white mb-4">Feature Importance — Radar View</p>
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
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                <div className="flex items-center gap-2 mb-3">
                    <Sparkles className="w-4 h-4 text-blue-400" />
                    <p className="text-sm font-semibold text-white">Interpretation</p>
                </div>
                <div className="space-y-2">
                    {fi.slice(0, 3).map(([name, val], i) => (
                        <p key={name} className="text-sm text-slate-400 leading-relaxed">
                            <span className="text-slate-500">{["Most important:", "Second:", "Third:"][i]}</span>{" "}
                            <span className="text-white font-mono">{name}</span>
                            {` with importance score of ${Number(val).toFixed(4)}.`}
                            {i === 0 && " This feature drives the model's predictions most strongly."}
                        </p>
                    ))}
                </div>
            </div>

            <button onClick={onNext}
                className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-500
          text-white text-sm font-semibold rounded-xl transition-colors">
                Next: Experiment Comparison <ArrowRight className="w-4 h-4" />
            </button>
        </div>
    );
}

// ─── COMPARE VIEW ─────────────────────────────────────────────────────────────

function CompareView({ runs, taskType }: { runs: RunRecord[]; taskType: string }) {
    const [runA, setRunA] = useState("");
    const [runB, setRunB] = useState("");

    const selectedA = runs.find(r => r.id === runA) || runs[0];
    const selectedB = runs.find(r => r.id === runB) || runs[1];

    const metricKeys = taskType === "classification"
        ? ["accuracy", "f1_weighted", "f1_macro", "roc_auc", "cv_mean"]
        : ["rmse", "mae", "r2", "cv_mean"];

    if (runs.length < 1) return (
        <div className="max-w-2xl">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 text-center">
                <p className="text-slate-400 mb-1">No training runs yet.</p>
                <p className="text-xs text-slate-600">Complete Step 10 and/or 11 to generate runs.</p>
            </div>
        </div>
    );

    return (
        <div className="max-w-3xl space-y-5">
            <p className="text-sm text-slate-400">
                Compare any two training runs — metrics, parameters, and visual differences.
            </p>

            <div className="grid grid-cols-2 gap-4">
                {[
                    { label: "Run A", val: runA, setter: setRunA, color: "blue" },
                    { label: "Run B", val: runB, setter: setRunB, color: "violet" },
                ].map(({ label: lbl, val, setter, color }) => (
                    <div key={lbl}>
                        <p className={`text-xs font-semibold mb-2 ${color === "blue" ? "text-blue-400" : "text-violet-400"}`}>{lbl}</p>
                        <select value={val} onChange={e => setter(e.target.value)}
                            className="w-full text-sm bg-slate-900 border border-slate-700 rounded-xl
                px-4 py-2.5 text-slate-300 focus:outline-none focus:border-blue-500 font-medium">
                            <option value="">Latest run</option>
                            {runs.map(r => (
                                <option key={r.id} value={r.id}>
                                    {TECHNIQUES.train.find(m => m.value === r.model_name)?.label ?? r.model_name} — {new Date(r.created_at).toLocaleTimeString()}
                                </option>
                            ))}
                        </select>
                    </div>
                ))}
            </div>

            {selectedA && selectedB && (
                <>
                    {/* Metric comparison */}
                    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                        <div className="grid grid-cols-3 px-5 py-3 border-b border-slate-800">
                            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Metric</span>
                            <span className="text-xs font-semibold text-blue-400 uppercase tracking-wider">{TECHNIQUES.train.find(m => m.value === selectedA.model_name)?.label ?? selectedA.model_name}</span>
                            <span className="text-xs font-semibold text-violet-400 uppercase tracking-wider">{TECHNIQUES.train.find(m => m.value === selectedB.model_name)?.label ?? selectedB.model_name}</span>
                        </div>
                        {metricKeys.map(k => {
                            const vA = selectedA.metrics[k] as number | undefined;
                            const vB = selectedB.metrics[k] as number | undefined;
                            if (vA === undefined && vB === undefined) return null;
                            const lowerIsBetter = ["rmse", "mae"].includes(k);
                            const aWins = vA !== undefined && vB !== undefined && (lowerIsBetter ? vA < vB : vA > vB);
                            const bWins = vA !== undefined && vB !== undefined && (lowerIsBetter ? vB < vA : vB > vA);
                            return (
                                <div key={k} className="grid grid-cols-3 px-5 py-3 border-b border-slate-800/50">
                                    <span className="text-sm text-slate-400">{label(k)}</span>
                                    <span className={`text-sm font-mono font-semibold ${aWins ? "text-emerald-400" : "text-slate-300"}`}>
                                        {vA !== undefined ? vA.toFixed(4) : "—"} {aWins && "✓"}
                                    </span>
                                    <span className={`text-sm font-mono font-semibold ${bWins ? "text-emerald-400" : "text-slate-300"}`}>
                                        {vB !== undefined ? vB.toFixed(4) : "—"} {bWins && "✓"}
                                    </span>
                                </div>
                            );
                        })}
                    </div>

                    {/* Bar comparison */}
                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                        <p className="text-sm font-semibold text-white mb-4">Visual Comparison</p>
                        <ResponsiveContainer width="100%" height={180}>
                            <BarChart data={metricKeys
                                .filter(k => selectedA.metrics[k] !== undefined)
                                .map(k => ({
                                    metric: label(k),
                                    [selectedA.model_name]: Number(selectedA.metrics[k] || 0),
                                    [selectedB?.model_name || "B"]: Number(selectedB?.metrics[k] || 0),
                                }))}>
                                <XAxis dataKey="metric" tick={{ fill: "#64748b", fontSize: 11 }} />
                                <YAxis tick={{ fill: "#64748b", fontSize: 11 }} domain={[0, 1]} />
                                <Tooltip contentStyle={TOOLTIP_STYLE} />
                                <Bar dataKey={selectedA.model_name} fill="#3b82f6" radius={[3, 3, 0, 0]} opacity={0.85} />
                                <Bar dataKey={selectedB?.model_name || "B"} fill="#8b5cf6" radius={[3, 3, 0, 0]} opacity={0.85} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </>
            )}

            {runs.length === 1 && (
                <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-4">
                    <p className="text-sm text-amber-400">
                        Only one run found. Train another model in Step 10 to compare.
                    </p>
                </div>
            )}
        </div>
    );
}

// ─── PREDICT VIEW ─────────────────────────────────────────────────────────────

function PredictView({ profile, session, finalModel }: {
    profile: DatasetProfile; session: Session; finalModel?: TrainResponse;
}) {
    const [inputs, setInputs] = useState<Record<string, string>>({});
    const [result, setResult] = useState<{ prediction: any; confidence: number | null; model_used: string } | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const inputCols = profile.columns.filter(c => !c.is_target);

    async function handlePredict() {
        setLoading(true);
        setError(null);
        try {
            const inputData: Record<string, any> = {};
            inputCols.forEach(col => {
                const val = inputs[col.name];
                inputData[col.name] = col.type === "numeric" ? parseFloat(val) || 0 : val || "";
            });
            const { predict } = await import("@/lib/api");
            const res = await predict(session.id, inputData);
            setResult(res);
        } catch (e: any) {
            setError(e.message || "Prediction failed. Train a model first.");
        } finally {
            setLoading(false);
        }
    }

    if (!finalModel) return (
        <div className="text-xs text-slate-500">Train a model in Step 10 before making predictions.</div>
    );

    return (
        <div className="max-w-2xl space-y-4">
            <p className="text-xs text-slate-500">
                Enter feature values below and get a live prediction from your trained{" "}
                <span className="text-slate-300 font-semibold">
                    {TECHNIQUES.train.find(m => m.value === finalModel.model_name)?.label || finalModel.model_name}
                </span> model.
            </p>

            <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 grid grid-cols-2 gap-3">
                {inputCols.map(col => (
                    <div key={col.name}>
                        <label className="text-xs text-slate-500 font-medium block mb-1">
                            {col.name} <span className="text-slate-700">({col.type})</span>
                        </label>
                        <input
                            value={inputs[col.name] || ""}
                            onChange={e => setInputs(p => ({ ...p, [col.name]: e.target.value }))}
                            placeholder={col.type === "numeric" ? String(col.stats?.mean ?? 0) : "value"}
                            className="w-full text-xs bg-slate-950 border border-slate-800 rounded-md
                px-2.5 py-1.5 text-slate-300 focus:outline-none focus:border-blue-500"
                        />
                    </div>
                ))}
            </div>

            {error && <p className="text-xs text-red-400">{error}</p>}

            <button onClick={handlePredict} disabled={loading}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500
          disabled:opacity-40 text-white text-xs font-semibold rounded-lg transition-colors">
                {loading ? <><Loader2 className="w-3.5 h-3.5 animate-spin" />Predicting...</> : "Predict →"}
            </button>

            {result && (
                <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-lg p-4">
                    <p className="text-xs text-slate-500 mb-1">Prediction</p>
                    <p className="text-2xl font-bold font-mono text-emerald-400">{String(result.prediction)}</p>
                    {result.confidence !== null && (
                        <p className="text-xs text-slate-500 mt-1">Confidence: {(result.confidence * 100).toFixed(1)}%</p>
                    )}
                </div>
            )}
        </div>
    );
}

// ─── BOTTOM CHART BAR ─────────────────────────────────────────────────────────

function BottomChartBar({ open, setOpen, chartType, setChartType, chartCol, setChartCol,
    profile, distributions, correlation, targetData, trainResult }: {
        open: boolean; setOpen: (v: boolean) => void;
        chartType: string; setChartType: (v: string) => void;
        chartCol: string; setChartCol: (v: string) => void;
        profile: DatasetProfile;
        distributions: EDADistributions | null;
        correlation: EDACorrelation | null;
        targetData: EDATargetAnalysis | null;
        trainResult?: TrainResponse;
    }) {
    const col = chartCol || Object.keys(distributions || {})[0] || "";

    const options = [
        { value: "distribution", label: "Distribution" },
        { value: "missing", label: "Missing Values" },
        { value: "correlation", label: "Correlations" },
        { value: "target", label: "Target Analysis" },
        ...(trainResult ? [{ value: "importance", label: "Feature Importance" }] : []),
    ];

    return (
        <div className="border-t border-slate-800 bg-slate-950 flex-shrink-0">
            <div className="flex items-center gap-4 px-5 py-2.5">
                <button onClick={() => setOpen(!open)}
                    className="flex items-center gap-2 text-xs font-semibold text-slate-500
            hover:text-slate-300 transition-colors uppercase tracking-wider">
                    <TrendingUp className="w-3.5 h-3.5" />
                    Charts
                    <span className="text-slate-700">{open ? "▲" : "▼"}</span>
                </button>

                {open && (
                    <>
                        <select value={chartType} onChange={e => setChartType(e.target.value)}
                            className="text-xs bg-slate-900 border border-slate-800 rounded-lg
                px-3 py-1.5 text-slate-400 focus:outline-none focus:border-slate-600 font-medium">
                            {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                        </select>
                        {chartType === "distribution" && distributions && (
                            <select value={col} onChange={e => setChartCol(e.target.value)}
                                className="text-xs bg-slate-900 border border-slate-800 rounded-lg
                  px-3 py-1.5 text-slate-400 focus:outline-none focus:border-slate-600 font-medium">
                                {Object.keys(distributions).map(c => <option key={c} value={c}>{c}</option>)}
                            </select>
                        )}
                    </>
                )}
            </div>

            {open && (
                <div className="px-5 pb-4">
                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4" style={{ height: 180 }}>

                        {chartType === "distribution" && distributions && distributions[col] && (
                            <ResponsiveContainer width="100%" height="100%">
                                {distributions[col].type === "numeric" ? (
                                    <AreaChart data={distributions[col].histogram!.counts.map((c, i) => ({
                                        x: Number(distributions[col].histogram!.edges[i]).toFixed(1), v: c
                                    }))}>
                                        <defs>
                                            <linearGradient id="distGrad" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
                                            </linearGradient>
                                        </defs>
                                        <XAxis dataKey="x" tick={{ fill: "#334155", fontSize: 9 }} interval={4} />
                                        <YAxis tick={{ fill: "#334155", fontSize: 9 }} />
                                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                                        <Area type="monotone" dataKey="v" stroke="#3b82f6" fill="url(#distGrad)" strokeWidth={2} />
                                    </AreaChart>
                                ) : (
                                    <BarChart data={distributions[col].bar!.labels.map((l, i) => ({
                                        label: l, count: distributions[col].bar!.counts[i]
                                    }))} layout="vertical">
                                        <XAxis type="number" tick={{ fill: "#334155", fontSize: 9 }} />
                                        <YAxis type="category" dataKey="label" tick={{ fill: "#334155", fontSize: 9 }} width={70} />
                                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                                        <Bar dataKey="count" fill="#8b5cf6" radius={[0, 2, 2, 0]} opacity={0.8} />
                                    </BarChart>
                                )}
                            </ResponsiveContainer>
                        )}

                        {chartType === "missing" && (
                            <div className="overflow-y-auto h-full space-y-2 pr-1">
                                {profile.columns.map(c => (
                                    <div key={c.name} className="flex items-center gap-3">
                                        <span className="text-xs font-mono text-slate-600 w-24 truncate">{c.name}</span>
                                        <div className="flex-1 bg-slate-800 rounded-full h-1.5">
                                            <div className="h-1.5 rounded-full" style={{
                                                width: `${Math.max(c.missing_pct, c.missing_pct > 0 ? 1 : 0)}%`,
                                                background: c.missing_pct > 20 ? "#ef4444" : c.missing_pct > 0 ? "#f59e0b" : "#10b981",
                                            }} />
                                        </div>
                                        <span className="text-xs text-slate-600 font-mono w-8 text-right">{c.missing_pct}%</span>
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
                                    <XAxis dataKey="label" tick={{ fill: "#334155", fontSize: 10 }} />
                                    <YAxis tick={{ fill: "#334155", fontSize: 10 }} />
                                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                                    <Bar dataKey="count" fill="#f59e0b" radius={[3, 3, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        )}

                        {chartType === "importance" && trainResult && (
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart layout="vertical"
                                    data={Object.entries(trainResult.feature_importance).slice(0, 8)
                                        .map(([k, v]) => ({ name: k, value: Number(v) }))}>
                                    <XAxis type="number" tick={{ fill: "#334155", fontSize: 9 }} />
                                    <YAxis type="category" dataKey="name" tick={{ fill: "#334155", fontSize: 9 }} width={90} />
                                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                                    <Bar dataKey="value" fill="#10b981" radius={[0, 3, 3, 0]}>
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

function RightPanel({ latestResult, trainResult }: {
    latestResult?: StepResponse;
    trainResult?: TrainResponse;
}) {
    const fi = trainResult ? Object.entries(trainResult.feature_importance).slice(0, 8) : [];
    const maxFi = fi.length > 0 ? Number(fi[0][1]) : 1;

    return (
        <aside className="w-64 bg-slate-900 border-l border-slate-800 flex flex-col overflow-y-auto flex-shrink-0">

            {/* AI explanation */}
            <div className="border-b border-slate-800">
                <div className="flex items-center gap-1.5 px-4 py-2.5 border-b border-slate-800">
                    <Sparkles className="w-3 h-3 text-blue-400" />
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">AI Explanation</span>
                </div>
                <div className="px-4 pb-3 pt-2">
                    {latestResult?.ai_explanation ? (
                        <p className="text-xs text-slate-400 leading-relaxed">{latestResult.ai_explanation}</p>
                    ) : (
                        <p className="text-xs text-slate-600 leading-relaxed">Apply a technique to see the explanation.</p>
                    )}
                </div>
            </div>

            {/* Warnings */}
            {latestResult?.warnings && latestResult.warnings.length > 0 && (
                <div className="border-b border-slate-800 px-4 py-3 space-y-2">
                    {latestResult.warnings.map((w, i) => (
                        <div key={i} className="flex gap-2 text-xs text-amber-400/80">
                            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                            <span className="leading-relaxed">{w}</span>
                        </div>
                    ))}
                </div>
            )}

            {/* Stats */}
            {latestResult?.stats && (
                <div className="border-b border-slate-800 px-4 py-3">
                    <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Statistics</p>
                    <div className="space-y-2">
                        {Object.entries(latestResult.stats)
                            .filter(([k]) => STATS_KEYS.includes(k))
                            .map(([k, v]) => (
                                <div key={k} className="flex justify-between items-center">
                                    <span className="text-xs text-slate-500">{label(k)}</span>
                                    <span className="text-xs font-mono text-slate-300 font-semibold">{String(v)}</span>
                                </div>
                            ))}
                    </div>
                </div>
            )}

            {/* Train metrics */}
            {trainResult && (
                <div className="border-b border-slate-800 px-4 py-3">
                    <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Model Metrics</p>
                    <div className="grid grid-cols-2 gap-2">
                        {Object.entries(trainResult.metrics)
                            .filter(([k]) => !["confusion_matrix", "trials", "best_trial_score", "n_trials", "imbalance_warning"].includes(k))
                            .slice(0, 6)
                            .map(([k, v]) => (
                                <MetricCard
                                    key={k}
                                    label={label(k)}
                                    value={typeof v === "number" ? v : Number(v)}
                                    size="sm"
                                />
                            ))}
                    </div>
                </div>
            )}

            {/* Feature importance */}
            {fi.length > 0 && (
                <div className="px-4 py-3">
                    <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Feature Importance</p>
                    <FeatureImportance data={Object.fromEntries(fi.map(([name, val]) => [name, Number(val)]))} />
                </div>
            )}

            {!latestResult && !trainResult && (
                <div className="flex-1 flex items-center justify-center px-4 py-8">
                    <p className="text-sm text-slate-700 text-center leading-relaxed">
                        Results appear here after each step.
                    </p>
                </div>
            )}
        </aside>
    );
}