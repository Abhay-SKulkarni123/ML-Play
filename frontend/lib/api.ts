import axios from "axios";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  timeout: 180_000,
});

// Normalise error messages
api.interceptors.response.use(
  (r) => r,
  (err) => {
    const detail = err?.response?.data?.detail;
    const msg =
      typeof detail === "string"
        ? detail
        : typeof detail === "object" && detail?.error
          ? detail.error
          : err?.message || "Unknown error";
    return Promise.reject(new Error(msg));
  },
);

// ─── TYPES ────────────────────────────────────────────────────────────────────

export interface Dataset {
  id: string;
  name: string;
  target: string;
  task: string;
  rows: number;
  cols: number;
}

export interface Session {
  id: string;
  dataset_id: string;
  task_type: string;
  current_step: number;
  pipeline_state: Record<string, any>;
  created_at: string;
}

export interface StepResponse {
  step: string;
  technique: string;
  params: Record<string, any>;
  stats: Record<string, any>;
  warnings: string[];
  ai_explanation: string;
  ai_recommendation: string;
}

export interface TrainResponse {
  run_id: string;
  model_name: string;
  metrics: Record<string, any>;
  feature_importance: Record<string, number>;
  train_size: number;
  test_size: number;
  n_features: number;
}

export interface RunRecord {
  id: string;
  model_name: string;
  params: Record<string, any>;
  metrics: Record<string, any>;
  feature_importance: Record<string, number>;
  created_at: string;
}

export interface DatasetProfile {
  shape: { rows: number; cols: number };
  missing_summary: { total_missing_cells: number };
  duplicate_rows: number;
  columns: Array<{
    name: string;
    type: string;
    missing_count: number;
    missing_pct: number;
    unique_count: number;
    is_target: boolean;
    stats?: Record<string, number>;
  }>;
}

export interface EDADistributions {
  [col: string]: {
    type: "numeric" | "categorical";
    histogram?: { counts: number[]; edges: number[] };
    bar?: { labels: string[]; counts: number[] };
    stats?: Record<string, number>;
  };
}

export interface EDACorrelation {
  columns: string[];
  matrix: number[][];
}

export interface EDATargetAnalysis {
  target: string;
  is_imbalanced: boolean;
  class_balance?: number;
  distribution?: { labels: (string | number)[]; counts: number[] };
}

export interface AutoMLStatus {
  status: "pending" | "running" | "done" | "error";
  progress: number;
  log?: string[];
  task?: string;
  metrics?: Record<string, number>;
  feature_importance?: Record<string, number>;
  pipeline_summary?: {
    n_trials: number;
    best_trial_score: number;
    best_params: Record<string, any>;
    model: string;
    missing_cols_dropped?: string[];
    categorical_cols_encoded?: string[];
    imputation?: string;
    scaling?: string;
  };
  shape?: { rows: number; cols: number };
  n_features?: number;
  train_size?: number;
  test_size?: number;
  error?: string;
}

// ─── DATASETS ─────────────────────────────────────────────────────────────────

export const getDatasets = () =>
  api.get<Dataset[]>("/datasets/").then((r) => r.data);

export const getProfile = (datasetId: string) =>
  api.get<DatasetProfile>(`/datasets/${datasetId}/profile`).then((r) => r.data);

export const uploadDataset = (formData: FormData) =>
  api
    .post<{
      dataset_id: string;
      name: string;
      target: string;
      task: string;
      rows: number;
      cols: number;
    }>("/datasets/upload", formData, { headers: { "Content-Type": "multipart/form-data" } })
    .then((r) => r.data);

// ─── EDA ──────────────────────────────────────────────────────────────────────

export const getDistributions = (datasetId: string) =>
  api
    .get<EDADistributions>(`/eda/${datasetId}/distributions`)
    .then((r) => r.data);

export const getCorrelation = (datasetId: string) =>
  api.get<EDACorrelation>(`/eda/${datasetId}/correlation`).then((r) => r.data);

export const getTargetAnalysis = (datasetId: string) =>
  api
    .get<EDATargetAnalysis>(`/eda/${datasetId}/target-analysis`)
    .then((r) => r.data);

// ─── SESSIONS ─────────────────────────────────────────────────────────────────

export const createSession = (datasetId: string) =>
  api
    .post<Session>("/sessions/", { dataset_id: datasetId })
    .then((r) => r.data);

export const getSession = (sessionId: string) =>
  api.get<Session>(`/sessions/${sessionId}`).then((r) => r.data);

// ─── STEPS ────────────────────────────────────────────────────────────────────

export const runStep = (
  sessionId: string,
  step: string,
  technique: string,
  params: Record<string, any> = {},
) => {
  const STEP_ROUTES: Record<string, string> = {
    missing: "missing",
    outliers: "outliers",
    features: "features",
    encoding: "encoding",
    selection: "selection",
    pca: "pca",
    scaling: "scaling",
  };
  const route = STEP_ROUTES[step] || step;
  return api
    .post<StepResponse>(`/sessions/${sessionId}/steps/${route}`, {
      technique,
      params,
    })
    .then((r) => r.data);
};

// ─── TRAINING ─────────────────────────────────────────────────────────────────

export const trainModel = (
  sessionId: string,
  modelName: string,
  params: Record<string, any> = {},
  testSize = 0.2,
) =>
  api
    .post<TrainResponse>(`/sessions/${sessionId}/train`, {
      model_name: modelName,
      params,
      test_size: testSize,
    })
    .then((r) => r.data);

export const tuneModel = (
  sessionId: string,
  modelName: string,
  testSize = 0.2,
) =>
  api
    .post<TrainResponse>(`/sessions/${sessionId}/tune`, {
      model_name: modelName,
      params: {},
      test_size: testSize,
    })
    .then((r) => r.data);

export const gridSearch = (sessionId: string, testSize = 0.2) =>
  api
    .post<TrainResponse>(`/sessions/${sessionId}/gridsearch`, {
      model_name: "random_forest",
      params: {},
      test_size: testSize,
    })
    .then((r) => r.data);

export const getRuns = (sessionId: string) =>
  api.get<RunRecord[]>(`/sessions/${sessionId}/runs`).then((r) => r.data);

// ─── RESET STEP ───────────────────────────────────────────────────────────────

export const resetStep = (sessionId: string, stepName: string) =>
  api
    .post<{ message: string; pipeline_state: Record<string, any> }>(
      `/sessions/${sessionId}/steps/${stepName}/reset`
    )
    .then((r) => r.data);

// ─── SAVE / SHARE ─────────────────────────────────────────────────────────────

export const saveSession = (sessionId: string, name: string) =>
  api
    .post<{ message: string; name: string }>(`/sessions/${sessionId}/save`, { name })
    .then((r) => r.data);

export const shareSession = (sessionId: string) =>
  api
    .post<{ share_token: string; share_url: string }>(`/sessions/${sessionId}/share`)
    .then((r) => r.data);

export const getSharedSession = (shareToken: string) =>
  api
    .get<{
      id: string;
      dataset_id: string;
      task_type: string;
      current_step: number;
      pipeline_state: Record<string, any>;
      name: string;
      created_at: string;
    }>(`/sessions/shared/${shareToken}`)
    .then((r) => r.data);

export const importSession = (sessionData: {
  dataset_id: string;
  task_type?: string;
  pipeline_state: Record<string, any>;
  name?: string;
}) =>
  api
    .post<{
      id: string;
      dataset_id: string;
      task_type: string;
      message: string;
    }>(`/sessions/import`, sessionData)
    .then((r) => r.data);

// ─── APPLY AUTOML ─────────────────────────────────────────────────────────────

export const applyAutoML = (sessionId: string, jobId: string) =>
  api
    .post<TrainResponse>(`/sessions/${sessionId}/apply-automl`, {
      job_id: jobId,
    })
    .then((r) => r.data);

// ─── PREDICT ──────────────────────────────────────────────────────────────────

export const predict = (sessionId: string, inputData: Record<string, any>) =>
  api
    .post<{
      prediction: number | string;
      confidence: number | null;
      model_used: string;
    }>(`/sessions/${sessionId}/predict`, { input_data: inputData })
    .then((r) => r.data);

// ─── AUTOML ───────────────────────────────────────────────────────────────────

// AutoML tab — file upload
export const runAutoML = (formData: FormData) =>
  api
    .post<{
      job_id: string;
      status: string;
      columns: string[];
    }>("/automl/run", formData, { headers: { "Content-Type": "multipart/form-data" } })
    .then((r) => r.data);

// Dataset cards — runs AutoML on already-stored dataset
export const runAutoMLForDataset = (datasetId: string, targetCol = "") =>
  api
    .post<{
      job_id: string;
      status: string;
      dataset: string;
      target: string;
    }>("/automl/run-for-dataset", null, { params: { dataset_id: datasetId, target_col: targetCol } })
    .then((r) => r.data);

// Poll job status
export const getAutoMLStatus = (jobId: string) =>
  api.get<AutoMLStatus>(`/automl/status/${jobId}`).then((r) => r.data);