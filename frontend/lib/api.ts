import axios from "axios";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  headers: { "Content-Type": "application/json" },
});

export interface Dataset {
  id: string;
  name: string;
  target: string;
  task: string;
  rows: number;
  cols: number;
}

export interface ColumnProfile {
  name: string;
  dtype: string;
  type: "numeric" | "categorical";
  missing_count: number;
  missing_pct: number;
  unique_count: number;
  is_target: boolean;
  stats: Record<string, any>;
  histogram?: { counts: number[]; edges: number[] };
}

export interface DatasetProfile {
  shape: { rows: number; cols: number };
  target: string;
  task: string;
  columns: ColumnProfile[];
  missing_summary: {
    total_missing_cells: number;
    columns_with_missing: string[];
    pct_complete: number;
  };
  duplicate_rows: number;
}

export interface StepResponse {
  step: string;
  technique: string;
  params: Record<string, any>;
  stats: Record<string, any>;
  warnings: string[];
  ai_explanation: string;
  metrics_delta: Record<string, any>;
}

export interface TrainResponse {
  run_id: string;
  model: string;
  metrics: Record<string, any>;
  feature_importance: Record<string, number>;
  train_size: number;
  test_size: number;
  n_features: number;
}

export interface Session {
  id: string;
  dataset_id: string;
  task_type: string;
  current_step: number;
  pipeline_state: Record<string, any>;
  created_at: string;
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
  task: string;
  distribution?: { labels: string[]; counts: number[] };
  class_balance?: number;
  is_imbalanced?: boolean;
  histogram?: { counts: number[]; edges: number[] };
  stats?: Record<string, number>;
}

export const getDatasets = () =>
  api.get<Dataset[]>("/datasets/").then((r) => r.data);
export const getProfile = (id: string) =>
  api.get<DatasetProfile>(`/datasets/${id}/profile`).then((r) => r.data);
export const createSession = (dataset_id: string) =>
  api.post<Session>("/sessions/", { dataset_id }).then((r) => r.data);
export const getSession = (id: string) =>
  api.get<Session>(`/sessions/${id}`).then((r) => r.data);
export const getDistributions = (id: string) =>
  api.get<EDADistributions>(`/eda/${id}/distributions`).then((r) => r.data);
export const getCorrelation = (id: string) =>
  api.get<EDACorrelation>(`/eda/${id}/correlation`).then((r) => r.data);
export const getTargetAnalysis = (id: string) =>
  api.get<EDATargetAnalysis>(`/eda/${id}/target-analysis`).then((r) => r.data);

export const runStep = (
  sessionId: string,
  step: string,
  technique: string,
  params = {},
) =>
  api
    .post<StepResponse>(`/sessions/${sessionId}/steps/${step}`, {
      technique,
      params,
    })
    .then((r) => r.data);

export const trainModel = (
  sessionId: string,
  model_name: string,
  params = {},
  test_size = 0.2,
) =>
  api
    .post<TrainResponse>(`/sessions/${sessionId}/train`, {
      model_name,
      params,
      test_size,
    })
    .then((r) => r.data);
