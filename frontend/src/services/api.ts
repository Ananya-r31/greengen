/** Axios REST API client for the GreenGen FastAPI Backend. */
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export async function uploadDatasetFile(file: File, isSynthetic = false) {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await apiClient.post(`/api/datasets/upload?is_synthetic=${isSynthetic}`, formData);
  return data;
}

export async function fetchDatasetProfile(datasetId: string, filters?: Record<string, string>) {
  const { data } = await apiClient.get(`/api/datasets/${datasetId}/profile`, { params: filters });
  return data;
}

export async function fetchDatasetPreview(datasetId: string, limit = 25) {
  const { data } = await apiClient.get(`/api/datasets/${datasetId}/preview`, { params: { limit } });
  return data;
}

export async function preprocessDataset(datasetId: string, payload: Record<string, unknown>) {
  const { data } = await apiClient.post(`/api/datasets/${datasetId}/preprocess`, payload);
  return data;
}

export async function trainDatasetModels(datasetId: string, payload: Record<string, unknown>) {
  const { data } = await apiClient.post(`/api/datasets/${datasetId}/train`, payload);
  return data;
}

export async function fetchTrainedModels(datasetId: string) {
  const { data } = await apiClient.get(`/api/datasets/${datasetId}/models`);
  return data;
}

export async function runPrediction(datasetId: string, payload: Record<string, unknown>) {
  const { data } = await apiClient.post(`/api/datasets/${datasetId}/predict`, payload);
  return data;
}

export async function runForecast(datasetId: string, payload: Record<string, unknown>) {
  const { data } = await apiClient.post(`/api/datasets/${datasetId}/forecast`, payload);
  return data;
}

export async function runCarbonOptimization(datasetId: string, payload: Record<string, unknown>) {
  const { data } = await apiClient.post(`/api/datasets/${datasetId}/optimize`, payload);
  return data;
}

export async function runWhatIfScenarios(datasetId: string, payload: Record<string, unknown>) {
  const { data } = await apiClient.post(`/api/datasets/${datasetId}/scenarios`, payload);
  return data;
}

export async function querySustainabilityAssistant(datasetId: string, question: string, sectorFilter?: string) {
  const { data } = await apiClient.post(`/api/datasets/${datasetId}/assistant/chat`, {
    question,
    sector_filter: sectorFilter,
  });
  return data;
}

export async function fetchRecommendationsAndPlan(datasetId: string) {
  const { data } = await apiClient.get(`/api/datasets/${datasetId}/recommendations`);
  return data;
}

export async function saveActionPlan(datasetId: string, items: unknown[]) {
  const { data } = await apiClient.post(`/api/datasets/${datasetId}/action-plan`, { items });
  return data;
}

export async function downloadSustainabilityReport(datasetId: string, format: 'pdf' | 'csv' | 'xlsx', organizationName: string) {
  const response = await apiClient.post(
    `/api/datasets/${datasetId}/reports`,
    { format, organization_name: organizationName },
    { responseType: 'blob' }
  );
  return response.data;
}
