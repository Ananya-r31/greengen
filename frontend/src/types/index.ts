/** TypeScript domain interfaces for the GreenGen Sustainability Platform. */

export type NavSection =
  | 'overview'
  | 'dataset'
  | 'models'
  | 'optimization'
  | 'assistant'
  | 'recommendations'
  | 'reports'
  | 'system';

export interface ColumnProfile {
  name: string;
  dtype: 'numeric' | 'categorical' | 'datetime';
  concept: string;
  unit: string;
  missingCount: number;
  missingPct: number;
  uniqueCount: number;
  mean?: number;
  std?: number;
  min?: number;
  median?: number;
  max?: number;
  iqrOutliers?: number;
  topValues?: Record<string, number>;
}

export interface DatasetBundle {
  datasetId: string;
  filename: string;
  isSynthetic: boolean;
  uploadedAt: string;
  rows: Record<string, string | number | null>[];
  rawRows: Record<string, string | number | null>[];
  columns: ColumnProfile[];
  duplicateRows: number;
  missingCells: number;
  missingPct: number;
  qualityScore: number;
  primaryTarget: string | null;
  targetCandidates: string[];
  controllableColumns: string[];
  conceptMap: Record<string, string>;
  unitMap: Record<string, string>;
  preprocessingLogs: string[];
  guidanceMessage: string;
}

export interface ModelEvaluationResult {
  modelKey: 'random_forest' | 'bayesian_ridge' | 'xgboost' | 'gaussian_process';
  modelName: string;
  status: 'Trained';
  durationMs: number;
  mae: number;
  rmse: number;
  r2: number;
  mape: number | null;
  mapeNote: string;
  nTrain: number;
  nTest: number;
  isBest: boolean;
  uncertaintySupported: boolean;
  featureImportance: { feature: string; importance: number }[];
  comparisonPoints: { index: number; actual: number; predicted: number; residual: number }[];
  limitations: string[];
  predictFn: (inputRow: Record<string, number | string>) => { mean: number; std: number };
}

export interface OptimizationVariableChange {
  variable: string;
  concept: string;
  unit: string;
  baselineValue: number;
  optimizedValue: number;
  absoluteChange: number;
  pctChange: number;
  boundMin: number;
  boundMax: number;
  withinTrainingDist: boolean;
}

export interface OptimizationResult {
  optId: string;
  modelKey: string;
  modelName: string;
  targetColumn: string;
  unit: string;
  targetReductionPct: number;
  baselineEmissions: number;
  optimizedEmissions: number;
  absoluteReduction: number;
  percentageReduction: number;
  zeroBaselineNote: string | null;
  status: string;
  variableChanges: OptimizationVariableChange[];
  constraintsReport: { constraint: string; satisfied: boolean; detail: string }[];
  estimatedCost: number | null;
  costStatusNote: string;
}

export interface ActionPlanItem {
  id: string;
  horizon: 'Immediate (0–3m)' | 'Short-Term (3–6m)' | 'Medium-Term (6–12m)' | 'Long-Term (1–3y)';
  description: string;
  priority: 'High' | 'Medium' | 'Low';
  responsibleRole: string;
  targetCompletionDate: string;
  requiredResources: string;
  estimatedCost: string;
  expectedOutcome: string;
  kpi: string;
  progressStatus: 'Not Started' | 'Planned' | 'In Progress' | 'Completed';
}
