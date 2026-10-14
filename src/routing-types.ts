import type { ModelThinkingLevel, Usage } from "@earendil-works/pi-ai";

export type TaskFamily = "mechanical" | "coding" | "reasoning" | "research" | "review" | "unknown";
export interface TaskAssessment {
  family: TaskFamily;
  risk: "low" | "high" | "unknown";
  verifiable: boolean;
  boundedExecution: boolean;
  phase: "planning" | "execution" | "review";
  outputTokens: number;
}
export interface EffectiveControl {
  level: ModelThinkingLevel;
  native: string;
  key: string;
  effortRank: number;
  outputReserve: number;
}
export interface ExecutionEvidence {
  fingerprint: string;
  edited: boolean;
  hasPlan: boolean;
  repeatedFailure: boolean;
  verificationFailed: boolean;
  failures: number;
}
export interface TaskBrief {
  summary: string;
  deliverables: string[];
  constraints: string[];
  architecture: string[];
  difficulties: string[];
  capabilities: string[];
  approach: string[];
  acceptanceChecks: string[];
  unknowns: string[];
  sources: { path: string; fingerprint: string }[];
}
export interface RouteDecision {
  method: "assessment" | "classifier" | "fallback" | "deterministic";
  reason: string;
  candidates: { provider: string; id: string; levels: ModelThinkingLevel[] }[];
  assessor?: { provider: string; id: string; thinkingLevel: ModelThinkingLevel };
  usage?: Usage[];
  durationMs?: number;
  alternatives?: string[];
}
export interface RouteState {
  taskBrief?: TaskBrief;
  taskPromptKey?: string;
  taskEntryId?: string;
  decision?: RouteDecision;
  provider: string;
  id: string;
  thinkingLevel: ModelThinkingLevel;
  taskId?: string;
  assessment?: TaskAssessment;
  phase?: TaskAssessment["phase"];
  strongest?: { provider: string; id: string };
  evidenceFingerprint?: string;
  escalations?: number;
  verificationAttempts?: number;
  verificationFailed?: boolean;
  classifierVersion?: string;
  prediction?: number;
  controlKey?: string;
  modelKey?: string;
  configurationKey?: string;
  resolvedName?: string;
  excluded?: string[];
  excludeLocalModels?: boolean;
}
