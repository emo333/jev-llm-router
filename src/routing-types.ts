import type { ModelThinkingLevel } from "@earendil-works/pi-ai";

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
export interface RouteState {
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
}
