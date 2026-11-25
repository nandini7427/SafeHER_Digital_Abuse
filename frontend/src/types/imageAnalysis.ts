/**
 * TypeScript definitions for SafeHER Visual Manipulation & Deepfake Intelligence.
 * Strictly aligned with Pydantic v2 schemas from ai_core/image_intelligence/schemas.py.
 */

export type DetectionTask =
  | 'ai_generated_image'
  | 'face_manipulation'
  | 'hybrid_manipulation'
  | 'forensic_triage';

export type AssessmentLabel =
  | 'no_obvious_artifacts'
  | 'suspicious_manipulation'
  | 'highly_likely_manipulated'
  | 'inconclusive';

export type UncertaintyState =
  | 'low'
  | 'moderate'
  | 'high'
  | 'insufficient_quality'
  | 'analysis_usable_evidence'
  | 'analysis_poor_quality_evidence'
  | 'model_uncertain'
  | 'model_unavailable'
  | 'outside_supported_domain'
  | 'analysis_cannot_be_completed';

export interface ImageQualityIndicators {
  width: number;
  height: number;
  megapixels: number;
  aspect_ratio: number;
  resolution_ok: boolean;
  blur_score: number;
  is_blurry: boolean;
  compression_artifact_score: number;
  has_heavy_compression: boolean;
  mean_luminance: number;
  overexposed_ratio: number;
  underexposed_ratio: number;
  exposure_level: 'normal' | 'overexposed' | 'underexposed' | 'extreme_contrast' | string;
  is_monochromatic_or_flat: boolean;
  is_screen_capture: boolean;
  face_detected?: boolean | null;
  face_count: number;
  face_resolution_ok?: boolean | null;
  face_size_ratio?: number | null;
  corrupted_data_detected: boolean;
  domain_category?: string | null;
  is_in_domain?: boolean | null;
  overall_quality_score: number;
}

export type ManipulationRegionType = 'bounding_box' | 'grid_cell' | 'attribution_patch';

export interface ManipulationRegion {
  /** [ymin, xmin, ymax, xmax] normalized between 0.0 and 1.0 */
  box_normalized: [number, number, number, number];
  anomaly_score: number;
  region_type: ManipulationRegionType;
  attribution_category?: 'suspected_manipulation_region' | 'model_attention_attribution' | string;
  description: string;
}

export interface ImageMetadataSummary {
  format: string;
  dimensions: [number, number];
  size_bytes: number;
  sha256: string;
  has_exif: boolean;
  sanitized_exif: Record<string, unknown>;
  color_mode: string;
}

export interface AIGeneratedResult {
  is_available: boolean;
  model_version: string;
  architecture: string;
  raw_logits: number[];
  uncalibrated_score: number;
  calibrated_probability: number | null;
  decision_state: string;
  uncertainty?: Record<string, unknown>;
  disclaimer: string;
}

export interface FacialManipulationResult {
  is_available: boolean;
  model_version: string;
  face_detected: boolean;
  face_count: number;
  /** Distinguishes true no-face outcome from unavailable upstream localization. */
  face_detection_status?: string;
  /** Reasons facial analysis was skipped or constrained. */
  analysis_limitations?: string[];
  face_bounding_box?: [number, number, number, number] | null;
  raw_score: number;
  calibrated_probability?: number | null;
  decision_state: string;
  manipulation_type: string;
  localized_regions: Record<string, unknown>[];
  face_conditions: string[];
  disclaimer: string;
}

export interface ELAAnalysisResult {
  raw_score: number;
  mean_error: number;
  variance: number;
  max_local_discrepancy: number;
  suspicious_blocks_count: number;
  has_compression_anomalies: boolean;
  quality_used: number;
  scale_used: number;
}

export interface FFTAnalysisResult {
  raw_score: number;
  high_freq_energy_ratio: number;
  azimuthal_spectral_discrepancy: number;
  grid_peak_prominence: number;
  has_frequency_artifacts: boolean;
}

export interface OODEvaluationResult {
  is_ood: boolean;
  domain_category: string;
  suitability_status: string;
  reasons: string[];
}

export interface NeuralForensicDisagreement {
  disagreement_type: string;
  neural_verdict: string;
  forensic_verdict: string;
  confidence_delta: number;
  synthesis_rationale: string;
  human_review_required: boolean;
  recommended_action: string;
}

export interface NeuralInferenceMetadata {
  status: string;
  model_a_version?: string | null;
  model_b_version?: string | null;
  supported_tasks: string[];
  uncertainty_state: string;
  calibration_status: string;
  warnings: string[];
}

export interface ImageAssessmentResponse {
  assessment_id: string;
  image_id: string | null;
  detector_name: string;
  detector_version: string;
  detection_task: DetectionTask;
  raw_model_score: number;
  is_calibrated: boolean;
  confidence_score: number | null;
  assessment_label: AssessmentLabel;
  uncertainty_state: UncertaintyState;
  decision?: string;
  quality_indicators?: ImageQualityIndicators | null;
  quality_warnings?: string[];
  analysis_eligibility?: boolean;
  reason_for_insufficient_evidence?: string | null;
  recommended_next_step?: string;
  manipulation_regions: ManipulationRegion[];
  preprocessing_warnings: string[];
  explanation: string;
  disagreement_state?: string | null;
  model_agreement_details?: Record<string, unknown> | null;
  visual_explanation?: Record<string, unknown> | null;
  model_limitations: string[];
  metadata_summary: ImageMetadataSummary | null;
  analysis_timestamp: string;
  disclaimer: string;
  // Additive Milestone 4 Multi-Modal Neural and Forensic fields
  is_neural_enabled?: boolean;
  neural_metadata?: NeuralInferenceMetadata | null;
  ai_generated_result?: AIGeneratedResult | null;
  facial_manipulation_result?: FacialManipulationResult | null;
  ela_analysis_result?: ELAAnalysisResult | null;
  fft_analysis_result?: FFTAnalysisResult | null;
  ood_evaluation_result?: OODEvaluationResult | null;
  neural_forensic_disagreement?: NeuralForensicDisagreement | null;
  human_review_guidance?: string | null;
}

export type IntegratedForensicAssessment = ImageAssessmentResponse;

export type AnalysisStage =
  | 'idle'
  | 'validating_image'
  | 'preparing_analysis'
  | 'running_detection'
  | 'preparing_assessment'
  | 'complete'
  | 'error';

export type HumanReviewStatus = 'pending' | 'under_review' | 'completed' | 'cancelled';

export type HumanReviewVerdict =
  | 'concurs_with_ai'
  | 'disagrees_suspicious_manipulation'
  | 'disagrees_no_obvious_artifacts'
  | 'inconclusive'
  | 'insufficient_evidence';

export interface HumanReviewRequest {
  evidence_id: string;
  case_id: string;
  ai_assessment_snapshot: Record<string, unknown>;
  request_reason?: string | null;
}

export interface HumanReviewSubmit {
  human_verdict: HumanReviewVerdict;
  reviewer_notes: string;
  disagreement_rationale?: string | null;
}

export interface HumanReviewResponse {
  id: string;
  evidence_id: string;
  case_id: string;
  requester_id: string;
  reviewer_id: string | null;
  status: HumanReviewStatus;
  ai_assessment_snapshot: Record<string, unknown>;
  request_reason: string | null;
  human_verdict: HumanReviewVerdict | null;
  reviewer_notes: string | null;
  disagreement_rationale: string | null;
  is_disagreement_with_ai: boolean;
  requested_at: string;
  reviewed_at: string | null;
  disclaimer: string;
}
