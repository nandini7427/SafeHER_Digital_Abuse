/**
 * Vitest suite for ForensicModalityBreakdown component.
 * Verifies rendering of all 8 distinct forensic sections, tab navigation,
 * operational banners (conflicts, model unavailable, OOD, low quality),
 * plain language safeguards (no fake confidence percentages),
 * and human-review recommendation callouts.
 */

import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ForensicModalityBreakdown } from '../components/image/ForensicModalityBreakdown';
import { ImageAssessmentResponse } from '../types/imageAnalysis';

const baseMockAssessment: ImageAssessmentResponse = {
  assessment_id: 'test-assessment-uuid-1',
  image_id: null,
  detector_name: 'SafeHER-Integrated-Forensic-Pipeline',
  detector_version: '2.0.0',
  detection_task: 'hybrid_manipulation',
  raw_model_score: 0.82,
  is_calibrated: true,
  confidence_score: 0.874,
  assessment_label: 'suspicious_manipulation',
  uncertainty_state: 'moderate',
  manipulation_regions: [],
  preprocessing_warnings: [],
  explanation: 'Multi-modal analysis performed across deep neural networks and forensic frequency heuristics.',
  model_limitations: ['Micro-forensic signals degrade under multiple social-media compressions.'],
  metadata_summary: {
    format: 'JPEG',
    dimensions: [512, 512],
    size_bytes: 45000,
    sha256: 'abc123def456',
    has_exif: false,
    sanitized_exif: {},
    color_mode: 'RGB',
  },
  analysis_timestamp: '2026-11-25T00:00:00Z',
  disclaimer: 'Preliminary automated triage by assistive AI. Not a definitive legal finding.',
  ai_generated_result: {
    is_available: true,
    raw_logits: [1.2, -0.8],
    uncalibrated_score: 0.84,
    calibrated_probability: 0.874,
    decision_state: 'INDICATORS_DETECTED',
    model_version: '1.0.0',
    architecture: 'MobileNetV3-Large',
    disclaimer: 'Model A is an assistive preliminary indicator.',
  },
  facial_manipulation_result: {
    is_available: true,
    raw_score: 0.76,
    calibrated_probability: 0.81,
    decision_state: 'INDICATORS_DETECTED',
    face_detected: true,
    face_count: 1,
    face_conditions: ['Frontal', 'Good lighting'],
    manipulation_type: 'face_swap_boundary',
    localized_regions: [{ x: 50, y: 50, w: 100, h: 100, confidence: 0.78 }],
    model_version: '1.0.0',
    disclaimer: 'Model B localized boundaries are experimental.',
  },
  ela_analysis_result: {
    raw_score: 0.65,
    mean_error: 4.2,
    variance: 42.5,
    max_local_discrepancy: 18.3,
    suspicious_blocks_count: 14,
    has_compression_anomalies: true,
    quality_used: 90,
    scale_used: 20,
  },
  fft_analysis_result: {
    raw_score: 0.58,
    has_frequency_artifacts: true,
    high_freq_energy_ratio: 0.182,
    azimuthal_spectral_discrepancy: 0.12,
    grid_peak_prominence: 0.08,
  },
  ood_evaluation_result: {
    is_ood: false,
    domain_category: 'photographic_portrait',
    suitability_status: 'In-Domain Usable',
    reasons: [],
  },
  quality_indicators: {
    width: 512,
    height: 512,
    megapixels: 0.26,
    aspect_ratio: 1.0,
    blur_score: 180.0,
    compression_artifact_score: 12.0,
    mean_luminance: 128.0,
    overexposed_ratio: 0.01,
    underexposed_ratio: 0.02,
    is_monochromatic_or_flat: false,
    is_screen_capture: false,
    overall_quality_score: 0.85,
    face_count: 1,
    is_in_domain: true,
    resolution_ok: true,
    is_blurry: false,
    exposure_level: 'optimal',
    has_heavy_compression: false,
    corrupted_data_detected: false,
  },
  neural_forensic_disagreement: {
    disagreement_type: 'NEURAL_AND_FORENSIC_AGREEMENT',
    neural_verdict: 'INDICATORS_DETECTED',
    forensic_verdict: 'INDICATORS_DETECTED',
    confidence_delta: 0.05,
    human_review_required: true,
    recommended_action: 'Caseworker review recommended to confirm boundary artifacts.',
    synthesis_rationale: 'Neural and forensic detectors both identify localized anomalies.',
  },
  neural_metadata: {
    status: 'success',
    model_a_version: '1.0.0',
    model_b_version: '1.0.0',
    supported_tasks: ['ai_generated_image', 'face_manipulation'],
    uncertainty_state: 'moderate',
    calibration_status: 'calibrated',
    warnings: [],
  },
};

describe('ForensicModalityBreakdown Component', () => {
  it('renders all 8 distinct forensic modality sections in default all view', () => {
    render(<ForensicModalityBreakdown assessment={baseMockAssessment} />);

    // 1. Neural AI Generation
    expect(screen.getByRole('heading', { name: /1\. AI-Generated Image Analysis \(Model A\)/i })).toBeInTheDocument();
    expect(screen.getByText(/Synthetic Indicators Detected/i)).toBeInTheDocument();
    expect(screen.getByText('0.84 / 1.00')).toBeInTheDocument();
    expect(screen.getByText('87.4%')).toBeInTheDocument();
    expect(screen.getByText(/Temperature Scaled/i)).toBeInTheDocument();

    // 2. Facial Manipulation (Model B)
    expect(screen.getByRole('heading', { name: /2\. Facial Manipulation & Edit Seams \(Model B\)/i })).toBeInTheDocument();
    expect(screen.getByText(/Facial Anomaly Indicators/i)).toBeInTheDocument();
    expect(screen.getByText(/1 face\(s\) localized/i)).toBeInTheDocument();
    expect(screen.getByText(/face swap boundary/i)).toBeInTheDocument();

    // 3. Error Level Analysis (ELA)
    expect(screen.getByRole('heading', { name: /3\. Error Level Analysis \(Compression Residue\)/i })).toBeInTheDocument();
    expect(screen.getByText(/Residue Inconsistencies/i)).toBeInTheDocument();
    expect(screen.getByText(/14 block\(s\) flagged/i)).toBeInTheDocument();

    // 4. Fast Fourier Transform (FFT)
    expect(screen.getByRole('heading', { name: /4\. Frequency Domain Spectral Analysis \(FFT\)/i })).toBeInTheDocument();
    expect(screen.getByText(/Spectral Peaks Detected/i)).toBeInTheDocument();
    expect(screen.getByText('18.2%')).toBeInTheDocument();

    // 5. Technical Image Quality Assessment
    expect(screen.getByRole('heading', { name: /5\. Technical Image Quality Assessment/i })).toBeInTheDocument();
    expect(screen.getByText(/Eligible for Triage/i)).toBeInTheDocument();
    expect(screen.getByText(/Pass/i)).toBeInTheDocument();
    expect(screen.getByText('Sharp')).toBeInTheDocument();

    // 6. Domain Suitability & OOD Verification
    expect(screen.getByRole('heading', { name: /6\. Domain Suitability & OOD Verification/i })).toBeInTheDocument();
    expect(screen.getByText(/Supported Domain/i)).toBeInTheDocument();
    expect(screen.getByText(/photographic portrait/i)).toBeInTheDocument();

    // 7. Overall Forensic Uncertainty Evaluation
    expect(screen.getByRole('heading', { name: /7\. Overall Forensic Uncertainty Evaluation/i })).toBeInTheDocument();
    expect(screen.getByText(/State: moderate/i)).toBeInTheDocument();

    // 8. Human Forensic Review Recommendation
    expect(screen.getByRole('heading', { name: /8\. Human Forensic Review Recommendation/i })).toBeInTheDocument();
    expect(screen.getByText(/Human Review Recommended/i)).toBeInTheDocument();
    expect(screen.getByText(/Caseworker review recommended to confirm boundary artifacts/i)).toBeInTheDocument();
  });

  it('filters visible sections when switching tabs', () => {
    render(<ForensicModalityBreakdown assessment={baseMockAssessment} />);

    // Switch to "Neural Models" tab
    const neuralTab = screen.getByRole('tab', { name: /Neural Models/i });
    fireEvent.click(neuralTab);

    expect(screen.getByRole('heading', { name: /1\. AI-Generated Image Analysis/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /2\. Facial Manipulation/i })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: /3\. Error Level Analysis/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: /4\. Frequency Domain Spectral Analysis/i })).not.toBeInTheDocument();

    // Switch to "Physics & ELA" tab
    const physicsTab = screen.getByRole('tab', { name: /Physics & ELA/i });
    fireEvent.click(physicsTab);

    expect(screen.queryByRole('heading', { name: /1\. AI-Generated Image Analysis/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: /2\. Facial Manipulation/i })).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /3\. Error Level Analysis/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /4\. Frequency Domain Spectral Analysis/i })).toBeInTheDocument();

    // Switch to "Quality & OOD" tab
    const qualityTab = screen.getByRole('tab', { name: /Quality & OOD/i });
    fireEvent.click(qualityTab);

    expect(screen.getByRole('heading', { name: /5\. Technical Image Quality Assessment/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /6\. Domain Suitability & OOD Verification/i })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: /1\. AI-Generated Image Analysis/i })).not.toBeInTheDocument();

    // Switch back to "All Findings" tab
    const allTab = screen.getByRole('tab', { name: /All Findings/i });
    fireEvent.click(allTab);

    expect(screen.getByRole('heading', { name: /1\. AI-Generated Image Analysis/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /3\. Error Level Analysis/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /5\. Technical Image Quality Assessment/i })).toBeInTheDocument();
  });

  it('renders conflicting modality results alert when neural and forensic disagree', () => {
    const conflictingAssessment: ImageAssessmentResponse = {
      ...baseMockAssessment,
      disagreement_state: 'MODALITY_CONFLICT',
      neural_forensic_disagreement: {
        disagreement_type: 'NEURAL_FORENSIC_DISAGREEMENT',
        neural_verdict: 'INDICATORS_DETECTED',
        forensic_verdict: 'NO_CLEAR_INDICATORS',
        confidence_delta: 0.65,
        human_review_required: true,
        recommended_action: 'Human review required due to conflicting neural and ELA indicators.',
        synthesis_rationale: 'Neural model flagged synthetic artifacts while ELA showed uniform compression.',
      },
    };

    render(<ForensicModalityBreakdown assessment={conflictingAssessment} />);

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/Conflicting Modality Results Intercepted/i)).toBeInTheDocument();
    expect(screen.getByText(/The system abstains from a definitive single verdict/i)).toBeInTheDocument();
  });

  it('renders model unavailable state cleanly when neural models are offline', () => {
    const unavailableAssessment: ImageAssessmentResponse = {
      ...baseMockAssessment,
      uncertainty_state: 'model_unavailable',
      neural_metadata: {
        status: 'model_unavailable',
        model_a_version: null,
        model_b_version: null,
        supported_tasks: [],
        uncertainty_state: 'model_unavailable',
        calibration_status: 'uncalibrated',
        warnings: ['Neural weights not configured on server.'],
      },
      ai_generated_result: {
        is_available: false,
        raw_logits: [],
        uncalibrated_score: 0.0,
        calibrated_probability: null,
        decision_state: 'MODEL_UNAVAILABLE',
        model_version: 'none',
        architecture: 'none',
        disclaimer: 'Model unavailable',
      },
      facial_manipulation_result: {
        is_available: false,
        raw_score: 0.0,
        calibrated_probability: null,
        decision_state: 'MODEL_UNAVAILABLE',
        face_detected: false,
        face_count: 0,
        face_conditions: [],
        manipulation_type: 'none',
        localized_regions: [],
        model_version: 'none',
        disclaimer: 'Model unavailable',
      },
    };

    render(<ForensicModalityBreakdown assessment={unavailableAssessment} />);

    expect(screen.getByRole('status')).toBeInTheDocument();
    expect(screen.getByText(/Neural Model Unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/Triage has safely completed using classical Error Level Analysis/i)).toBeInTheDocument();
    expect(screen.getAllByText('Model Unavailable').length).toBeGreaterThanOrEqual(1);
  });

  it('distinguishes unavailable face localization from a verified no-face result', () => {
    const localizationUnavailable: ImageAssessmentResponse = {
      ...baseMockAssessment,
      facial_manipulation_result: {
        is_available: true,
        raw_score: 0,
        calibrated_probability: null,
        decision_state: 'FACE_LOCALIZATION_UNAVAILABLE',
        face_detected: false,
        face_count: 0,
        face_detection_status: 'localization_unavailable',
        analysis_limitations: ['No verified face detector output or annotated face bounding box was supplied.'],
        face_conditions: [],
        manipulation_type: 'not_assessed',
        localized_regions: [],
        model_version: '1.0.0-m4.4-model-b',
        disclaimer: 'Facial analysis was not run.',
      },
    };

    render(<ForensicModalityBreakdown assessment={localizationUnavailable} />);

    expect(screen.getByText('Face Localization Unavailable')).toBeInTheDocument();
    expect(screen.getAllByText('Not assessed')).toHaveLength(2);
    expect(screen.queryByText('No Face in Frame')).not.toBeInTheDocument();
    expect(screen.getByText(/No verified face detector output or annotated face bounding box was supplied/i)).toBeInTheDocument();
  });

  it('renders unsupported out-of-distribution image alert', () => {
    const oodAssessment: ImageAssessmentResponse = {
      ...baseMockAssessment,
      uncertainty_state: 'outside_supported_domain',
      ood_evaluation_result: {
        is_ood: true,
        domain_category: 'digital_vector_art',
        suitability_status: 'Outside Supported Domain',
        reasons: ['No camera sensor pattern', 'Synthetic color distribution'],
      },
    };

    render(<ForensicModalityBreakdown assessment={oodAssessment} />);

    expect(screen.getByText(/Unsupported Out-of-Distribution Image/i)).toBeInTheDocument();
    expect(screen.getByText(/The input image belongs to a non-photographic category/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Out-of-Distribution/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/digital vector art/i)).toBeInTheDocument();
  });

  it('renders low evidence quality alert when image is blurry or small', () => {
    const lowQualityAssessment: ImageAssessmentResponse = {
      ...baseMockAssessment,
      analysis_eligibility: false,
      quality_indicators: {
        ...baseMockAssessment.quality_indicators!,
        is_in_domain: true,
        resolution_ok: false,
        is_blurry: true,
        exposure_level: 'underexposed',
        has_heavy_compression: true,
        corrupted_data_detected: false,
      },
    };

    render(<ForensicModalityBreakdown assessment={lowQualityAssessment} />);

    expect(screen.getByText(/Low Evidence Quality Detected/i)).toBeInTheDocument();
    expect(screen.getByText(/Excessive optical blur, low pixel dimensions/i)).toBeInTheDocument();
    expect(screen.getByText(/Low Evidence Usability/i)).toBeInTheDocument();
    expect(screen.getByText(/Low \(<128px\)/i)).toBeInTheDocument();
    expect(screen.getByText('Blurry')).toBeInTheDocument();
  });

  it('safeguards against fake confidence percentages for uncalibrated scores', () => {
    const uncalibratedAssessment: ImageAssessmentResponse = {
      ...baseMockAssessment,
      ai_generated_result: {
        is_available: true,
        raw_logits: [0.6, -0.4],
        uncalibrated_score: 0.63,
        calibrated_probability: null, // Uncalibrated!
        decision_state: 'NO_CLEAR_INDICATORS',
        model_version: '1.0.0',
        architecture: 'MobileNetV3-Large',
        disclaimer: 'Uncalibrated output',
      },
    };

    render(<ForensicModalityBreakdown assessment={uncalibratedAssessment} />);

    // Should display raw activation, NOT a percentage
    expect(screen.getByText('0.63 / 1.00')).toBeInTheDocument();
    expect(screen.getByText('Raw Softmax Activation')).toBeInTheDocument();
    expect(screen.getByText('Not Asserted')).toBeInTheDocument();
    expect(screen.getByText('Uncalibrated')).toBeInTheDocument();
    // Ensure "63%" is not rendered as calibrated probability
    expect(screen.queryByText('63.0%')).not.toBeInTheDocument();
  });

  it('displays routine triage when no indicators or conflicts detected', () => {
    const routineAssessment: ImageAssessmentResponse = {
      ...baseMockAssessment,
      assessment_label: 'no_obvious_artifacts',
      uncertainty_state: 'low',
      ai_generated_result: {
        is_available: true,
        raw_logits: [-1.2, 1.8],
        uncalibrated_score: 0.12,
        calibrated_probability: 0.14,
        decision_state: 'NO_CLEAR_INDICATORS',
        model_version: '1.0.0',
        architecture: 'MobileNetV3-Large',
        disclaimer: 'Assistive scan',
      },
      facial_manipulation_result: {
        is_available: true,
        raw_score: 0.10,
        calibrated_probability: 0.11,
        decision_state: 'NO_CLEAR_INDICATORS',
        face_detected: true,
        face_count: 1,
        face_conditions: ['Frontal'],
        manipulation_type: 'none',
        localized_regions: [],
        model_version: '1.0.0',
        disclaimer: 'Assistive scan',
      },
      ela_analysis_result: {
        raw_score: 0.15,
        mean_error: 1.1,
        variance: 2.1,
        max_local_discrepancy: 3.4,
        suspicious_blocks_count: 0,
        has_compression_anomalies: false,
        quality_used: 90,
        scale_used: 20,
      },
      fft_analysis_result: {
        raw_score: 0.10,
        high_freq_energy_ratio: 0.05,
        azimuthal_spectral_discrepancy: 0.02,
        grid_peak_prominence: 0.01,
        has_frequency_artifacts: false,
      },
      neural_forensic_disagreement: {
        disagreement_type: 'NEURAL_AND_FORENSIC_AGREEMENT',
        neural_verdict: 'NO_CLEAR_INDICATORS',
        forensic_verdict: 'NO_CLEAR_INDICATORS',
        confidence_delta: 0.02,
        human_review_required: false,
        recommended_action: 'Automated triage found no obvious artifacts.',
        synthesis_rationale: 'All modalities indicate uniform compression and natural sensor features.',
      },
    };

    render(<ForensicModalityBreakdown assessment={routineAssessment} />);

    expect(screen.getByText(/Routine Automated Triage/i)).toBeInTheDocument();
    expect(screen.getByText(/No Obvious Synthesis Markers/i)).toBeInTheDocument();
    expect(screen.getByText(/No clear indicators detected/i)).toBeInTheDocument();
    expect(screen.getByText('Uniform Compression')).toBeInTheDocument();
    expect(screen.getByText(/Smooth Spectral Decay/i)).toBeInTheDocument();
  });
});
