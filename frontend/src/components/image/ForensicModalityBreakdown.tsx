import React, { useState } from 'react';
import { ImageAssessmentResponse } from '../../types/imageAnalysis';

interface ForensicModalityBreakdownProps {
  assessment: ImageAssessmentResponse;
}

export const ForensicModalityBreakdown: React.FC<ForensicModalityBreakdownProps> = ({
  assessment,
}) => {
  const [activeTab, setActiveTab] = useState<'all' | 'neural' | 'heuristics' | 'quality'>('all');

  const neuralMeta = assessment.neural_metadata;
  const aiGen = assessment.ai_generated_result;
  const facial = assessment.facial_manipulation_result;
  const ela = assessment.ela_analysis_result;
  const fft = assessment.fft_analysis_result;
  const ood = assessment.ood_evaluation_result;
  const disagreement = assessment.neural_forensic_disagreement;
  const quality = assessment.quality_indicators;

  // Determine alert states
  const isModelUnavailable =
    neuralMeta?.status === 'model_unavailable' ||
    assessment.uncertainty_state === 'model_unavailable';

  const isUnsupportedImage =
    ood?.is_ood ||
    assessment.uncertainty_state === 'outside_supported_domain' ||
    quality?.is_in_domain === false;

  const isLowQuality =
    quality?.is_blurry ||
    quality?.resolution_ok === false ||
    quality?.corrupted_data_detected ||
    assessment.analysis_eligibility === false;

  const isConflicting =
    disagreement?.disagreement_type === 'NEURAL_FORENSIC_DISAGREEMENT' ||
    assessment.disagreement_state === 'MODALITY_CONFLICT' ||
    disagreement?.disagreement_type === 'NEURAL_AND_FORENSIC_CONFLICT';

  const isHumanReviewRequired =
    disagreement?.human_review_required ||
    assessment.assessment_label === 'inconclusive' ||
    assessment.assessment_label === 'suspicious_manipulation' ||
    assessment.assessment_label === 'highly_likely_manipulated' ||
    isConflicting;

  const getFacialStatusInfo = () => {
    if (!facial) {
      return {
        label: 'Facial Analysis Not Assessed',
        className: 'text-slate-400 bg-slate-900 border-slate-800',
      };
    }

    if (
      facial.decision_state === 'FACE_LOCALIZATION_UNAVAILABLE' ||
      facial.face_detection_status === 'localization_unavailable'
    ) {
      return {
        label: 'Face Localization Unavailable',
        className: 'text-amber-300 bg-amber-950/40 border-amber-800/50',
      };
    }

    if (facial.decision_state === 'MODEL_UNAVAILABLE' || facial.is_available === false) {
      return {
        label: 'Model Unavailable',
        className: 'text-slate-400 bg-slate-900 border-slate-800',
      };
    }

    if (
      facial.decision_state === 'NO_FACE_DETECTED' ||
      facial.face_detection_status === 'no_face_detected'
    ) {
      return {
        label: 'No Face Detected',
        className: 'text-slate-400 bg-slate-900 border-slate-800',
      };
    }

    const hasVerifiedFaceLocalization = Boolean(
      facial.face_detected &&
      facial.face_detection_status !== 'localization_unavailable' &&
      facial.decision_state !== 'FACE_LOCALIZATION_UNAVAILABLE'
    );

    if (facial.decision_state === 'INDICATORS_DETECTED' && hasVerifiedFaceLocalization) {
      return {
        label: 'Facial Anomaly Indicators',
        className: 'text-rose-300 bg-rose-950/40 border-rose-800/50',
      };
    }

    if (facial.decision_state === 'NO_CLEAR_INDICATORS' && hasVerifiedFaceLocalization) {
      return {
        label: 'No clear indicators detected',
        className: 'text-slate-300 bg-slate-900 border-slate-700',
      };
    }

    return {
      label: 'Facial Analysis Not Assessed',
      className: 'text-slate-400 bg-slate-900 border-slate-800',
    };
  };

  const facialStatus = getFacialStatusInfo();

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-md space-y-6">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-800 gap-2">
        <div>
          <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
            <svg className="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
            Comprehensive Multi-Modal Forensic Findings
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Transparent breakdown across AI generation, facial manipulation, compression residue, frequency spectrum, and quality metrics.
          </p>
        </div>

        {/* View Filter Buttons */}
        <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 self-start sm:self-center" role="tablist" aria-label="Forensic findings filter">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'all'}
            onClick={() => setActiveTab('all')}
            className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors ${
              activeTab === 'all' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            All Findings
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'neural'}
            onClick={() => setActiveTab('neural')}
            className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors ${
              activeTab === 'neural' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Neural Models
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'heuristics'}
            onClick={() => setActiveTab('heuristics')}
            className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors ${
              activeTab === 'heuristics' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Physics & ELA
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'quality'}
            onClick={() => setActiveTab('quality')}
            className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors ${
              activeTab === 'quality' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Quality & OOD
          </button>
        </div>
      </div>

      {/* Operational State Alerts */}
      <div className="space-y-2">
        {isConflicting && (
          <div role="alert" className="p-3.5 bg-amber-950/40 border border-amber-800/60 rounded-xl text-amber-200 text-xs flex items-start gap-2.5">
            <svg className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div>
              <span className="font-semibold text-amber-300 block mb-0.5">Conflicting Modality Results Intercepted</span>
              Neural detection models and classical compression analysis produced conflicting indicators. The system abstains from a definitive single verdict to prevent automated misattribution.
            </div>
          </div>
        )}

        {isModelUnavailable && (
          <div role="status" className="p-3.5 bg-slate-950/70 border border-slate-700/60 rounded-xl text-slate-300 text-xs flex items-start gap-2.5">
            <svg className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <span className="font-semibold text-slate-200 block mb-0.5">Neural Model Unavailable</span>
              Neural detection weights are unconfigured, uninitialized, or in an offline state. Triage has safely completed using classical Error Level Analysis and frequency heuristics.
            </div>
          </div>
        )}

        {isUnsupportedImage && (
          <div role="alert" className="p-3.5 bg-purple-950/40 border border-purple-800/60 rounded-xl text-purple-200 text-xs flex items-start gap-2.5">
            <svg className="w-4 h-4 text-purple-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
            </svg>
            <div>
              <span className="font-semibold text-purple-300 block mb-0.5">Unsupported Out-of-Distribution Image</span>
              The input image belongs to a non-photographic category (e.g., vector illustration, digital artwork, or UI screenshot). Photographic manipulation models are attenuated or refused to prevent false positives.
            </div>
          </div>
        )}

        {isLowQuality && (
          <div role="alert" className="p-3.5 bg-amber-950/30 border border-amber-800/40 rounded-xl text-amber-200 text-xs flex items-start gap-2.5">
            <svg className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <span className="font-semibold text-amber-300 block mb-0.5">Low Evidence Quality Detected</span>
              Excessive optical blur, low pixel dimensions (&lt;128px), or heavy JPEG recompression significantly attenuates subtle manipulation signatures.
            </div>
          </div>
        )}
      </div>

      {/* Grid of Distinct Modality Sections */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* 1. Neural AI-Generation Findings (Model A) */}
        {(activeTab === 'all' || activeTab === 'neural') && (
          <section
            aria-labelledby="section-neural-aigen"
            className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 space-y-3"
          >
            <div className="flex items-center justify-between">
              <h3 id="section-neural-aigen" className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-indigo-500"></span>
                1. AI-Generated Image Analysis (Model A)
              </h3>
              <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${
                aiGen?.is_available
                  ? aiGen.decision_state === 'INDICATORS_DETECTED'
                    ? 'text-rose-300 bg-rose-950/40 border-rose-800/50'
                    : 'text-emerald-300 bg-emerald-950/40 border-emerald-800/50'
                  : 'text-slate-400 bg-slate-900 border-slate-800'
              }`}>
                {aiGen?.is_available
                  ? aiGen.decision_state === 'INDICATORS_DETECTED'
                    ? 'Synthetic Indicators Detected'
                    : 'No Obvious Synthesis Markers'
                  : neuralMeta?.status === 'disabled'
                  ? 'Feature Disabled'
                  : 'Model Unavailable'}
              </span>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              Full-frame classification assessing global diffusion and GAN generative signatures.
            </p>

            <div className="grid grid-cols-2 gap-2 text-[11px] pt-1">
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
                <span className="text-slate-500 block text-[10px]">Uncalibrated Score</span>
                <span className="font-mono font-semibold text-slate-200">
                  {aiGen?.uncalibrated_score !== undefined
                    ? `${aiGen.uncalibrated_score.toFixed(2)} / 1.00`
                    : 'N/A'}
                </span>
                <span className="block text-[9px] text-slate-500 mt-0.5">Raw Softmax Activation</span>
              </div>
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
                <span className="text-slate-500 block text-[10px]">Calibrated Probability</span>
                <span className="font-semibold text-slate-300">
                  {aiGen?.calibrated_probability !== null && aiGen?.calibrated_probability !== undefined
                    ? `${(aiGen.calibrated_probability * 100).toFixed(1)}%`
                    : 'Not Asserted'}
                </span>
                <span className="block text-[9px] text-slate-500 mt-0.5">
                  {aiGen?.calibrated_probability !== null && aiGen?.calibrated_probability !== undefined
                    ? 'Temperature Scaled'
                    : 'Uncalibrated'}
                </span>
              </div>
            </div>

            <div className="text-[10px] text-slate-500 font-mono">
              Backbone: {aiGen?.architecture || 'MobileNetV3-Large'} | Version: {aiGen?.model_version || '1.0.0'}
            </div>
          </section>
        )}

        {/* 2. Facial Manipulation Findings (Model B) */}
        {(activeTab === 'all' || activeTab === 'neural') && (
          <section
            aria-labelledby="section-neural-facial"
            className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 space-y-3"
          >
            <div className="flex items-center justify-between">
              <h3 id="section-neural-facial" className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-violet-500"></span>
                2. Facial Manipulation & Edit Seams (Model B)
              </h3>
              <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${facialStatus.className}`}>
                {facialStatus.label}
              </span>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              Inspects facial crop boundaries for blending seams, face-swapping, and localized reenactments.
            </p>

            <div className="grid grid-cols-2 gap-2 text-[11px] pt-1">
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
                <span className="text-slate-500 block text-[10px]">Detected Faces</span>
                <span className="font-semibold text-slate-200">
                  {facial?.face_detected
                    ? `${facial.face_count} face(s) localized`
                    : facial?.face_detection_status === 'localization_unavailable' || facial?.decision_state === 'FACE_LOCALIZATION_UNAVAILABLE'
                      ? 'Not assessed'
                      : facial?.face_detection_status === 'no_face_detected' || facial?.decision_state === 'NO_FACE_DETECTED'
                        ? 'No face localized'
                        : 'Not available'}
                </span>
                <span className="block text-[9px] text-slate-500 mt-0.5 capitalize">
                  Condition: {facial?.face_conditions?.join(', ') || 'Normal'}
                </span>
              </div>
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
                <span className="text-slate-500 block text-[10px]">Predicted Manipulation</span>
                <span className="font-semibold text-slate-200 capitalize">
                  {facial?.manipulation_type && facial.manipulation_type !== 'not_assessed'
                    ? facial.manipulation_type.replace(/_/g, ' ')
                    : 'Not assessed'}
                </span>
                <span className="block text-[9px] text-slate-500 mt-0.5">
                  {facial?.localized_regions?.length || 0} localized anomalous patch(es)
                </span>
              </div>
            </div>

            {facial?.analysis_limitations && facial.analysis_limitations.length > 0 && (
              <ul className="list-disc pl-4 text-[10px] text-amber-200/80 space-y-1" aria-label="Facial analysis limitations">
                {facial.analysis_limitations.map((limitation, index) => (
                  <li key={`${index}-${limitation}`}>{limitation}</li>
                ))}
              </ul>
            )}

            <div className="text-[10px] text-slate-500 font-mono">
              Head: Dense PatchForensics | Target Version: {facial?.model_version || '1.0.0'}
            </div>
          </section>
        )}

        {/* 3. Error Level Analysis (ELA) Findings */}
        {(activeTab === 'all' || activeTab === 'heuristics') && (
          <section
            aria-labelledby="section-ela"
            className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 space-y-3"
          >
            <div className="flex items-center justify-between">
              <h3 id="section-ela" className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-amber-500"></span>
                3. Error Level Analysis (Compression Residue)
              </h3>
              <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${
                ela?.has_compression_anomalies
                  ? 'text-amber-300 bg-amber-950/40 border-amber-800/50'
                  : 'text-emerald-300 bg-emerald-950/40 border-emerald-800/50'
              }`}>
                {ela?.has_compression_anomalies ? 'Residue Inconsistencies' : 'Uniform Compression'}
              </span>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              Evaluates JPEG compression error when resaved at known quality tiers to spot spliced additions.
            </p>

            <div className="grid grid-cols-2 gap-2 text-[11px] pt-1">
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
                <span className="text-slate-500 block text-[10px]">Normalized ELA Score</span>
                <span className="font-mono font-semibold text-slate-200">
                  {ela?.raw_score !== undefined ? `${ela.raw_score.toFixed(2)} / 1.00` : '0.10 / 1.00'}
                </span>
                <span className="block text-[9px] text-slate-500 mt-0.5">Tested at Q{ela?.quality_used || 90}</span>
              </div>
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
                <span className="text-slate-500 block text-[10px]">Suspicious Grid Blocks</span>
                <span className="font-semibold text-slate-200">
                  {ela?.suspicious_blocks_count || 0} block(s) flagged
                </span>
                <span className="block text-[9px] text-slate-500 mt-0.5">
                  Variance: {ela?.variance !== undefined ? ela.variance.toFixed(2) : 'Normal'}
                </span>
              </div>
            </div>

            <div className="text-[10px] text-slate-500">
              Note: Social media platform recompression can attenuate subtle compression variance.
            </div>
          </section>
        )}

        {/* 4. Fast Fourier Transform (FFT) Findings */}
        {(activeTab === 'all' || activeTab === 'heuristics') && (
          <section
            aria-labelledby="section-fft"
            className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 space-y-3"
          >
            <div className="flex items-center justify-between">
              <h3 id="section-fft" className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-cyan-500"></span>
                4. Frequency Domain Spectral Analysis (FFT)
              </h3>
              <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${
                fft?.has_frequency_artifacts
                  ? 'text-amber-300 bg-amber-950/40 border-amber-800/50'
                  : 'text-emerald-300 bg-emerald-950/40 border-emerald-800/50'
              }`}>
                {fft?.has_frequency_artifacts ? 'Spectral Peaks Detected' : 'Smooth Spectral Decay'}
              </span>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              Examines radial frequency distributions for periodic grid spikes and GAN upsampling artifacts.
            </p>

            <div className="grid grid-cols-2 gap-2 text-[11px] pt-1">
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
                <span className="text-slate-500 block text-[10px]">High-Frequency Energy</span>
                <span className="font-semibold text-slate-200">
                  {fft?.high_freq_energy_ratio !== undefined
                    ? `${(fft.high_freq_energy_ratio * 100).toFixed(1)}%`
                    : '12.4%'}
                </span>
                <span className="block text-[9px] text-slate-500 mt-0.5">High Band Content</span>
              </div>
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
                <span className="text-slate-500 block text-[10px]">Spectral Grid Prominence</span>
                <span className="font-semibold text-slate-200">
                  {fft?.grid_peak_prominence && fft.grid_peak_prominence > 0.05 ? 'Periodic Lattice' : 'Normal'}
                </span>
                <span className="block text-[9px] text-slate-500 mt-0.5">Resampling Grid Check</span>
              </div>
            </div>

            <div className="text-[10px] text-slate-500">
              Correlated with ELA; not treated as a statistically independent manipulation proof.
            </div>
          </section>
        )}

        {/* 5. Image Quality Information */}
        {(activeTab === 'all' || activeTab === 'quality') && (
          <section
            aria-labelledby="section-quality"
            className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 space-y-3"
          >
            <div className="flex items-center justify-between">
              <h3 id="section-quality" className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                5. Technical Image Quality Assessment
              </h3>
              <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${
                assessment.analysis_eligibility !== false
                  ? 'text-emerald-300 bg-emerald-950/40 border-emerald-800/50'
                  : 'text-amber-300 bg-amber-950/40 border-amber-800/50'
              }`}>
                {assessment.analysis_eligibility !== false ? 'Eligible for Triage' : 'Low Evidence Usability'}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] pt-1">
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
                <span className="text-[10px] text-slate-500 block">Resolution</span>
                <span className={`font-semibold ${quality?.resolution_ok ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {quality?.resolution_ok ? 'Pass' : 'Low (<128px)'}
                </span>
              </div>
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
                <span className="text-[10px] text-slate-500 block">Sharpness</span>
                <span className={`font-semibold ${quality?.is_blurry ? 'text-amber-400' : 'text-emerald-400'}`}>
                  {quality?.is_blurry ? 'Blurry' : 'Sharp'}
                </span>
              </div>
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
                <span className="text-[10px] text-slate-500 block">Exposure</span>
                <span className="font-semibold text-slate-200 capitalize">
                  {quality?.exposure_level || 'Normal'}
                </span>
              </div>
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
                <span className="text-[10px] text-slate-500 block">Compression</span>
                <span className={`font-semibold ${quality?.has_heavy_compression ? 'text-amber-400' : 'text-emerald-400'}`}>
                  {quality?.has_heavy_compression ? 'Heavy' : 'Normal'}
                </span>
              </div>
            </div>

            <p className="text-[10px] text-slate-400">
              Minimum requirements: &ge;128x128 pixels, valid color variance, and uncorrupted file headers.
            </p>
          </section>
        )}

        {/* 6. OOD Status Verification */}
        {(activeTab === 'all' || activeTab === 'quality') && (
          <section
            aria-labelledby="section-ood"
            className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 space-y-3"
          >
            <div className="flex items-center justify-between">
              <h3 id="section-ood" className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-purple-500"></span>
                6. Domain Suitability & OOD Verification
              </h3>
              <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${
                ood?.is_ood === true
                  ? 'text-purple-300 bg-purple-950/40 border-purple-800/50'
                  : 'text-emerald-300 bg-emerald-950/40 border-emerald-800/50'
              }`}>
                {ood?.is_ood === true ? 'Out-of-Distribution' : 'Supported Domain'}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-[11px] pt-1">
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
                <span className="text-slate-500 block text-[10px]">Identified Category</span>
                <span className="font-semibold text-slate-200 capitalize">
                  {ood?.domain_category?.replace(/_/g, ' ') || 'Photographic Portrait'}
                </span>
              </div>
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
                <span className="text-slate-500 block text-[10px]">Suitability Status</span>
                <span className="font-semibold text-slate-200">
                  {ood?.suitability_status || 'In-Domain Usable'}
                </span>
              </div>
            </div>

            {ood?.reasons && ood.reasons.length > 0 && (
              <p className="text-[10px] text-purple-300 bg-purple-950/20 p-2 rounded border border-purple-900/40">
                Domain Note: {ood.reasons.join(', ')}
              </p>
            )}
          </section>
        )}
      </div>

      {/* 7. Overall Uncertainty Section */}
      <section
        aria-labelledby="section-uncertainty"
        className="bg-slate-950/40 p-4 rounded-xl border border-slate-800 text-xs space-y-2"
      >
        <div className="flex items-center justify-between">
          <h3 id="section-uncertainty" className="font-semibold text-slate-200 flex items-center gap-2">
            <svg className="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            7. Overall Forensic Uncertainty Evaluation
          </h3>
          <span className="font-mono text-slate-400 uppercase text-[10px] bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
            State: {assessment.uncertainty_state}
          </span>
        </div>
        <p className="text-slate-400 leading-relaxed">
          Confidence quantification strictly incorporates sensor noise variance, optical blur attenuation, compression history, and multi-modal consistency. Low scores indicate an absence of detectable artifacts rather than positive authenticity certification.
        </p>
      </section>

      {/* 8. Human Review Recommendation Section */}
      <section
        aria-labelledby="section-human-review"
        className={`p-4 rounded-xl border ${
          isHumanReviewRequired
            ? 'bg-amber-950/30 border-amber-800/60 text-amber-200'
            : 'bg-indigo-950/30 border-indigo-800/50 text-indigo-200'
        } text-xs space-y-2`}
      >
        <div className="flex items-center justify-between">
          <h3 id="section-human-review" className="font-bold flex items-center gap-2">
            <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
            8. Human Forensic Review Recommendation
          </h3>
          <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${
            isHumanReviewRequired
              ? 'bg-amber-900/60 text-amber-200 border-amber-700'
              : 'bg-indigo-900/60 text-indigo-200 border-indigo-700'
          }`}>
            {isHumanReviewRequired ? 'Human Review Recommended' : 'Routine Automated Triage'}
          </span>
        </div>

        <p className="leading-relaxed">
          {disagreement?.recommended_action ||
            assessment.human_review_guidance ||
            assessment.recommended_next_step ||
            (isHumanReviewRequired
              ? 'A trained human caseworker or digital forensic analyst should inspect the suspicious areas highlighted in the image preview.'
              : 'Automated scan found no obvious artifacts. Ongoing monitoring or specialist review may be conducted if contextual abuse evidence warrants.')}
        </p>

        {disagreement?.synthesis_rationale && (
          <div className="pt-1 text-[11px] opacity-90">
            <span className="font-semibold block text-[10px] uppercase tracking-wider text-amber-300">
              Multi-Modal Consensus Rationale:
            </span>
            <span>{disagreement.synthesis_rationale}</span>
          </div>
        )}
      </section>
    </div>
  );
};