import React from 'react';
import {
  AssessmentLabel,
  ImageAssessmentResponse,
  UncertaintyState,
} from '../../types/imageAnalysis';
import { ForensicModalityBreakdown } from './ForensicModalityBreakdown';

interface ResultsDashboardProps {
  assessment: ImageAssessmentResponse;
  onOpenCaseAssociation: () => void;
  onReset: () => void;
}

export const ResultsDashboard: React.FC<ResultsDashboardProps> = ({
  assessment,
  onOpenCaseAssociation,
  onReset,
}) => {
  const getLabelBadge = (label: AssessmentLabel) => {
    switch (label) {
      case 'highly_likely_manipulated':
        return {
          title: 'Multiple Anomalies Detected',
          subtitle: 'Elevated compression and spectral variance — human review recommended',
          bg: 'bg-rose-950/40 border-rose-800 text-rose-300',
          dot: 'bg-rose-500',
        };
      case 'suspicious_manipulation':
        return {
          title: 'Suspicious Anomalies Detected',
          subtitle: 'Moderate compression variance detected across quadrants — human review recommended',
          bg: 'bg-amber-950/40 border-amber-800 text-amber-300',
          dot: 'bg-amber-500',
        };
      case 'no_obvious_artifacts':
        return {
          title: 'No Obvious Artifacts Detected',
          subtitle: 'Does NOT confirm authenticity; modern AI may leave no trace',
          bg: 'bg-emerald-950/40 border-emerald-800 text-emerald-300',
          dot: 'bg-emerald-500',
        };
      case 'inconclusive':
      default:
        return {
          title: 'Inconclusive Analysis',
          subtitle: 'Image resolution or quality is insufficient for triage',
          bg: 'bg-slate-800/80 border-slate-700 text-slate-300',
          dot: 'bg-slate-500',
        };
    }
  };

  const getUncertaintyBadge = (uncertainty: UncertaintyState) => {
    switch (uncertainty) {
      case 'analysis_usable_evidence':
      case 'low':
        return { text: 'Usable Evidence (Low Uncertainty)', color: 'text-emerald-400 bg-emerald-950/40 border-emerald-800/40' };
      case 'model_uncertain':
      case 'moderate':
        return { text: 'Moderate Uncertainty', color: 'text-amber-400 bg-amber-950/40 border-amber-800/40' };
      case 'high':
        return { text: 'High Uncertainty', color: 'text-rose-400 bg-rose-950/40 border-rose-800/40' };
      case 'analysis_poor_quality_evidence':
      case 'insufficient_quality':
        return { text: 'Poor Evidence Quality', color: 'text-amber-300 bg-amber-950/50 border-amber-800/50' };
      case 'outside_supported_domain':
        return { text: 'Outside Supported Domain', color: 'text-purple-400 bg-purple-950/40 border-purple-800/40' };
      case 'model_unavailable':
        return { text: 'Model Unavailable', color: 'text-rose-300 bg-rose-950/40 border-rose-800/40' };
      case 'analysis_cannot_be_completed':
        return { text: 'Analysis Cannot Be Completed', color: 'text-red-400 bg-red-950/60 border-red-800/60' };
      default:
        return { text: 'Insufficient Quality', color: 'text-slate-400 bg-slate-800 border-slate-700' };
    }
  };

  const badge = getLabelBadge(assessment.assessment_label);
  const uncertaintyInfo = getUncertaintyBadge(assessment.uncertainty_state);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-md space-y-6">
      {/* Header Verdict Banner */}
      <div className={`p-4 rounded-xl border ${badge.bg} flex flex-col sm:flex-row sm:items-center justify-between gap-3`}>
        <div className="flex items-start gap-3">
          <span className={`w-3 h-3 rounded-full mt-1.5 shrink-0 ${badge.dot}`}></span>
          <div>
            <h2 className="text-lg font-bold tracking-tight">{badge.title}</h2>
            <p className="text-xs opacity-90 mt-0.5">{badge.subtitle}</p>
          </div>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-center">
          <span className={`text-[11px] font-medium px-2.5 py-1 rounded-full border ${uncertaintyInfo.color}`}>
            {uncertaintyInfo.text}
          </span>
        </div>
      </div>

      {/* Metric Breakdown Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
          <span className="text-[11px] text-slate-400 block">Raw Model Score</span>
          <div className="flex items-baseline gap-1.5 mt-1">
            <span className="text-xl font-bold font-mono text-slate-100">
              {assessment.raw_model_score.toFixed(2)}
            </span>
            <span className="text-[10px] text-slate-500">/ 1.00</span>
          </div>
          <span className="inline-block mt-1 text-[10px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded font-medium">
            Uncalibrated Heuristic
          </span>
        </div>

        <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
          <span className="text-[11px] text-slate-400 block">Calibrated Confidence</span>
          <div className="mt-1">
            <span className="text-sm font-semibold text-slate-400">
              {assessment.is_calibrated && assessment.confidence_score !== null
                ? `${(assessment.confidence_score * 100).toFixed(1)}%`
                : 'Not Applicable'}
            </span>
          </div>
          <span className="inline-block mt-1 text-[10px] text-slate-500">
            {assessment.is_calibrated ? 'Empirically Calibrated' : 'Calibration Not Asserted'}
          </span>
        </div>

        <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
          <span className="text-[11px] text-slate-400 block">Detector Engine</span>
          <div className="mt-1">
            <span className="text-xs font-semibold text-indigo-300 truncate block" title={assessment.detector_name}>
              {assessment.detector_name}
            </span>
          </div>
          <span className="inline-block mt-1 text-[10px] text-slate-500 font-mono">
            v{assessment.detector_version}
          </span>
        </div>

        <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
          <span className="text-[11px] text-slate-400 block">Evidence ID</span>
          <div className="mt-1">
            <span className="text-xs font-mono text-slate-300 truncate block">
              {assessment.image_id ? assessment.image_id.slice(0, 8) + '...' : 'Unlinked Preview'}
            </span>
          </div>
          <span className="inline-block mt-1 text-[10px] text-emerald-400">
            {assessment.image_id ? 'Preserved in Vault' : 'Not Associated'}
          </span>
        </div>
      </div>

      {/* Technical Forensic & Audit Trail Metadata */}
      <div className="bg-slate-950/40 p-3.5 rounded-lg border border-slate-800/80 text-xs">
        <span className="text-slate-400 font-medium block mb-2">Forensic Assessment & Audit Trail:</span>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 font-mono text-slate-300">
          <div>
            <span className="text-slate-500">Assessment ID: </span>
            <span className="truncate inline-block max-w-[180px] align-bottom text-indigo-300" title={assessment.assessment_id}>
              {assessment.assessment_id.slice(0, 8)}...
            </span>
          </div>
          <div>
            <span className="text-slate-500">Detection Task: </span>
            <span className="text-slate-300 capitalize">{assessment.detection_task.replace(/_/g, ' ')}</span>
          </div>
          <div>
            <span className="text-slate-500">Timestamp: </span>
            <span className="text-slate-300">{new Date(assessment.analysis_timestamp).toUTCString()}</span>
          </div>
          {assessment.metadata_summary && (
            <>
              <div>
                <span className="text-slate-500">SHA-256: </span>
                <span className="truncate inline-block max-w-[180px] align-bottom" title={assessment.metadata_summary.sha256}>
                  {assessment.metadata_summary.sha256.slice(0, 16)}...
                </span>
              </div>
              <div>
                <span className="text-slate-500">Dimensions: </span>
                {assessment.metadata_summary.dimensions[0]} x {assessment.metadata_summary.dimensions[1]} px
              </div>
              <div>
                <span className="text-slate-500">Format / Mode: </span>
                {assessment.metadata_summary.format} ({assessment.metadata_summary.color_mode})
              </div>
            </>
          )}
          <div>
            <span className="text-slate-500">Localization Provenance: </span>
            <span className="text-slate-300">
              {assessment.visual_explanation?.localization_method
                ? String(assessment.visual_explanation.localization_method)
                : assessment.manipulation_regions.length > 0
                ? 'Forensic ELA Decomposition'
                : 'Not Triggered / No Anomalies'}
            </span>
          </div>
        </div>
      </div>

      {/* Recommended Next Step Banner */}
      {assessment.recommended_next_step && (
        <div className="bg-indigo-950/30 border border-indigo-800/40 p-3.5 rounded-lg flex items-start gap-3">
          <div className="p-1 bg-indigo-900/50 rounded text-indigo-400 mt-0.5 shrink-0">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div>
            <span className="text-[11px] font-semibold text-indigo-300 block">Recommended Next Step:</span>
            <p className="text-xs text-slate-300 mt-0.5 leading-relaxed">{assessment.recommended_next_step}</p>
          </div>
        </div>
      )}

      {/* Forensic Quality Indicators & Integrity Diagnostics */}
      {assessment.quality_indicators && (
        <div className="bg-slate-950/50 p-4 rounded-lg border border-slate-800 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <svg className="w-3.5 h-3.5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Image Quality & Forensic Suitability Assessment
            </span>
            <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${
              assessment.analysis_eligibility
                ? 'text-emerald-400 bg-emerald-950/40 border-emerald-800/40'
                : 'text-amber-400 bg-amber-950/40 border-amber-800/40'
            }`}>
              {assessment.analysis_eligibility ? 'Eligible for Triage' : 'Low Evidence Usability'}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
            <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
              <span className="text-[10px] text-slate-500 block">Resolution Check</span>
              <span className={`font-semibold ${assessment.quality_indicators.resolution_ok ? 'text-emerald-400' : 'text-amber-400'}`}>
                {assessment.quality_indicators.resolution_ok ? 'Pass (>=128px)' : 'Low (<128px)'}
              </span>
            </div>
            <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
              <span className="text-[10px] text-slate-500 block">Sharpness / Blur</span>
              <span className={`font-semibold ${assessment.quality_indicators.is_blurry ? 'text-amber-400' : 'text-emerald-400'}`}>
                {assessment.quality_indicators.is_blurry ? 'Excessive Blur' : 'Sharp'}
              </span>
            </div>
            <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
              <span className="text-[10px] text-slate-500 block">Exposure Status</span>
              <span className={`font-semibold capitalize ${
                assessment.quality_indicators.exposure_level === 'normal' ? 'text-emerald-400' : 'text-amber-400'
              }`}>
                {assessment.quality_indicators.exposure_level}
              </span>
            </div>
            <div className="bg-slate-900/80 p-2 rounded border border-slate-800/60">
              <span className="text-[10px] text-slate-500 block">Compression Artifacts</span>
              <span className={`font-semibold ${assessment.quality_indicators.has_heavy_compression ? 'text-amber-400' : 'text-emerald-400'}`}>
                {assessment.quality_indicators.has_heavy_compression ? 'Heavy Blockiness' : 'Normal'}
              </span>
            </div>
          </div>

          {/* Quality Warnings List */}
          {assessment.quality_warnings && assessment.quality_warnings.length > 0 && (
            <div className="space-y-1.5 pt-1">
              <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">Quality Warnings:</span>
              <div className="space-y-1">
                {assessment.quality_warnings.map((w, idx) => (
                  <div key={idx} className="flex items-start gap-1.5 text-[11px] text-amber-300/90 bg-amber-950/20 px-2 py-1 rounded border border-amber-900/30">
                    <span className="text-amber-500 shrink-0">•</span>
                    <span>{w}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Comprehensive Multi-Modal Forensic Findings (8 Distinct Sections) */}
      <ForensicModalityBreakdown assessment={assessment} />

      {/* Workflow Actions */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
        <button
          type="button"
          onClick={onReset}
          className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg transition-colors flex items-center gap-1.5"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Analyze Another Image
        </button>

        {!assessment.image_id && (
          <button
            type="button"
            onClick={onOpenCaseAssociation}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg shadow transition-colors flex items-center gap-1.5"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
            </svg>
            Associate with Incident Case
          </button>
        )}
      </div>
    </div>
  );
};
