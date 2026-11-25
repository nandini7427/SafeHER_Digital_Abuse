import React from 'react';

interface ForensicExplanationCardProps {
  explanation: string;
  limitations: string[];
  disclaimer: string;
}

export const ForensicExplanationCard: React.FC<ForensicExplanationCardProps> = ({
  explanation,
  limitations,
  disclaimer,
}) => {
  return (
    <div className="space-y-4">
      {/* Forensic Examination Explanation */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2 mb-3">
          <svg className="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          Technical Analysis Summary
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed">{explanation}</p>
      </div>

      {/* Forensic Model Limitations */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-sm">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2 mb-3">
          <svg className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          Forensic Boundaries & Operational Limitations
        </h3>
        <ul className="space-y-2 text-xs text-slate-400 list-disc list-inside">
          {limitations.map((limitation, idx) => (
            <li key={idx} className="leading-normal">
              {limitation}
            </li>
          ))}
        </ul>
      </div>

      {/* Mandatory Non-Overridable Legal & Safety Disclaimer */}
      <div
        role="region"
        aria-label="Mandatory legal and forensic disclaimer"
        className="bg-amber-950/20 border border-amber-800/40 rounded-xl p-4 text-amber-200/90 text-xs leading-relaxed flex items-start space-x-3"
      >
        <svg className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <div>
          <span className="font-semibold text-amber-300 uppercase tracking-wide block mb-1">
            Important Legal Notice
          </span>
          <p>{disclaimer}</p>
        </div>
      </div>
    </div>
  );
};
