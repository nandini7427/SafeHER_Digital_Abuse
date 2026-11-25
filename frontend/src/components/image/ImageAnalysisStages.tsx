import React from 'react';
import { AnalysisStage } from '../../types/imageAnalysis';

interface ImageAnalysisStagesProps {
  currentStage: AnalysisStage;
}

interface StageStep {
  id: AnalysisStage;
  label: string;
  description: string;
}

const STAGES: StageStep[] = [
  {
    id: 'validating_image',
    label: 'Validating Image',
    description: 'Verifying binary signatures, dimensions, and decompression safety',
  },
  {
    id: 'preparing_analysis',
    label: 'Preparing Analysis',
    description: 'Normalizing color space, orientation, and isolating working copy',
  },
  {
    id: 'running_detection',
    label: 'Running Detection',
    description: 'Analyzing Error Level Analysis (ELA) residuals and compression metrics',
  },
  {
    id: 'preparing_assessment',
    label: 'Preparing Assessment',
    description: 'Generating explainability summary and uncertainty triage',
  },
];

export const ImageAnalysisStages: React.FC<ImageAnalysisStagesProps> = ({ currentStage }) => {
  const getStageStatus = (stageId: AnalysisStage): 'pending' | 'in_progress' | 'done' => {
    const stageOrder: AnalysisStage[] = [
      'validating_image',
      'preparing_analysis',
      'running_detection',
      'preparing_assessment',
      'complete',
    ];

    const currentIndex = stageOrder.indexOf(currentStage);
    const targetIndex = stageOrder.indexOf(stageId);

    if (currentIndex > targetIndex || currentStage === 'complete') {
      return 'done';
    }
    if (currentIndex === targetIndex) {
      return 'in_progress';
    }
    return 'pending';
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl w-full max-w-xl mx-auto">
      <div className="text-center mb-6">
        <h3 className="text-base font-semibold text-slate-100 flex items-center justify-center gap-2">
          <svg className="w-5 h-5 text-indigo-400 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          Processing Visual Evidence
        </h3>
        <p className="text-xs text-slate-400 mt-1">
          Executing forensic checks within an isolated memory sandbox
        </p>
      </div>

      <div className="space-y-4">
        {STAGES.map((stage) => {
          const status = getStageStatus(stage.id);

          return (
            <div
              key={stage.id}
              className={`flex items-start gap-3.5 p-3 rounded-lg transition-colors ${
                status === 'in_progress'
                  ? 'bg-indigo-950/40 border border-indigo-800/40'
                  : status === 'done'
                  ? 'bg-slate-800/30'
                  : 'opacity-40'
              }`}
            >
              <div className="mt-0.5 shrink-0">
                {status === 'done' ? (
                  <div className="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 flex items-center justify-center">
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                ) : status === 'in_progress' ? (
                  <div className="w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 border border-indigo-500/40 flex items-center justify-center">
                    <span className="w-2 h-2 rounded-full bg-indigo-400 animate-ping"></span>
                  </div>
                ) : (
                  <div className="w-5 h-5 rounded-full border border-slate-700 bg-slate-800 flex items-center justify-center">
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-600"></span>
                  </div>
                )}
              </div>

              <div>
                <h4
                  className={`text-sm font-medium ${
                    status === 'in_progress'
                      ? 'text-indigo-200'
                      : status === 'done'
                      ? 'text-slate-200'
                      : 'text-slate-500'
                  }`}
                >
                  {stage.label}
                </h4>
                <p className="text-xs text-slate-400 mt-0.5">{stage.description}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
