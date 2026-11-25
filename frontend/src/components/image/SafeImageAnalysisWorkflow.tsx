import React, { useEffect, useRef, useState } from 'react';
import { AnalysisStage, ImageAssessmentResponse } from '../../types/imageAnalysis';
import { assessImageEvidence } from '../../services/imageService';
import { ApiError } from '../../services/api';
import { ImageUploadZone } from './ImageUploadZone';
import { ImagePreviewCard } from './ImagePreviewCard';
import { ImageAnalysisStages } from './ImageAnalysisStages';
import { ManipulationCanvasOverlay } from './ManipulationCanvasOverlay';
import { ResultsDashboard } from './ResultsDashboard';
import { ForensicExplanationCard } from './ForensicExplanationCard';
import { CaseAssociationModal } from './CaseAssociationModal';

interface SafeImageAnalysisWorkflowProps {
  initialCaseId?: string;
  onAssessmentCompleted?: (response: ImageAssessmentResponse) => void;
}

export const SafeImageAnalysisWorkflow: React.FC<SafeImageAnalysisWorkflowProps> = ({
  initialCaseId,
  onAssessmentCompleted,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewSrc, setPreviewSrc] = useState<string | null>(null);
  const [stage, setStage] = useState<AnalysisStage>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [assessment, setAssessment] = useState<ImageAssessmentResponse | null>(null);
  const [isCaseModalOpen, setIsCaseModalOpen] = useState(false);
  const [isAssociating, setIsAssociating] = useState(false);
  const [successBanner, setSuccessBanner] = useState<string | null>(null);
  const isRunningRef = useRef(false);
  const isAssociatingRef = useRef(false);

  // Privacy safeguard: automatically revoke blob object URL when previewSrc changes or on unmount
  useEffect(() => {
    return () => {
      if (previewSrc) {
        URL.revokeObjectURL(previewSrc);
      }
    };
  }, [previewSrc]);

  // Handle local file selection
  const handleFileSelected = (file: File) => {
    if (previewSrc) {
      URL.revokeObjectURL(previewSrc);
    }
    setSelectedFile(file);
    setErrorMessage(null);
    setSuccessBanner(null);
    setAssessment(null);
    setStage('idle');

    // Create an object URL for the preview image
    const objectUrl = URL.createObjectURL(file);
    setPreviewSrc(objectUrl);
  };

  // Handle file removal / reset
  const handleRemoveFile = () => {
    if (previewSrc) {
      URL.revokeObjectURL(previewSrc);
    }
    setSelectedFile(null);
    setPreviewSrc(null);
    setAssessment(null);
    setStage('idle');
    setErrorMessage(null);
    setSuccessBanner(null);
  };

  // Execute assessment workflow
  const handleRunAnalysis = async (targetCaseId?: string) => {
    if (!selectedFile || isProcessing || isRunningRef.current) return;
    isRunningRef.current = true;

    setErrorMessage(null);
    setSuccessBanner(null);

    // Advance through the realistic processing stages
    setStage('validating_image');

    // Brief UI yield so user perceives the stage progression naturally
    await new Promise((r) => setTimeout(r, 120));
    setStage('preparing_analysis');

    await new Promise((r) => setTimeout(r, 120));
    setStage('running_detection');

    try {
      const result = await assessImageEvidence(selectedFile, targetCaseId || initialCaseId);
      setStage('preparing_assessment');
      await new Promise((r) => setTimeout(r, 80));

      setAssessment(result);
      setStage('complete');

      if (onAssessmentCompleted) {
        onAssessmentCompleted(result);
      }
    } catch (err: unknown) {
      setStage('error');
      if (err instanceof ApiError) {
        if (err.status === 401) {
          setErrorMessage('Authentication session expired. Please log in to evaluate evidence.');
        } else if (err.status === 403) {
          setErrorMessage('You do not have authorization to associate evidence with this case.');
        } else if (err.status === 404) {
          setErrorMessage('The designated case could not be found.');
        } else if (err.status === 429) {
          setErrorMessage('Inference service rate limit or capacity reached. Please wait before submitting another request.');
        } else if (err.status === 504) {
          setErrorMessage('The forensic analysis request timed out. Please try again with an optimized image.');
        } else {
          setErrorMessage(err.detail || 'Analysis request rejected by server.');
        }
      } else if (err instanceof TypeError && (err.message.includes('fetch') || err.message.includes('Network') || err.message.includes('Failed'))) {
        setErrorMessage('Network connection failure. Please check your internet connection or server availability.');
      } else if (err instanceof Error) {
        if (err.name === 'AbortError' || err.message.toLowerCase().includes('timeout') || err.message.toLowerCase().includes('aborted')) {
          setErrorMessage('The forensic analysis request timed out. Please try again with an optimized image.');
        } else if (err.message.toLowerCase().includes('network') || err.message.toLowerCase().includes('failed to fetch')) {
          setErrorMessage('Network connection failure. Please check your internet connection or server availability.');
        } else {
          setErrorMessage(err.message);
        }
      } else {
        setErrorMessage('An unexpected error occurred during image triage.');
      }
    } finally {
      isRunningRef.current = false;
    }
  };

  // Associate current evidence with an authorized case
  const handleAssociateCase = async (caseId: string) => {
    if (!selectedFile || isAssociating || isAssociatingRef.current) return;
    isAssociatingRef.current = true;
    setIsAssociating(true);

    try {
      const result = await assessImageEvidence(selectedFile, caseId);
      setAssessment(result);
      setIsCaseModalOpen(false);
      setSuccessBanner('Evidence was uploaded, integrity-checked, and associated with your case. Storage encryption depends on your deployment configuration.');
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        if (err.status === 401) {
          setErrorMessage('Authentication session expired. Please log in to save case evidence.');
        } else if (err.status === 403) {
          setErrorMessage('You do not have authorization to associate evidence with this case.');
        } else if (err.status === 404) {
          setErrorMessage('The designated case could not be found.');
        } else {
          setErrorMessage(`Case association failed: ${err.detail || 'Server rejected case update.'}`);
        }
      } else if (err instanceof Error) {
        setErrorMessage(`Case association failed: ${err.message}`);
      } else {
        setErrorMessage('Failed to link evidence to case. Please try again.');
      }
    } finally {
      setIsAssociating(false);
      isAssociatingRef.current = false;
    }
  };

  const isProcessing = [
    'validating_image',
    'preparing_analysis',
    'running_detection',
    'preparing_assessment',
  ].includes(stage);

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6 text-slate-100">
      {/* Workflow Header */}
      <div className="border-b border-slate-800 pb-4">
        <div className="flex items-center space-x-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider mb-1">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          Digital Evidence Forensics
        </div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">
          Visual Manipulation & Deepfake Intelligence
        </h1>
        <p className="text-xs text-slate-400 mt-1 max-w-2xl leading-relaxed">
          Triage suspicious photos, face-swaps, and AI-generated imagery. Identifies Error Level Analysis (ELA)
          compression discrepancies and visual anomalies in an isolated memory sandbox.
        </p>
      </div>

      {/* Success Notification Banner */}
      {successBanner && (
        <div
          role="status"
          aria-live="polite"
          className="p-4 bg-emerald-950/40 border border-emerald-800/60 rounded-xl text-emerald-200 text-xs flex items-center justify-between"
        >
          <div className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-emerald-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            <span>{successBanner}</span>
          </div>
          <button
            onClick={() => setSuccessBanner(null)}
            className="text-emerald-400 hover:text-emerald-300 text-xs font-medium"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Error Notification Banner */}
      {errorMessage && (
        <div
          role="alert"
          aria-live="assertive"
          className="p-4 bg-rose-950/50 border border-rose-800/60 rounded-xl text-rose-200 text-xs flex items-center justify-between"
        >
          <div className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-rose-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>{errorMessage}</span>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-rose-400 hover:text-rose-300 text-xs font-medium"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Step 1 & 2: Image Selection Zone */}
      {!selectedFile && (
        <ImageUploadZone onFileSelected={handleFileSelected} />
      )}

      {/* Step 3, 4 & 5: Preview, File Details & Confirmation */}
      {selectedFile && !assessment && !isProcessing && (
        <div className="space-y-5 animate-fadeIn">
          <ImagePreviewCard
            file={selectedFile}
            onRemove={handleRemoveFile}
            disabled={isProcessing}
          />

          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="text-xs text-slate-400">
              <span className="font-semibold text-slate-200 block mb-0.5">Ready for Analysis</span>
              Click below to initiate non-destructive forensic examination on an isolated working copy.
            </div>

            <button
              type="button"
              disabled={isProcessing}
              aria-busy={isProcessing}
              onClick={() => handleRunAnalysis()}
              className="w-full sm:w-auto px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold text-xs rounded-lg shadow-md transition-all flex items-center justify-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Run Deepfake & Manipulation Scan
            </button>
          </div>
        </div>
      )}

      {/* Step 6: Professional Processing Experience */}
      {isProcessing && (
        <div className="py-6 animate-fadeIn" role="status" aria-live="polite">
          <ImageAnalysisStages currentStage={stage} />
        </div>
      )}

      {/* Step 7, 8 & 9: Results Interface, Overlay & Case Association */}
      {assessment && previewSrc && (
        <div className="space-y-6 animate-fadeIn">
          {/* Spatial Visual Overlay */}
          <ManipulationCanvasOverlay
            imageSrc={previewSrc}
            regions={assessment.manipulation_regions}
            altText={selectedFile?.name || 'Evidence image'}
          />

          {/* Results Metric Breakdown */}
          <ResultsDashboard
            assessment={assessment}
            onOpenCaseAssociation={() => setIsCaseModalOpen(true)}
            onReset={handleRemoveFile}
          />

          {/* Forensic Explanation, Boundaries & Disclaimer */}
          <ForensicExplanationCard
            explanation={assessment.explanation}
            limitations={assessment.model_limitations}
            disclaimer={assessment.disclaimer}
          />
        </div>
      )}

      {/* Case Association Modal */}
      <CaseAssociationModal
        isOpen={isCaseModalOpen}
        onClose={() => setIsCaseModalOpen(false)}
        onConfirm={handleAssociateCase}
        isSubmitting={isAssociating}
      />
    </div>
  );
};
