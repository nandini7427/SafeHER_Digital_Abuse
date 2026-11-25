import React, { useEffect, useState } from 'react';
import { Case } from '../../types/case';
import { fetchUserCases } from '../../services/caseService';

interface CaseAssociationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (caseId: string) => void;
  isSubmitting?: boolean;
}

export const CaseAssociationModal: React.FC<CaseAssociationModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  isSubmitting = false,
}) => {
  const [cases, setCases] = useState<Case[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setIsLoading(true);
      setError(null);
      fetchUserCases()
        .then((fetchedCases) => {
          setCases(fetchedCases);
          if (fetchedCases.length > 0) {
            setSelectedCaseId(fetchedCases[0].id);
          }
        })
        .catch(() => {
          setError('Failed to load user cases. Please try again.');
        })
        .finally(() => {
          setIsLoading(false);
        });
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="case-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fadeIn"
    >
      <div className="bg-slate-900 border border-slate-800 rounded-xl max-w-md w-full p-6 shadow-2xl relative">
        <h3 id="case-modal-title" className="text-base font-semibold text-slate-100 flex items-center gap-2">
          <svg className="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          Associate Evidence with Incident Case
        </h3>

        <p className="text-xs text-slate-400 mt-1.5">
          Select an incident case you own. The original file is preserved in private storage, integrity-checked, and linked to the case audit log. This development build does not provide application-level encryption at rest.
        </p>

        <div className="mt-5 space-y-4">
          {isLoading ? (
            <div className="text-center py-6 text-slate-400 text-xs flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin"></span>
              Loading your active cases...
            </div>
          ) : error ? (
            <div className="p-3 bg-rose-950/40 border border-rose-800/40 rounded-lg text-rose-300 text-xs">
              {error}
            </div>
          ) : cases.length === 0 ? (
            <div className="p-4 bg-slate-950/60 border border-slate-800 rounded-lg text-center">
              <p className="text-xs text-slate-300">No active cases found.</p>
              <p className="text-[11px] text-slate-500 mt-1">
                You can create a case in the Case Management section before linking evidence.
              </p>
            </div>
          ) : (
            <div>
              <label htmlFor="case-select" className="block text-xs font-medium text-slate-300 mb-1.5">
                Target Incident Case
              </label>
              <select
                id="case-select"
                value={selectedCaseId}
                onChange={(e) => setSelectedCaseId(e.target.value)}
                disabled={isSubmitting}
                className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:ring-2 focus:ring-indigo-500 focus:outline-none"
              >
                {cases.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.title} (Status: {c.status})
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        <div className="mt-6 flex items-center justify-end space-x-3 pt-3 border-t border-slate-800">
          <button
            type="button"
            onClick={onClose}
            disabled={isSubmitting}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium rounded-lg transition-colors"
          >
            Cancel
          </button>

          <button
            type="button"
            onClick={() => selectedCaseId && onConfirm(selectedCaseId)}
            disabled={isSubmitting || isLoading || cases.length === 0}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-semibold rounded-lg shadow transition-colors flex items-center gap-1.5"
          >
            {isSubmitting ? (
              <>
                <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                Preserving...
              </>
            ) : (
              'Save & Associate'
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
