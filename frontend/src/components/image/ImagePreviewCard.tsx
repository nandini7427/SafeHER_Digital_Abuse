import React, { useEffect, useState } from 'react';

interface ImagePreviewCardProps {
  file: File;
  onRemove: () => void;
  disabled?: boolean;
}

export const ImagePreviewCard: React.FC<ImagePreviewCardProps> = ({
  file,
  onRemove,
  disabled = false,
}) => {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isBlurred, setIsBlurred] = useState(false);

  useEffect(() => {
    // Generate isolated object URL
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    // Memory-safe cleanup: Revoke URL immediately on unmount or file change
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-md">
      <div className="flex flex-col sm:flex-row items-center gap-5">
        {/* Preview Container with Blur Capability */}
        <div className="relative w-36 h-36 rounded-lg overflow-hidden bg-slate-950 border border-slate-800 shrink-0 flex items-center justify-center">
          {previewUrl ? (
            <img
              src={previewUrl}
              alt="Evidence thumbnail preview"
              className={`w-full h-full object-cover transition-all duration-300 ${
                isBlurred ? 'blur-lg scale-110' : ''
              }`}
            />
          ) : (
            <div className="text-slate-600 text-xs">Loading...</div>
          )}

          {/* Sensitivity blur overlay badge */}
          <button
            type="button"
            onClick={() => setIsBlurred(!isBlurred)}
            title={isBlurred ? 'Reveal image' : 'Blur sensitive image'}
            aria-label={isBlurred ? 'Reveal image' : 'Blur sensitive image'}
            className="absolute bottom-1.5 right-1.5 p-1.5 bg-slate-950/80 hover:bg-slate-900 text-slate-300 rounded-md backdrop-blur-sm transition-colors text-xs flex items-center gap-1"
          >
            {isBlurred ? (
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            ) : (
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l18 18" />
              </svg>
            )}
          </button>
        </div>

        {/* File Details & Actions */}
        <div className="flex-1 w-full flex flex-col justify-between">
          <div>
            <div className="flex items-start justify-between gap-2">
              <h4 className="text-sm font-semibold text-slate-100 truncate max-w-[240px] sm:max-w-xs" title={file.name}>
                {file.name}
              </h4>
              <span className="text-[11px] font-mono uppercase bg-slate-800 text-slate-300 px-2 py-0.5 rounded border border-slate-700">
                {file.type.split('/')[1] || 'IMAGE'}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 mt-2 text-xs text-slate-400">
              <div>
                <span className="text-slate-500">Size: </span>
                <span className="font-mono text-slate-300">{formatFileSize(file.size)}</span>
              </div>
              <div>
                <span className="text-slate-500">MIME: </span>
                <span className="font-mono text-slate-300">{file.type}</span>
              </div>
            </div>
          </div>

          <div className="mt-4 flex items-center justify-between border-t border-slate-800/80 pt-3">
            <span className="text-[11px] text-slate-400 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
              Client validation passed
            </span>

            <button
              type="button"
              disabled={disabled}
              onClick={onRemove}
              className="text-xs text-rose-400 hover:text-rose-300 disabled:opacity-50 transition-colors flex items-center gap-1 font-medium"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
              Change file
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
