import React, { useRef, useState } from 'react';

interface ImageUploadZoneProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

const ALLOWED_MIME_TYPES = [
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/bmp',
  'image/tiff',
];

const MAX_SIZE_BYTES = 50 * 1024 * 1024; // 50 MB

export const ImageUploadZone: React.FC<ImageUploadZoneProps> = ({
  onFileSelected,
  disabled = false,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndSelectFile = (file: File) => {
    setErrorMessage(null);

    if (file.size === 0) {
      setErrorMessage('The selected file is empty. Please choose a valid image.');
      return;
    }

    if (file.size > MAX_SIZE_BYTES) {
      setErrorMessage('File size exceeds the 50 MB limit. Please select a smaller image.');
      return;
    }

    if (!ALLOWED_MIME_TYPES.includes(file.type.toLowerCase())) {
      setErrorMessage('Unsupported format. Please upload a JPEG, PNG, WEBP, BMP, or TIFF file.');
      return;
    }

    onFileSelected(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSelectFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSelectFile(e.target.files[0]);
    }
  };

  return (
    <div className="w-full">
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Upload evidence image for deepfake and manipulation analysis"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        onKeyDown={(e) => {
          if (!disabled && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault();
            fileInputRef.current?.click();
          }
        }}
        className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 outline-none focus:ring-2 focus:ring-indigo-500 ${
          isDragOver
            ? 'border-indigo-500 bg-indigo-950/20 shadow-lg'
            : 'border-slate-700 hover:border-slate-500 bg-slate-900/60'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp,image/tiff"
          className="hidden"
          disabled={disabled}
          onChange={handleFileInputChange}
        />

        <div className="flex flex-col items-center justify-center space-y-3">
          <div className="p-3 bg-slate-800 rounded-full text-indigo-400">
            <svg
              className="w-8 h-8"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
          </div>

          <div>
            <p className="text-base font-medium text-slate-200">
              Drag and drop evidence image here, or{' '}
              <span className="text-indigo-400 underline underline-offset-2">browse</span>
            </p>
            <p className="text-xs text-slate-400 mt-1">
              Supports JPEG, PNG, WEBP, BMP, and TIFF (up to 50 MB)
            </p>
          </div>

          <div className="flex items-center space-x-2 text-xs text-emerald-400 bg-emerald-950/40 px-3 py-1.5 rounded-full border border-emerald-800/40">
            <svg
              className="w-3.5 h-3.5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
              />
            </svg>
            <span>Original file is preserved and integrity-checked after upload</span>
          </div>
        </div>
      </div>

      {errorMessage && (
        <div
          role="alert"
          className="mt-3 p-3 bg-rose-950/50 border border-rose-800/50 rounded-lg text-rose-300 text-sm flex items-center space-x-2"
        >
          <svg className="w-4 h-4 shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
              clipRule="evenodd"
            />
          </svg>
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
};
