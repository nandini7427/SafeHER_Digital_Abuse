import React, { useState } from 'react';
import { ManipulationRegion } from '../../types/imageAnalysis';

interface ManipulationCanvasOverlayProps {
  imageSrc: string;
  regions: ManipulationRegion[];
  altText?: string;
}

export const ManipulationCanvasOverlay: React.FC<ManipulationCanvasOverlayProps> = ({
  imageSrc,
  regions,
  altText = 'Evaluated visual evidence',
}) => {
  const [showOverlay, setShowOverlay] = useState(true);
  const [isBlurred, setIsBlurred] = useState(false);
  const [activeRegionIndex, setActiveRegionIndex] = useState<number | null>(null);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg">
      <div className="p-3.5 bg-slate-950/60 border-b border-slate-800/80 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Forensic Visual Overlay
          </span>
          {regions.length > 0 && (
            <span className="text-[11px] bg-amber-500/20 text-amber-300 border border-amber-500/30 px-2 py-0.5 rounded-full font-mono">
              {regions.length} anomaly region{regions.length > 1 ? 's' : ''}
            </span>
          )}
        </div>

        <div className="flex items-center space-x-3 text-xs">
          {regions.length > 0 && (
            <label className="flex items-center space-x-1.5 cursor-pointer text-slate-300 select-none">
              <input
                type="checkbox"
                checked={showOverlay}
                onChange={(e) => setShowOverlay(e.target.checked)}
                className="rounded bg-slate-800 border-slate-700 text-indigo-600 focus:ring-indigo-500 h-3.5 w-3.5"
              />
              <span>Highlight Anomalies</span>
            </label>
          )}

          <button
            type="button"
            onClick={() => setIsBlurred(!isBlurred)}
            className="text-slate-400 hover:text-slate-200 transition-colors flex items-center gap-1"
            title={isBlurred ? 'Reveal image' : 'Blur image for safety'}
          >
            {isBlurred ? 'Reveal' : 'Blur'}
          </button>
        </div>
      </div>

      {/* Interactive Canvas / Image Viewer */}
      <div className="relative w-full max-h-[460px] overflow-hidden flex items-center justify-center bg-black/40">
        <img
          src={imageSrc}
          alt={altText}
          className={`max-h-[460px] w-auto max-w-full object-contain select-none transition-all duration-200 ${
            isBlurred ? 'blur-xl scale-105' : ''
          }`}
        />

        {/* Spatial Anomaly Boxes Overlay */}
        {showOverlay && !isBlurred && regions.length > 0 && (
          <div className="absolute inset-0 pointer-events-none">
            {regions.map((region, idx) => {
              const [ymin, xmin, ymax, xmax] = region.box_normalized;
              const top = `${ymin * 100}%`;
              const left = `${xmin * 100}%`;
              const width = `${(xmax - xmin) * 100}%`;
              const height = `${(ymax - ymin) * 100}%`;
              const isActive = activeRegionIndex === idx;

              return (
                <div
                  key={idx}
                  style={{ top, left, width, height }}
                  className={`absolute pointer-events-auto cursor-pointer border-2 transition-all duration-150 ${
                    isActive
                      ? 'border-amber-300 bg-amber-400/30 z-20 shadow-lg'
                      : 'border-amber-500/80 bg-amber-500/15 hover:border-amber-400 z-10'
                  }`}
                  onMouseEnter={() => setActiveRegionIndex(idx)}
                  onMouseLeave={() => setActiveRegionIndex(null)}
                  onClick={() => setActiveRegionIndex(isActive ? null : idx)}
                >
                  <span className="absolute top-1 left-1 text-[10px] font-mono bg-black/80 text-amber-300 px-1 py-0.5 rounded leading-none">
                    # {idx + 1} (Score: {region.anomaly_score.toFixed(2)})
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Active Region Tooltip / Explanation Bar */}
      {activeRegionIndex !== null && regions[activeRegionIndex] && (
        <div className="p-3 bg-amber-950/40 border-t border-amber-800/40 text-xs text-amber-200 flex items-start space-x-2 animate-fadeIn">
          <svg className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div>
            <span className="font-semibold">
              Region #{activeRegionIndex + 1} [
              {regions[activeRegionIndex].attribution_category === 'model_attention_attribution'
                ? 'Model Attention Attribution (Not Proof of Manipulation)'
                : 'Suspected Compression Anomaly'}
              ]:{' '}
            </span>
            {regions[activeRegionIndex].description}
          </div>
        </div>
      )}
    </div>
  );
};
