import React, { useState } from 'react';
import { Copy, Check, GitCompare, ArrowRight } from 'lucide-react';

export default function DiffViewer({ originalText, correctedText, diffChunks }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (correctedText) {
      navigator.clipboard.writeText(correctedText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const isIdentical = originalText.trim() === correctedText.trim();

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
      <div className="px-5 py-3 border-b border-slate-100 bg-slate-50/50 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <GitCompare className="w-4 h-4 text-emerald-600" />
          <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
            Correction & Visual Diff
          </span>
        </div>

        <button
          onClick={handleCopy}
          className="flex items-center space-x-1 px-2.5 py-1 text-xs text-slate-600 hover:text-emerald-700 hover:bg-emerald-50 rounded-lg transition-colors border border-slate-200"
          title="Copy corrected text"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-600" />
              <span className="text-emerald-600 font-medium">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      <div className="p-5 space-y-4">
        {/* Full Corrected Sentence Highlight */}
        <div>
          <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
            Recommended Correction
          </label>
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-slate-800 text-base font-medium leading-relaxed">
            {correctedText}
          </div>
        </div>

        {/* Visual Diff View */}
        <div>
          <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
            Word-Level Edit Diff
          </label>
          <div className="p-3.5 rounded-xl bg-white border border-slate-200 text-base leading-relaxed">
            {isIdentical ? (
              <span className="text-emerald-600 font-medium flex items-center gap-1.5">
                <Check className="w-4 h-4" /> No edits required — sentence is grammatically sound!
              </span>
            ) : (
              <div className="flex flex-wrap items-center gap-1">
                {diffChunks && diffChunks.map((chunk, idx) => {
                  if (chunk.type === 'delete') {
                    return (
                      <span
                        key={idx}
                        className="bg-rose-100 text-rose-800 line-through px-1.5 py-0.5 rounded text-sm font-medium border border-rose-200"
                        title="Removed original error"
                      >
                        {chunk.text}
                      </span>
                    );
                  }
                  if (chunk.type === 'insert') {
                    return (
                      <span
                        key={idx}
                        className="bg-emerald-100 text-emerald-900 font-semibold px-1.5 py-0.5 rounded text-sm border border-emerald-300"
                        title="Added correction"
                      >
                        {chunk.text}
                      </span>
                    );
                  }
                  return (
                    <span key={idx} className="text-slate-700 text-sm">
                      {chunk.text}
                    </span>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
