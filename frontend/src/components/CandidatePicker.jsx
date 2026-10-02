import React from 'react';
import { Layers, CheckCircle2, ChevronRight } from 'lucide-react';

export default function CandidatePicker({ candidates, selectedCandidate, onSelectCandidate }) {
  if (!candidates || candidates.length <= 1) {
    return null;
  }

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
      <div className="px-5 py-3 border-b border-slate-100 bg-slate-50/50 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Layers className="w-4 h-4 text-purple-600" />
          <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
            Alternative Candidates ({candidates.length})
          </span>
        </div>
        <span className="text-[11px] text-slate-400">Beam Search Rank & Confidence</span>
      </div>

      <div className="p-4 space-y-2.5">
        {candidates.map((cand, idx) => {
          const isSelected = selectedCandidate?.candidate_text === cand.candidate_text || (!selectedCandidate && cand.is_selected);

          return (
            <button
              key={idx}
              onClick={() => onSelectCandidate(cand)}
              className={`w-full text-left p-3 rounded-xl border transition-all flex items-center justify-between gap-3 ${
                isSelected
                  ? 'bg-purple-50/60 border-purple-300 ring-2 ring-purple-500/10'
                  : 'bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50'
              }`}
            >
              <div className="flex items-start space-x-3">
                <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                  isSelected ? 'bg-purple-600 text-white' : 'bg-slate-100 text-slate-600'
                }`}>
                  #{cand.rank}
                </span>
                <div>
                  <p className="text-sm font-medium text-slate-800 leading-snug">
                    "{cand.candidate_text}"
                  </p>
                  <div className="flex items-center space-x-3 mt-1 text-[11px] text-slate-500">
                    <span>Confidence: <strong className="text-slate-700">{(cand.confidence * 100).toFixed(1)}%</strong></span>
                    <span>•</span>
                    <span>Edits: {cand.edit_distance}</span>
                  </div>
                </div>
              </div>

              <div className="shrink-0">
                {isSelected ? (
                  <CheckCircle2 className="w-5 h-5 text-purple-600" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-slate-400" />
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
