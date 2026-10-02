import React from 'react';
import { AlertTriangle, CheckCircle, ShieldCheck, HelpCircle, ArrowRight } from 'lucide-react';

export default function ErrorList({ detectedErrors, onApplyErrorFix }) {
  if (!detectedErrors || detectedErrors.length === 0) {
    return (
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 text-center">
        <div className="w-12 h-12 rounded-full bg-emerald-50 text-emerald-600 flex items-center justify-center mx-auto mb-3">
          <CheckCircle className="w-6 h-6" />
        </div>
        <h4 className="text-sm font-semibold text-slate-900">Zero Grammatical Errors Detected</h4>
        <p className="text-xs text-slate-500 mt-1">The sentence appears grammatically sound and well-formed.</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
      <div className="px-5 py-3 border-b border-slate-100 bg-slate-50/50 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 text-amber-500" />
          <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
            Detected Errors ({detectedErrors.length})
          </span>
        </div>
        <span className="text-xs text-slate-400">Localized spans with verified tiers</span>
      </div>

      <div className="p-4 space-y-3">
        {detectedErrors.map((err, idx) => {
          const isDirect = err.verification_tier === 'DIRECTLY_IDENTIFIED';

          return (
            <div
              key={err.error_id || idx}
              className="p-4 rounded-xl border border-slate-200 bg-white hover:border-slate-300 transition-all shadow-xs"
            >
              <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-sm text-slate-900">{err.category}</span>
                  
                  {/* Tier Badge */}
                  {isDirect ? (
                    <span
                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-blue-50 text-blue-700 border border-blue-200"
                      title="Directly verified by syntactic/morphological rules"
                    >
                      <ShieldCheck className="w-3 h-3 text-blue-600" />
                      Directly Identified
                    </span>
                  ) : (
                    <span
                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-amber-50 text-amber-800 border border-amber-200"
                      title="Inferred by neural model alignment and contextual fluency (not ground-truth rule)"
                    >
                      <HelpCircle className="w-3 h-3 text-amber-600" />
                      Inferred
                    </span>
                  )}
                </div>

                <span className="text-[11px] text-slate-400 font-mono">
                  chars [{err.start_char}:{err.end_char}]
                </span>
              </div>

              {/* Edit Replacement Snippet */}
              <div className="flex items-center space-x-2 my-2 bg-slate-50 p-2.5 rounded-lg border border-slate-100 text-sm">
                <span className="text-rose-700 font-medium line-through bg-rose-50 px-1.5 py-0.5 rounded">
                  {err.original_text || '(missing)'}
                </span>
                <ArrowRight className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                <span className="text-emerald-700 font-semibold bg-emerald-50 px-1.5 py-0.5 rounded">
                  {err.replacement_text || '(deleted)'}
                </span>
                
                {onApplyErrorFix && (
                  <button
                    onClick={() => onApplyErrorFix(err)}
                    className="ml-auto text-xs px-2.5 py-1 bg-white hover:bg-emerald-50 text-emerald-700 font-medium rounded border border-slate-200 hover:border-emerald-300 transition-colors"
                  >
                    Apply Fix
                  </button>
                )}
              </div>

              {/* Pedagogical Explanation */}
              <p className="text-xs text-slate-600 leading-relaxed mt-1">
                {err.explanation}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
