import React from 'react';
import { Activity, ShieldCheck, HelpCircle, Gauge, FileText, Info } from 'lucide-react';

export default function MetricsDisplay({ metrics }) {
  if (!metrics) return null;

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-5">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-4">
        <div className="flex items-center space-x-2">
          <Activity className="w-4 h-4 text-emerald-600" />
          <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
            Interpretable Quality Metrics
          </span>
        </div>
        <div className="flex items-center space-x-1 text-slate-400" title="All metrics use documented statistical formulas, never fabricated arbitrary numbers">
          <Info className="w-3.5 h-3.5" />
          <span className="text-[11px]">Justified metrics</span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Error Count */}
        <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-xs text-slate-500 font-medium block">Detected Errors</span>
          <span className="text-2xl font-bold text-slate-900 mt-1 block">
            {metrics.error_count}
          </span>
          <span className="text-[11px] text-slate-400 mt-0.5 block">
            in {metrics.original_word_count} input words
          </span>
        </div>

        {/* Verification Tiers */}
        <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-xs text-slate-500 font-medium block">Verification Tiers</span>
          <div className="mt-1 space-y-0.5 text-xs">
            <div className="flex items-center justify-between text-blue-700 font-medium">
              <span className="flex items-center gap-1"><ShieldCheck className="w-3 h-3" /> Direct:</span>
              <span>{metrics.directly_identified_count}</span>
            </div>
            <div className="flex items-center justify-between text-amber-700 font-medium">
              <span className="flex items-center gap-1"><HelpCircle className="w-3 h-3" /> Inferred:</span>
              <span>{metrics.inferred_count}</span>
            </div>
          </div>
        </div>

        {/* Error Density */}
        <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-xs text-slate-500 font-medium block">Edit Density</span>
          <span className="text-2xl font-bold text-slate-900 mt-1 block">
            {metrics.edit_density_per_100_words}
          </span>
          <span className="text-[10px] text-slate-400 mt-0.5 block">
            errors / 100 words (justified)
          </span>
        </div>

        {/* Error-Free Ratio */}
        <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-xs text-slate-500 font-medium block">Error-Free Token Ratio</span>
          <span className="text-2xl font-bold text-emerald-600 mt-1 block">
            {(metrics.error_free_token_ratio * 100).toFixed(1)}%
          </span>
          <span className="text-[10px] text-slate-400 mt-0.5 block">
            Conf: {(metrics.model_confidence * 100).toFixed(0)}%
          </span>
        </div>
      </div>
    </div>
  );
}
