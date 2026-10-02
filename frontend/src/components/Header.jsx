import React from 'react';
import { Sparkles, Cpu, CheckCircle2, AlertCircle, BarChart3 } from 'lucide-react';

export default function Header({ backendHealth, onOpenBenchmark }) {
  const isHealthy = backendHealth?.status === 'healthy';

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500 flex items-center justify-center text-white shadow-md shadow-emerald-500/20">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-lg text-slate-900 tracking-tight">GrammarSense</span>
              <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200">
                GEC / GED
              </span>
            </div>
            <p className="text-xs text-slate-500">Neural Grammar Error Detection & Correction</p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {/* Backend Status Badge */}
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 text-xs">
            {isHealthy ? (
              <>
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span className="text-slate-600 font-medium">Model:</span>
                <span className="text-slate-900 font-mono text-[11px] max-w-[140px] truncate" title={backendHealth?.model_name}>
                  {backendHealth?.model_name?.split('/').pop() || 'T5 GEC'}
                </span>
                <span className="px-1.5 py-0.5 rounded bg-slate-200 text-[10px] text-slate-700 uppercase font-bold flex items-center gap-1">
                  <Cpu className="w-3 h-3" />
                  {backendHealth?.device || 'CPU'}
                </span>
              </>
            ) : (
              <>
                <AlertCircle className="w-4 h-4 text-amber-500" />
                <span className="text-slate-600 font-medium">Connecting to Backend...</span>
              </>
            )}
          </div>

          {/* Benchmark Evaluation Modal Button */}
          <button
            onClick={onOpenBenchmark}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border border-emerald-200 text-xs font-semibold transition-colors"
          >
            <BarChart3 className="w-4 h-4" />
            <span>Benchmark Evaluation</span>
          </button>
        </div>
      </div>
    </header>
  );
}
