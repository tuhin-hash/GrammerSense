import React, { useState } from 'react';
import { X, Play, Loader2, Award, CheckCircle, BarChart2, Shield } from 'lucide-react';
import { apiService } from '../services/api';

export default function BenchmarkModal({ isOpen, onClose }) {
  const [datasetName, setDatasetName] = useState('demo');
  const [isRunning, setIsRunning] = useState(false);
  const [evalResult, setEvalResult] = useState(null);
  const [error, setError] = useState(null);

  if (!isOpen) return null;

  const handleRunEvaluation = async () => {
    setIsRunning(true);
    setError(null);
    try {
      const data = await apiService.evaluateDataset(datasetName);
      setEvalResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Evaluation run failed.');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl border border-slate-200 w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-emerald-600 text-white flex items-center justify-center">
              <Award className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900">Benchmark Evaluation Suite</h3>
              <p className="text-xs text-slate-500">Live evaluation on held-out dataset (GLEU, BLEU, F0.5)</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-6 overflow-y-auto space-y-6">
          {/* Controls */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center space-x-3">
              <label className="text-xs font-semibold text-slate-700">Select Dataset:</label>
              <select
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                className="text-xs bg-white border border-slate-300 rounded-lg px-3 py-1.5 text-slate-700 focus:outline-none focus:ring-2 focus:ring-emerald-500/20"
              >
                <option value="demo">Demonstration Set (Immediate Offline Test)</option>
                <option value="test">Held-out Test Split (test.jsonl)</option>
                <option value="val">Validation Split (val.jsonl)</option>
              </select>
            </div>

            <button
              onClick={handleRunEvaluation}
              disabled={isRunning}
              className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold shadow-sm transition-all disabled:opacity-50"
            >
              {isRunning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Evaluating Dataset with Model...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Run Live Evaluation</span>
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs">
              <strong>Error:</strong> {error}
            </div>
          )}

          {/* Results Display */}
          {evalResult && (
            <div className="space-y-5 animate-in fade-in duration-200">
              {/* Metrics Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200">
                  <span className="text-[11px] font-semibold text-emerald-800 uppercase tracking-wider block">
                    Corpus GLEU
                  </span>
                  <span className="text-2xl font-bold text-emerald-700 mt-1 block">
                    {evalResult.metrics?.corpus_gleu}%
                  </span>
                  <span className="text-[10px] text-emerald-600 block mt-0.5">
                    +{evalResult.metrics?.gleu_improvement}% vs source
                  </span>
                </div>

                <div className="p-3.5 rounded-xl bg-blue-50 border border-blue-200">
                  <span className="text-[11px] font-semibold text-blue-800 uppercase tracking-wider block">
                    Corpus BLEU
                  </span>
                  <span className="text-2xl font-bold text-blue-700 mt-1 block">
                    {evalResult.metrics?.corpus_bleu}%
                  </span>
                  <span className="text-[10px] text-blue-600 block mt-0.5">
                    4-gram precision
                  </span>
                </div>

                <div className="p-3.5 rounded-xl bg-purple-50 border border-purple-200">
                  <span className="text-[11px] font-semibold text-purple-800 uppercase tracking-wider block">
                    Token F0.5 Score
                  </span>
                  <span className="text-2xl font-bold text-purple-700 mt-1 block">
                    {evalResult.metrics?.token_f0_5_score_percent}%
                  </span>
                  <span className="text-[10px] text-purple-600 block mt-0.5">
                    P: {evalResult.metrics?.token_precision_percent}% | R: {evalResult.metrics?.token_recall_percent}%
                  </span>
                </div>

                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                  <span className="text-[11px] font-semibold text-slate-700 uppercase tracking-wider block">
                    Exact Match (EM)
                  </span>
                  <span className="text-2xl font-bold text-slate-800 mt-1 block">
                    {evalResult.metrics?.exact_match_accuracy_percent}%
                  </span>
                  <span className="text-[10px] text-slate-500 block mt-0.5">
                    {evalResult.reproducibility?.num_evaluated_samples} samples evaluated
                  </span>
                </div>
              </div>

              {/* Sample Predictions Table */}
              <div>
                <h4 className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                  Sample Predictions on Held-Out Pairs
                </h4>
                <div className="border border-slate-200 rounded-xl overflow-hidden text-xs">
                  <div className="max-h-60 overflow-y-auto">
                    <table className="w-full text-left">
                      <thead className="bg-slate-50 border-b border-slate-200 sticky top-0 text-slate-600">
                        <tr>
                          <th className="p-2.5">Source (Input)</th>
                          <th className="p-2.5">Model Hypothesis</th>
                          <th className="p-2.5">Gold Target</th>
                          <th className="p-2.5 text-center">GLEU</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {evalResult.sample_results?.map((s, idx) => (
                          <tr key={idx} className="hover:bg-slate-50/50">
                            <td className="p-2.5 text-slate-700">{s.source}</td>
                            <td className="p-2.5 font-medium text-emerald-700">{s.hypothesis}</td>
                            <td className="p-2.5 text-slate-500">{s.target}</td>
                            <td className="p-2.5 text-center font-mono">
                              {s.gleu !== undefined ? s.gleu : (s.is_exact_match ? '1.0' : '-')}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              {/* Reproducibility Info */}
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-[11px] text-slate-600 flex flex-wrap items-center justify-between gap-2">
                <span><strong>Model:</strong> {evalResult.reproducibility?.model_name_or_path}</span>
                <span><strong>Seed:</strong> {evalResult.reproducibility?.random_seed}</span>
                <span><strong>Device:</strong> {evalResult.reproducibility?.device}</span>
                <span><strong>Beams:</strong> {evalResult.reproducibility?.num_beams}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
