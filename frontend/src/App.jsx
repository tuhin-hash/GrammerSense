import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Editor from './components/Editor';
import DiffViewer from './components/DiffViewer';
import ErrorList from './components/ErrorList';
import CandidatePicker from './components/CandidatePicker';
import MetricsDisplay from './components/MetricsDisplay';
import BenchmarkModal from './components/BenchmarkModal';
import { apiService } from './services/api';
import { AlertCircle, Layers, CheckSquare, Sparkles } from 'lucide-react';

export default function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [selectedCandidate, setSelectedCandidate] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [backendHealth, setBackendHealth] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [isBenchmarkOpen, setIsBenchmarkOpen] = useState(false);

  // Check backend health on mount
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const data = await apiService.getHealth();
        setBackendHealth(data);
      } catch (err) {
        console.warn('Backend currently offline or starting up:', err);
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 8000);
    return () => clearInterval(interval);
  }, []);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setIsLoading(true);
    setErrorMsg(null);

    try {
      const data = await apiService.correctText(text, 3, 4, 1.0);
      setResult(data);
      setSelectedCandidate(data.candidates?.[0] || null);
    } catch (err) {
      console.error('Correction request failed:', err);
      setErrorMsg(
        err.response?.data?.detail ||
        err.message ||
        'Failed to connect to the GEC backend service. Make sure backend is running on http://127.0.0.1:8000'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setText('');
    setResult(null);
    setSelectedCandidate(null);
    setErrorMsg(null);
  };

  const handleSelectCandidate = (cand) => {
    setSelectedCandidate(cand);
    // When an alternative candidate is selected, we can update the corrected text view
    if (result) {
      setResult({
        ...result,
        corrected_text: cand.candidate_text,
      });
    }
  };

  const handleApplyErrorFix = (err) => {
    if (!text || err.start_char === undefined || err.end_char === undefined) return;
    const before = text.slice(0, err.start_char);
    const after = text.slice(err.end_char);
    const newText = before + (err.replacement_text || '') + after;
    setText(newText);
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col text-slate-800">
      <Header
        backendHealth={backendHealth}
        onOpenBenchmark={() => setIsBenchmarkOpen(true)}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {/* Pipeline Stage Indicators Banner */}
        <div className="bg-white rounded-2xl p-4 border border-slate-200 shadow-xs flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="flex items-center space-x-2 text-slate-700 font-semibold">
            <Layers className="w-4 h-4 text-emerald-600" />
            <span>End-to-End NLP Pipeline Stages:</span>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="px-2.5 py-1 rounded-lg bg-emerald-50 text-emerald-800 border border-emerald-200 font-medium">
              A. Dataset Pairs
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-emerald-50 text-emerald-800 border border-emerald-200 font-medium">
              B. Error Taxonomy
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-emerald-50 text-emerald-800 border border-emerald-200 font-medium">
              C. Span Detection
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-emerald-50 text-emerald-800 border border-emerald-200 font-medium">
              D. Beam Candidates
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-emerald-50 text-emerald-800 border border-emerald-200 font-medium">
              E. Candidate Selection
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-emerald-50 text-emerald-800 border border-emerald-200 font-medium">
              F. GLEU Evaluation
            </span>
          </div>
        </div>

        {errorMsg && (
          <div className="p-4 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold">Backend Connection Issue</p>
              <p className="text-xs text-rose-700 mt-0.5">{errorMsg}</p>
            </div>
          </div>
        )}

        {/* Top Section: Editor */}
        <Editor
          text={text}
          setText={setText}
          onAnalyze={handleAnalyze}
          isLoading={isLoading}
          onClear={handleClear}
        />

        {/* Results Section */}
        {result && (
          <div className="space-y-6">
            {/* Quality Metrics */}
            <MetricsDisplay metrics={result.metrics} />

            {/* Split Grid: Diff Viewer + Detected Errors */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <DiffViewer
                originalText={result.original_text}
                correctedText={result.corrected_text}
                diffChunks={result.diff_chunks}
              />

              <ErrorList
                detectedErrors={result.detected_errors}
                onApplyErrorFix={handleApplyErrorFix}
              />
            </div>

            {/* Alternative Candidates */}
            <CandidatePicker
              candidates={result.candidates}
              selectedCandidate={selectedCandidate}
              onSelectCandidate={handleSelectCandidate}
            />
          </div>
        )}
      </main>

      {/* Benchmark Evaluation Modal */}
      <BenchmarkModal
        isOpen={isBenchmarkOpen}
        onClose={() => setIsBenchmarkOpen(false)}
      />

      <footer className="border-t border-slate-200 bg-white py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 text-center text-xs text-slate-500">
          GrammarSense NLP • Real Transformer Inference (T5) • Strict Zero-Fabrication Benchmarks
        </div>
      </footer>
    </div>
  );
}
