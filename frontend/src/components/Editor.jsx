import React from 'react';
import { Send, Trash2, BookOpen, Loader2 } from 'lucide-react';

const PRESETS = [
  { label: "1. SVA: She go to college every day.", text: "She go to college every day." },
  { label: "2. SVA: He have two brothers.", text: "He have two brothers." },
  { label: "3. SVA: They was playing cricket.", text: "They was playing cricket." },
  { label: "4. Correct: I went to the market yesterday.", text: "I went to the market yesterday." },
  { label: "5. Correct: She is a good student.", text: "She is a good student." },
  { label: "6. Tense: I have went to the dentist last week.", text: "I have went to the dentist last week." },
  { label: "7. Prep: He is interested on learning data science.", text: "He is interested on learning data science." },
  { label: "8. Article: This is an useful book for students.", text: "This is an useful book for students." },
];

export default function Editor({ text, setText, onAnalyze, isLoading, onClear }) {
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const charCount = text.length;

  const handlePresetSelect = (e) => {
    const val = e.target.value;
    if (val) {
      setText(val);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      onAnalyze();
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
      {/* Top Toolbar */}
      <div className="px-5 py-3 border-b border-slate-100 bg-slate-50/50 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-2">
          <BookOpen className="w-4 h-4 text-slate-500" />
          <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">Input Text</span>
        </div>

        <div className="flex items-center space-x-2">
          <label className="text-xs text-slate-500 font-medium hidden sm:inline">Try Sample:</label>
          <select
            onChange={handlePresetSelect}
            value={PRESETS.some(p => p.text === text) ? text : ""}
            className="text-xs bg-white border border-slate-300 rounded-lg px-2.5 py-1.5 text-slate-700 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 cursor-pointer"
          >
            <option value="" disabled>Select a test sentence...</option>
            {PRESETS.map((p, idx) => (
              <option key={idx} value={p.text}>{p.label}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Text Area */}
      <div className="p-5">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter an English sentence or paragraph to check for grammatical errors..."
          rows={4}
          className="w-full resize-y text-slate-800 text-base leading-relaxed placeholder-slate-400 focus:outline-none focus:ring-0 border-0 p-0"
        />
      </div>

      {/* Bottom Action Bar */}
      <div className="px-5 py-3 bg-slate-50 border-t border-slate-100 flex items-center justify-between">
        <div className="flex items-center space-x-4 text-xs text-slate-500">
          <span>{wordCount} words</span>
          <span>•</span>
          <span>{charCount} characters</span>
          <span className="hidden md:inline text-slate-400">• Press Ctrl+Enter to check</span>
        </div>

        <div className="flex items-center space-x-2">
          {text && (
            <button
              onClick={onClear}
              disabled={isLoading}
              className="p-2 text-slate-400 hover:text-rose-600 rounded-lg hover:bg-rose-50 transition-colors"
              title="Clear text"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}

          <button
            onClick={onAnalyze}
            disabled={isLoading || !text.trim()}
            className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-200 text-white font-medium text-sm shadow-sm transition-all duration-150 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <Send className="w-4 h-4" />
                <span>Check Grammar</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
