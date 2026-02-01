import React from 'react';
import { Volume2, Mic, BookOpen, Cpu, Activity, Zap } from 'lucide-react';
import { StatusItem } from '../shared/StatusItem';
import { MetricCard } from '../shared/MetricCard';
import { EXAMPLE_QUERIES } from '../../utils/constants';

export function Sidebar({ metrics, isRecording, isProcessing, isSpeaking, onExampleClick }) {
  return (
    <aside className="hidden lg:flex flex-col w-80 bg-[#0f172a] border-r border-slate-800 p-6 z-20">
      <div className="flex items-center gap-3 mb-10 group cursor-default">
        <div className="p-2 bg-blue-500 rounded-lg shadow-[0_0_20px_rgba(59,130,246,0.5)] transition-transform group-hover:scale-110">
          <Volume2 className="text-white" size={24} />
        </div>
        <h1 className="text-xl font-bold tracking-tight text-white group-hover:text-blue-400 transition-colors">Voice RAG</h1>
      </div>

      <nav className="flex-1 space-y-6 overflow-y-auto no-scrollbar">
        <div>
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
            <Activity size={12} /> Pipeline Status
          </h3>
          <div className="space-y-3 px-1">
            <StatusItem icon={<Mic size={14} />} label="ASR (Whisper)" status={metrics.asr} active={isRecording} />
            <StatusItem icon={<BookOpen size={14} />} label="Retrieval" status="Active" active />
            <StatusItem icon={<Cpu size={14} />} label="LLM (Trinity)" status="Streaming" active={isProcessing} />
            <StatusItem icon={<Volume2 size={14} />} label="TTS (Browser)" status={metrics.tts} active={isSpeaking} />
          </div>
        </div>

        <div>
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
            <Zap size={12} /> Live Metrics
          </h3>
          <div className="grid grid-cols-2 gap-3 px-1">
            <MetricCard label="TTFB" value={`${Math.round(metrics.ttfb)}ms`} color={metrics.ttfb < 800 ? 'text-emerald-400' : 'text-amber-400'} />
            <MetricCard label="Total" value={`${(metrics.total/1000).toFixed(1)}s`} color="text-blue-400" />
          </div>
        </div>

        <div>
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">Example Queries</h3>
          <div className="space-y-2 px-1">
            {EXAMPLE_QUERIES.map((q, i) => (
              <button 
                key={i}
                onClick={() => onExampleClick(q)}
                className="w-full text-left p-2.5 text-sm rounded-lg hover:bg-slate-800 transition-all border border-transparent hover:border-slate-700 text-slate-400 hover:text-white group"
              >
                <span className="opacity-0 group-hover:opacity-100 transition-opacity mr-1">›</span>
                {q}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <div className="pt-6 border-t border-slate-800">
        <div className="flex justify-between items-center text-slate-500 text-xs text-center">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${true ? 'bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-red-500'}`} />
            Backend: Online
          </div>
          <a href="#" className="hover:text-white transition-colors">v0.1.0</a>
        </div>
      </div>
    </aside>
  );
}
