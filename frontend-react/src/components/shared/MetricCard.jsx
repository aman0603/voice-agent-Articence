import React from 'react';
import { cn } from '../../utils/cn';

export function MetricCard({ label, value, color }) {
  return (
    <div className="bg-[#1e293b]/50 border border-slate-800 p-3 rounded-xl transition-all hover:bg-[#1e293b]/80 group">
      <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-1 group-hover:text-slate-400 transition-colors">{label}</div>
      <div className={cn("text-lg font-bold font-mono transition-transform", color)}>{value}</div>
    </div>
  );
}
