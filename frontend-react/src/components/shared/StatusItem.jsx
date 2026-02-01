import React from 'react';
import { cn } from '../../utils/cn';

export function StatusItem({ icon, label, status, active }) {
  return (
    <div className="flex items-center justify-between group">
      <div className="flex items-center gap-3">
        <div className={cn(
          "p-1.5 rounded-md transition-all",
          active ? "bg-blue-500/20 text-blue-400 glow-sm" : "bg-slate-800 text-slate-500"
        )}>
          {icon}
        </div>
        <span className={cn("text-xs font-medium transition-colors", active ? "text-slate-200" : "text-slate-500")}>
          {label}
        </span>
      </div>
      <span className={cn(
        "text-[10px] px-2 py-0.5 rounded-full border transition-all",
        active 
          ? "border-blue-500/50 text-blue-400 bg-blue-500/10 shadow-[0_0_10px_rgba(59,130,246,0.2)]" 
          : "border-slate-800 text-slate-600 bg-slate-900/50"
      )}>
        {status}
      </span>
    </div>
  );
}
