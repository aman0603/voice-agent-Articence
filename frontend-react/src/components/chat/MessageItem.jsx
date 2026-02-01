import React from 'react';
import { motion } from 'framer-motion';
import { Volume2 } from 'lucide-react';
import { cn } from '../../utils/cn';

export function MessageItem({ msg }) {
  const isUser = msg.role === 'user';
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95, y: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={cn(
        "flex group",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div className={cn(
        "max-w-[85%] md:max-w-[70%] rounded-2xl p-4 md:p-5 relative",
        isUser 
          ? "bg-blue-600 text-white rounded-tr-none shadow-[0_4px_20px_rgba(37,99,235,0.4)]" 
          : "glass rounded-tl-none border border-slate-700/50 text-slate-100"
      )}>
        <p className="text-[15px] md:text-base leading-relaxed whitespace-pre-wrap">
          {msg.text}
          {msg.isStreaming && <span className="inline-block w-1.5 h-4 ml-1 bg-white/50 animate-pulse rounded-full align-middle" />}
        </p>
        <div className={cn(
          "mt-2 text-[10px] opacity-40 uppercase tracking-widest font-bold flex items-center gap-2",
          isUser ? "justify-end text-white/70" : "text-slate-400"
        )}>
          {!isUser && <Volume2 size={10} className="text-blue-400" />}
          {time}
        </div>
      </div>
    </motion.div>
  );
}
