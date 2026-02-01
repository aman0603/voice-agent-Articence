import React from 'react';
import { Mic, Send, Trash2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '../../utils/cn';

export function InputBar({ 
  inputValue, 
  setInputValue, 
  handleSend, 
  toggleRecording, 
  isRecording, 
  isProcessing, 
  clearChat 
}) {
  return (
    <div className="p-4 md:p-8 pt-0 z-10">
      <div className="max-w-4xl mx-auto relative">
        <div className="relative glass border border-slate-700/50 rounded-2xl p-2 md:p-3 flex items-center gap-2 md:gap-4 shadow-[0_10px_40px_rgba(0,0,0,0.5)] overflow-hidden">
          {/* Waveform Visualization Placeholder during recording */}
          <AnimatePresence>
            {isRecording && (
              <motion.div 
                className="absolute inset-0 bg-blue-500/5 pointer-events-none flex items-center justify-center -z-10"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className="flex gap-1.5 h-full items-center">
                  {[...Array(20)].map((_, i) => (
                    <motion.div 
                      key={i}
                      className="w-1 bg-blue-400/30 rounded-full"
                      animate={{ height: [8, 32, 12, 48, 8] }}
                      transition={{ 
                        repeat: Infinity, 
                        duration: 1 + Math.random(), 
                        delay: i * 0.05 
                      }}
                    />
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <button 
            type="button"
            onClick={toggleRecording}
            className={cn(
              "p-3 rounded-xl transition-all duration-300 shadow-lg",
              isRecording 
                ? "bg-red-500 text-white rotate-90 scale-110 shadow-[0_0_20px_rgba(239,68,68,0.5)]" 
                : "bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white"
            )}
          >
            <Mic size={20} />
          </button>
          
          <input
            type="text"
            placeholder={isRecording ? "Listening..." : "Ask a technical question..."}
            className="flex-1 bg-transparent border-none focus:ring-0 text-slate-100 placeholder-slate-500 h-10 px-2 text-sm md:text-base selection:bg-blue-500/30"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          />

          <div className="flex items-center gap-2 pr-1">
            <button 
              type="button"
              onClick={clearChat}
              className="hidden md:flex p-2 rounded-lg hover:bg-red-500/10 text-slate-500 hover:text-red-400 transition-all"
              title="Clear Chat"
            >
              <Trash2 size={18} />
            </button>
            <button 
              type="button"
              onClick={() => handleSend()}
              disabled={isProcessing || !inputValue.trim()}
              className={cn(
                "p-3 rounded-xl transition-all flex items-center justify-center shadow-lg",
                inputValue.trim() && !isProcessing
                  ? "bg-blue-600 text-white hover:bg-blue-500 shadow-[0_0_15px_rgba(37,99,235,0.4)]"
                  : "bg-slate-800 text-slate-600 cursor-not-allowed"
              )}
            >
              <Send size={20} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
