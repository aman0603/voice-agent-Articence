import React, { useState, useEffect, useRef } from 'react';
import { Volume2, Trash2, Zap } from 'lucide-react';
import { Sidebar } from './components/layout/Sidebar';
import { MessageList } from './components/chat/MessageList';
import { InputBar } from './components/chat/InputBar';
import { API_BASE } from './utils/constants';

export default function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', text: "Hello! I'm your Voice AI Agent. I can help you with technical documentation and hardware troubleshooting. Click the mic to speak or type your question below." }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [metrics, setMetrics] = useState({
    ttfb: 0,
    total: 0,
    asr: 'Ready',
    tts: 'Stopped'
  });
  
  const chatEndRef = useRef(null);
  const recognitionRef = useRef(null);
  const abortControllerRef = useRef(null);
  
  // Speech synthesis refs (browser TTS)
  const speechSynthesisRef = useRef(window.speechSynthesis);
  const speechQueueRef = useRef([]);
  const isSpeakingRef = useRef(false);

  // Initialize Speech Recognition
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const rec = new SpeechRecognition();
      rec.continuous = false;
      rec.interimResults = true;
      rec.lang = 'en-US';
      
      rec.onstart = () => {
        setIsRecording(true);
        setMetrics(prev => ({ ...prev, asr: 'Listening...' }));
      };
      
      rec.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }
        
        if (interimTranscript) {
          setInputValue(interimTranscript);
        }
        
        if (finalTranscript) {
          setInputValue(finalTranscript);
          handleSend(finalTranscript);
        }
      };
      
      rec.onerror = (event) => {
        console.error('Speech error:', event.error);
        setIsRecording(false);
        setMetrics(prev => ({ ...prev, asr: 'Error' }));
      };
      
      rec.onend = () => {
        setIsRecording(false);
      };
      
      recognitionRef.current = rec;
    }
  }, []);


  const speakSentence = (text) => {
    return new Promise((resolve) => {
      if (!speechSynthesisRef.current || !text.trim()) {
        resolve();
        return;
      }

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.05;
      utterance.pitch = 1.0;
      
      const voices = speechSynthesisRef.current.getVoices();
      const preferredVoice = voices.find(v => 
        v.name.includes('Google') || v.name.includes('Microsoft') || v.name.includes('Natural')
      ) || voices.find(v => v.lang.startsWith('en')) || voices[0];
      
      if (preferredVoice) utterance.voice = preferredVoice;
      
      utterance.onstart = () => {
        setIsSpeaking(true);
        setMetrics(prev => ({ ...prev, tts: 'Speaking (Browser)' }));
      };
      
      utterance.onend = () => {
        resolve();
      };
      
      utterance.onerror = (e) => {
        console.error('TTS Error:', e);
        resolve();
      };
      
      speechSynthesisRef.current.speak(utterance);
    });
  };

  const processSpeechQueue = async () => {
    if (isSpeakingRef.current) return;
    isSpeakingRef.current = true;
    
    while (speechQueueRef.current.length > 0) {
      const sentence = speechQueueRef.current.shift();
      await speakSentence(sentence);
    }
    
    isSpeakingRef.current = false;
    setIsSpeaking(false);
    setMetrics(prev => ({ ...prev, tts: 'Done' }));
  };

  const handleStop = () => {
    // Stop browser speech synthesis
    if (speechSynthesisRef.current) {
      speechSynthesisRef.current.cancel();
    }
    speechQueueRef.current = [];
    isSpeakingRef.current = false;
    
    // Abort the fetch request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    
    // Reset UI state
    setIsSpeaking(false);
    setIsProcessing(false);
    setMetrics(prev => ({ ...prev, tts: 'Stopped' }));
    
    // Remove streaming message
    setMessages(prev => prev.filter(m => !m.isStreaming));
  };

  const handleSend = async (textToSubmit = inputValue) => {
    const text = textToSubmit.trim();
    if (!text || isProcessing) return;

    setInputValue('');
    setIsProcessing(true);
    
    // Stop any ongoing playback/requests
    handleStop();
    
    // Create new abort controller
    abortControllerRef.current = new AbortController();
    
    // Reset speech queue
    speechQueueRef.current = [];
    isSpeakingRef.current = false;
    setIsSpeaking(false);
    
    // Add user message
    setMessages(prev => [...prev, { role: 'user', text }]);
    setMessages(prev => [...prev, { role: 'assistant', text: '', isStreaming: true }]);
    
    try {
      const response = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, session_id: 'react-session' }),
        signal: abortControllerRef.current?.signal
      });

      if (!response.ok) throw new Error('Server unreachable');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'filler' || data.type === 'sentence') {
                accumulatedText += data.text + ' ';
                setMessages(prev => {
                  const newMsgs = [...prev];
                  const last = newMsgs[newMsgs.length - 1];
                  if (last && last.isStreaming) {
                    last.text = accumulatedText;
                  }
                  return newMsgs;
                });
                
                // Queue text for browser TTS
                speechQueueRef.current.push(data.text);
                processSpeechQueue();
              } else if (data.type === 'ttfb') {
                setMetrics(prev => ({ ...prev, ttfb: data.ttfb_ms }));
              } else if (data.type === 'done') {
                setMetrics(prev => ({ ...prev, total: data.total_latency_ms }));
                setMessages(prev => {
                  const newMsgs = [...prev];
                  const last = newMsgs[newMsgs.length - 1];
                  if (last) last.isStreaming = false;
                  return newMsgs;
                });
              } else if (data.type === 'audio') { // Added for audio streaming
                playAudioChunk(data.audio_base64);
              }
            } catch (e) {
              console.warn('Chunk parse error', e);
            }
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Request was cancelled');
      } else {
        console.error('Fetch error:', error);
        setMessages(prev => [
          ...prev.filter(m => !m.isStreaming),
          { role: 'assistant', text: `Error: ${error.message}. Please check if the backend is running.` }
        ]);
      }
      setIsSpeaking(false);
    } finally {
      setIsProcessing(false);
      abortControllerRef.current = null;
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      recognitionRef.current?.stop();
    } else {
      speechSynthesisRef?.cancel();
      recognitionRef.current?.start();
    }
  };

  const clearChat = () => {
    setMessages([{ role: 'assistant', text: "Chat cleared. How can I help you starting fresh?" }]);
    setMetrics({ ttfb: 0, total: 0, asr: 'Ready', tts: 'Stopped' });
    speechSynthesisRef?.cancel();
  };

  return (
    <div className="flex h-screen bg-[#020617] text-slate-200 overflow-hidden font-sans selection:bg-blue-500/30">
      <Sidebar 
        metrics={metrics} 
        isRecording={isRecording} 
        isProcessing={isProcessing} 
        isSpeaking={isSpeaking}
        onExampleClick={(q) => handleSend(q)}
      />

      <main className="flex-1 flex flex-col relative h-full min-w-0 bg-[#020617]">
        {/* Header - Mobile Only or Mobile Look */}
        <header className="flex lg:hidden items-center justify-between p-4 bg-[#0f172a] border-b border-slate-800 shadow-xl overflow-hidden">
          <div className="flex items-center gap-2">
            <Volume2 className="text-blue-500" size={20} />
            <span className="font-bold text-white tracking-tight">Voice RAG</span>
          </div>
          <button onClick={clearChat} className="p-2 hover:bg-slate-800 rounded-full text-slate-400 transition-colors">
            <Trash2 size={20} />
          </button>
        </header>

        {/* Top Decorative Gradient */}
        <div className="absolute top-0 left-0 right-0 h-48 bg-gradient-to-b from-blue-500/10 to-transparent pointer-events-none" />

        <MessageList messages={messages} chatEndRef={chatEndRef} />

        <InputBar 
          inputValue={inputValue}
          setInputValue={setInputValue}
          handleSend={handleSend}
          handleStop={handleStop}
          toggleRecording={toggleRecording}
          isRecording={isRecording}
          isProcessing={isProcessing}
          isSpeaking={isSpeaking}
          clearChat={clearChat}
        />

        {/* Floating Status - Mobile Only */}
        <div className="lg:hidden absolute bottom-24 right-4 flex flex-col gap-2 pointer-events-none">
           <div className="glass p-3 rounded-full border border-slate-700 shadow-2xl animate-bounce">
              <Zap size={20} className={metrics.ttfb < 800 ? "text-emerald-400" : "text-amber-400"} />
           </div>
        </div>
      </main>
    </div>
  );
}
