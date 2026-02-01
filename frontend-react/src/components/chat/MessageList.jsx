import React from 'react';
import { AnimatePresence } from 'framer-motion';
import { MessageItem } from './MessageItem';

export function MessageList({ messages, chatEndRef }) {
  return (
    <section className="flex-1 overflow-y-auto p-4 md:p-8 space-y-8 no-scrollbar scroll-smooth">
      <AnimatePresence initial={false}>
        {messages.map((msg, idx) => (
          <MessageItem key={idx} msg={msg} />
        ))}
      </AnimatePresence>
      <div ref={chatEndRef} />
    </section>
  );
}
