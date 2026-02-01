"""Voice optimizer for converting technical text to speech-friendly format."""

import re
from typing import List, Tuple

from ..config import get_settings
from .phonetic import PhoneticConverter


class VoiceOptimizer:
    """
    Optimizes text for natural-sounding TTS output.
    
    Features:
    - Sentence length limiting
    - Technical jargon simplification
    - Natural pause insertion
    - Phonetic term conversion
    """
    
    def __init__(self, max_words_per_sentence: int = None):
        """Initialize voice optimizer."""
        self.settings = get_settings()
        self.max_words = max_words_per_sentence or self.settings.max_sentence_words
        self.phonetic = PhoneticConverter()
    
    def optimize(self, text: str) -> str:
        """
        Optimize text for TTS.
        
        Args:
            text: Input text (may be technical/long)
            
        Returns:
            Voice-optimized text
        """
        # Step 1: Clean and normalize
        text = self._clean_text(text)
        
        # Step 2: Split into sentences
        sentences = self._split_sentences(text)
        
        # Step 3: Process each sentence
        optimized = []
        for sentence in sentences:
            # Break long sentences
            short_sentences = self._break_long_sentence(sentence)
            
            for short in short_sentences:
                # Simplify language
                simplified = self._simplify_language(short)
                
                # Convert technical terms
                phonetic = self.phonetic.convert(simplified)
                phonetic = self.phonetic.format_number_sequences(phonetic)
                
                optimized.append(phonetic)
        
        return " ".join(optimized)
    
    def optimize_streaming(self, text: str) -> List[str]:
        """
        Optimize text and return as list of speakable chunks.
        
        For streaming TTS where we need sentence-level chunks.
        """
        optimized = self.optimize(text)
        return self._split_sentences(optimized)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove reference patterns like (see Table 7-4)
        text = re.sub(r'\(see\s+[^)]+\)', '', text)
        text = re.sub(r'\(refer to\s+[^)]+\)', '', text)
        text = re.sub(r'\(Figure\s+\d+[^)]*\)', '', text)
        text = re.sub(r'\(Table\s+\d+[^)]*\)', '', text)
        
        # Remove footnote markers
        text = re.sub(r'\[\d+\]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common abbreviations that have periods
        abbrevs = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Jr.', 'Sr.', 'Inc.', 'Ltd.', 'etc.', 'i.e.', 'e.g.']
        for abbrev in abbrevs:
            text = text.replace(abbrev, abbrev.replace('.', '<DOT>'))
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _break_long_sentence(self, sentence: str) -> List[str]:
        """Break a long sentence into shorter ones."""
        words = sentence.split()
        
        if len(words) <= self.max_words:
            return [sentence]
        
        # Find natural break points
        break_patterns = [
            r',\s*(?:and|but|or|so|then|while|because|however)\s',
            r',\s',
            r'\s(?:and|but|or|so|then)\s',
        ]
        
        results = [sentence]
        
        for pattern in break_patterns:
            new_results = []
            for part in results:
                if len(part.split()) > self.max_words:
                    # Try to split at this pattern
                    splits = re.split(f'({pattern})', part, maxsplit=1)
                    if len(splits) > 1:
                        # Rejoin connector with second part
                        new_results.append(splits[0].strip().rstrip(',') + '.')
                        remaining = ''.join(splits[1:]).strip()
                        if remaining:
                            # Capitalize first letter
                            remaining = remaining[0].upper() + remaining[1:] if remaining else remaining
                            new_results.append(remaining)
                    else:
                        new_results.append(part)
                else:
                    new_results.append(part)
            results = new_results
        
        # Ensure all end with punctuation
        final = []
        for r in results:
            r = r.strip()
            if r and not r[-1] in '.!?':
                r += '.'
            if r:
                final.append(r)
        
        return final
    
    def _simplify_language(self, text: str) -> str:
        """Simplify technical language for voice."""
        # Replace jargon with simpler terms
        replacements = [
            (r'\bensure\b', 'make sure'),
            (r'\bverify\b', 'check'),
            (r'\butilize\b', 'use'),
            (r'\bproceed to\b', 'go to'),
            (r'\bsubsequently\b', 'then'),
            (r'\bprior to\b', 'before'),
            (r'\bin the event that\b', 'if'),
            (r'\bat this time\b', 'now'),
            (r'\bmodule\b', 'part'),
            (r'\bcomponent\b', 'part'),
            (r'\bconfigure\b', 'set up'),
            (r'\binitiate\b', 'start'),
            (r'\bterminate\b', 'stop'),
            (r'\bexecute\b', 'run'),
        ]
        
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def get_voice_friendly_chunks(
        self, 
        text: str, 
        chunk_duration_hint: float = 3.0
    ) -> List[Tuple[str, float]]:
        """
        Get voice-friendly chunks with estimated duration.
        
        Args:
            text: Input text
            chunk_duration_hint: Target chunk duration in seconds
            
        Returns:
            List of (chunk_text, estimated_duration_sec) tuples
        """
        sentences = self.optimize_streaming(text)
        
        chunks = []
        # Average speaking rate: ~150 words per minute
        words_per_second = 2.5
        
        for sentence in sentences:
            word_count = len(sentence.split())
            duration = word_count / words_per_second
            chunks.append((sentence, duration))
        
        return chunks
