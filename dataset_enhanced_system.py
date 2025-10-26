# Dataset-Enhanced Multilingual Sentiment System
# Uses comprehensive word datasets instead of manual word lists

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import re
from typing import List, Tuple, Dict, Any
import warnings
import json
import time
from dataclasses import dataclass
warnings.filterwarnings('ignore')

# Install and import required packages
try:
    from textblob import TextBlob
    import langdetect
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    import nltk
except ImportError as e:
    print(f"Missing package: {e}")
    print("Run: pip install textblob langdetect scikit-learn nltk")
    exit(1)

from dataset_word_loader import DatasetWordLoader

@dataclass
class SentimentResult:
    """Data class for sentiment analysis results"""
    original_text: str
    final_sentiment: str
    polarity: float
    confidence: float
    num_languages: int
    languages_detected: List[str]
    processing_time: float
    detailed_analysis: Dict = None

class DatasetEnhancedLanguageDetector:
    """Language detector using comprehensive word datasets"""
    
    def __init__(self, use_saved_datasets=True):
        self.language_patterns = {}
        self.dataset_loader = DatasetWordLoader()
        
        if use_saved_datasets:
            # Try to load saved datasets first
            self.datasets = self.dataset_loader.load_saved_datasets()
            if not self.datasets:
                print("🔄 No saved datasets found, loading comprehensive datasets...")
                self.datasets = self.dataset_loader.load_comprehensive_datasets()
                self.dataset_loader.save_datasets(self.datasets)
        else:
            # Load fresh datasets
            self.datasets = self.dataset_loader.load_comprehensive_datasets()
            self.dataset_loader.save_datasets(self.datasets)
        
        # Convert datasets to language patterns format
        self._convert_datasets_to_patterns()
        
        print(f"✅ Language detector initialized with comprehensive datasets")
        for lang, patterns in self.language_patterns.items():
            total_words = sum(len(words) for words in patterns.values())
            print(f"   {lang.upper()}: {total_words} words across {len(patterns)} categories")
    
    def _convert_datasets_to_patterns(self):
        """Convert loaded datasets to language patterns format"""
        for lang, categories in self.datasets.items():
            self.language_patterns[lang] = {}
            for category, words in categories.items():
                self.language_patterns[lang][category] = words
    
    def calculate_fuzzy_membership(self, word: str, language: str) -> float:
        """Calculate fuzzy membership score for word-language pair"""
        word_clean = word.lower().strip('.,!?;:"()[]{}')
        if not word_clean:
            return 0.0
        
        patterns = self.language_patterns.get(language, {})
        score = 0.0
        max_score = 4.0
        
        # Check different linguistic features with weights
        for category, word_list in patterns.items():
            if word_clean in word_list:
                # Different weights for different categories
                if category in ['function_words', 'pronouns']:
                    score += 1.5  # High weight for function words
                elif category in ['positive', 'negative']:
                    score += 1.2  # Medium-high weight for sentiment words
                elif category in ['time_words', 'numbers']:
                    score += 1.0  # Medium weight for time/numbers
                else:
                    score += 0.8  # Lower weight for general words
        
        # Check for language-specific patterns
        if language == 'en':
            if re.match(r'^[a-z]+(ing|ed|ly|tion|ness|ment)$', word_clean):
                score += 0.5
        elif language == 'es':
            if re.match(r'^[a-z]+(ando|iendo|ado|ido|ción|dad|mente)$', word_clean):
                score += 0.5
        elif language == 'fr':
            if re.match(r'^[a-z]+(ant|ent|é|ée|er|ir|tion|ment)$', word_clean):
                score += 0.5
        elif language == 'de':
            if re.match(r'^[a-z]+(en|er|es|ung|keit|heit|lich)$', word_clean):
                score += 0.5
        
        return min(score / max_score, 1.0)
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language with confidence score"""
        if not text or not text.strip():
            return 'unknown', 0.0
        
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 'unknown', 0.0
        
        language_scores = {}
        
        for lang in self.language_patterns.keys():
            total_score = 0.0
            word_count = 0
            
            for word in words:
                if len(word) > 2:  # Skip very short words
                    membership = self.calculate_fuzzy_membership(word, lang)
                    total_score += membership
                    word_count += 1
            
            if word_count > 0:
                language_scores[lang] = total_score / word_count
        
        if not language_scores:
            return 'unknown', 0.0
        
        best_lang = max(language_scores, key=language_scores.get)
        confidence = language_scores[best_lang]
        
        return best_lang, confidence

class DatasetEnhancedSentimentAnalyzer:
    """Enhanced sentiment analyzer using comprehensive datasets"""
    
    def __init__(self):
        self.language_detector = DatasetEnhancedLanguageDetector()
        self.sentiment_lexicons = self._build_sentiment_lexicons()
    
    def _build_sentiment_lexicons(self) -> Dict[str, Dict[str, float]]:
        """Build sentiment lexicons from comprehensive datasets"""
        lexicons = {}
        
        for lang, patterns in self.language_detector.language_patterns.items():
            lexicons[lang] = {
                'positive': {},
                'negative': {},
                'neutral': {}
            }
            
            # Extract sentiment words from datasets
            if 'positive' in patterns:
                for word in patterns['positive']:
                    lexicons[lang]['positive'][word] = 1.0
            
            if 'negative' in patterns:
                for word in patterns['negative']:
                    lexicons[lang]['negative'][word] = -1.0
            
            if 'neutral' in patterns:
                for word in patterns['neutral']:
                    lexicons[lang]['neutral'][word] = 0.0
        
        return lexicons
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment using comprehensive datasets"""
        start_time = time.time()
        
        # Detect language
        detected_lang, lang_confidence = self.language_detector.detect_language(text)
        
        # Use TextBlob as fallback
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        
        # Calculate sentiment using our lexicons
        if detected_lang in self.sentiment_lexicons:
            lexicon_sentiment = self._calculate_lexicon_sentiment(text, detected_lang)
        else:
            lexicon_sentiment = textblob_polarity
        
        # Combine results
        final_polarity = (textblob_polarity + lexicon_sentiment) / 2
        
        # Determine sentiment label
        if final_polarity > 0.1:
            sentiment = 'positive'
        elif final_polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        processing_time = time.time() - start_time
        
        return SentimentResult(
            original_text=text,
            final_sentiment=sentiment,
            polarity=final_polarity,
            confidence=lang_confidence,
            num_languages=1,
            languages_detected=[detected_lang],
            processing_time=processing_time,
            detailed_analysis={
                'textblob_polarity': textblob_polarity,
                'lexicon_sentiment': lexicon_sentiment,
                'language_confidence': lang_confidence,
                'detected_language': detected_lang
            }
        )
    
    def _calculate_lexicon_sentiment(self, text: str, language: str) -> float:
        """Calculate sentiment using lexicon"""
        words = re.findall(r'\b\w+\b', text.lower())
        sentiment_score = 0.0
        word_count = 0
        
        lexicon = self.sentiment_lexicons.get(language, {})
        
        for word in words:
            word_lower = word.lower()
            
            # Check positive words
            if word_lower in lexicon.get('positive', {}):
                sentiment_score += lexicon['positive'][word_lower]
                word_count += 1
            # Check negative words
            elif word_lower in lexicon.get('negative', {}):
                sentiment_score += lexicon['negative'][word_lower]
                word_count += 1
            # Check neutral words
            elif word_lower in lexicon.get('neutral', {}):
                sentiment_score += lexicon['neutral'][word_lower]
                word_count += 1
        
        if word_count > 0:
            return sentiment_score / word_count
        else:
            return 0.0

# Test the enhanced system
if __name__ == "__main__":
    print("🚀 Dataset-Enhanced Multilingual Sentiment System")
    print("=" * 60)
    
    # Initialize the system
    analyzer = DatasetEnhancedSentimentAnalyzer()
    
    # Test sentences in different languages
    test_sentences = [
        "I love this amazing product! It's fantastic and wonderful.",
        "Me encanta este producto increíble! Es fantástico y maravilloso.",
        "J'adore ce produit incroyable! C'est fantastique et merveilleux.",
        "Ich liebe dieses erstaunliche Produkt! Es ist fantastisch und wunderbar.",
        "This is terrible and awful. I hate it completely.",
        "Esto es terrible y horrible. Lo odio completamente.",
        "C'est terrible et affreux. Je le déteste complètement.",
        "Das ist schrecklich und furchtbar. Ich hasse es völlig."
    ]
    
    print("\n📊 Testing sentiment analysis:")
    print("-" * 60)
    
    for sentence in test_sentences:
        result = analyzer.analyze_sentiment(sentence)
        print(f"\nText: {sentence}")
        print(f"Language: {result.languages_detected[0]} (confidence: {result.confidence:.2f})")
        print(f"Sentiment: {result.final_sentiment} (polarity: {result.polarity:.2f})")
        print(f"Processing time: {result.processing_time:.3f}s")
    
    print(f"\n✅ System ready with comprehensive word datasets!")
