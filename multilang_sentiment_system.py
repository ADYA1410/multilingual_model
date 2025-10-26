# Multi-Language Sentiment Analysis System
# Detects multiple languages in code-switched text and analyzes sentiment

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
class MultiLanguageResult:
    """Data class for multi-language sentiment analysis results"""
    original_text: str
    detected_languages: List[str]
    language_confidence: Dict[str, float]
    dominant_language: str
    sentiment_by_language: Dict[str, str]
    overall_sentiment: str
    overall_polarity: float
    confidence: float
    processing_time: float
    detailed_analysis: Dict = None

class MultiLanguageDetector:
    """Detects multiple languages in code-switched text"""
    
    def __init__(self):
        self.dataset_loader = DatasetWordLoader()
        self.datasets = self.dataset_loader.load_saved_datasets()
        if not self.datasets:
            print("🔄 No saved datasets found, loading comprehensive datasets...")
            self.datasets = self.dataset_loader.load_comprehensive_datasets()
            self.dataset_loader.save_datasets(self.datasets)
        
        # Convert datasets to language patterns format
        self.language_patterns = {}
        for lang, categories in self.datasets.items():
            self.language_patterns[lang] = {}
            for category, words in categories.items():
                self.language_patterns[lang][category] = words
        
        print(f"✅ Multi-language detector initialized")
        for lang, patterns in self.language_patterns.items():
            total_words = sum(len(words) for words in patterns.values())
            print(f"   {lang.upper()}: {total_words} words across {len(patterns)} categories")
    
    def detect_languages_in_text(self, text: str) -> Tuple[List[str], Dict[str, float]]:
        """Detect all languages present in the text"""
        if not text or not text.strip():
            return [], {}
        
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return [], {}
        
        language_scores = {}
        word_language_matches = defaultdict(int)  # Count words matched per language
        word_language_details = defaultdict(list)  # Store which words matched which languages
        
        # Calculate scores for each language
        for lang in self.language_patterns.keys():
            total_score = 0.0
            word_count = 0
            matched_words = 0
            strong_matches = 0  # High confidence matches
            
            for word in words:
                if len(word) > 1:  # Include shorter words too
                    membership = self.calculate_word_membership(word, lang)
                    total_score += membership
                    word_count += 1
                    
                    # Count if word has any membership in this language
                    if membership > 0.2:  # Higher threshold for strong matches
                        matched_words += 1
                        strong_matches += 1
                        word_language_details[lang].append((word, membership))
                    elif membership > 0.1:  # Lower threshold for weak matches
                        matched_words += 1
                        word_language_details[lang].append((word, membership))
            
            if word_count > 0:
                # Use both average score and word count ratio
                avg_score = total_score / word_count
                word_ratio = matched_words / word_count
                strong_ratio = strong_matches / word_count
                
                # Combined score with emphasis on strong matches
                combined_score = (avg_score * 0.4) + (word_ratio * 0.3) + (strong_ratio * 0.3)
                language_scores[lang] = combined_score
                word_language_matches[lang] = matched_words
        
        # Only include languages with strong evidence
        significant_languages = {
            lang: score for lang, score in language_scores.items() 
            if score >= 0.15 or word_language_matches[lang] >= 2  # Need at least 2 word matches or high score
        }
        
        # Sort by confidence score
        sorted_languages = sorted(significant_languages.items(), key=lambda x: x[1], reverse=True)
        
        detected_languages = [lang for lang, score in sorted_languages]
        confidence_scores = dict(sorted_languages)
        
        return detected_languages, confidence_scores
    
    def calculate_word_membership(self, word: str, language: str) -> float:
        """Calculate membership score for word-language pair"""
        word_clean = word.lower().strip('.,!?;:"()[]{}')
        if not word_clean:
            return 0.0
        
        patterns = self.language_patterns.get(language, {})
        score = 0.0
        max_score = 3.0  # Reduced max score for more precise scoring
        
        # Check exact matches first (most important)
        exact_match = False
        for category, word_list in patterns.items():
            if word_clean in word_list:
                exact_match = True
                # Different weights for different categories
                if category in ['function_words', 'pronouns']:
                    score += 2.5  # Highest weight for function words
                elif category in ['positive', 'negative']:
                    score += 2.0  # High weight for sentiment words
                elif category in ['time_words', 'numbers']:
                    score += 1.8  # High weight for time/numbers
                else:
                    score += 1.5  # Medium weight for general words
        
        # Only check patterns if we have an exact match
        if exact_match:
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
        
        # Only check partial matches for very short words or if no exact match
        if not exact_match and len(word_clean) <= 3:
            for category, word_list in patterns.items():
                for dict_word in word_list:
                    if word_clean in dict_word or dict_word in word_clean:
                        score += 0.2  # Small bonus for partial matches
        
        return min(score / max_score, 1.0)
    
    def get_dominant_language(self, languages: List[str], confidence_scores: Dict[str, float]) -> str:
        """Get the dominant language from detected languages"""
        if not languages:
            return 'unknown'
        
        # Return the language with highest confidence
        return max(confidence_scores, key=confidence_scores.get)

class MultiLanguageSentimentAnalyzer:
    """Analyzes sentiment in multi-language text"""
    
    def __init__(self):
        self.language_detector = MultiLanguageDetector()
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
    
    def analyze_sentiment(self, text: str) -> MultiLanguageResult:
        """Analyze sentiment in multi-language text"""
        start_time = time.time()
        
        # Detect all languages in the text
        detected_languages, language_confidence = self.language_detector.detect_languages_in_text(text)
        dominant_language = self.language_detector.get_dominant_language(detected_languages, language_confidence)
        
        # Analyze sentiment for each detected language
        sentiment_by_language = {}
        language_polarities = {}
        
        for lang in detected_languages:
            lang_sentiment = self._analyze_language_sentiment(text, lang)
            sentiment_by_language[lang] = lang_sentiment['sentiment']
            language_polarities[lang] = lang_sentiment['polarity']
        
        # Calculate overall sentiment
        if language_polarities:
            overall_polarity = np.mean(list(language_polarities.values()))
        else:
            # Fallback to TextBlob
            blob = TextBlob(text)
            overall_polarity = blob.sentiment.polarity
        
        # Determine overall sentiment
        if overall_polarity > 0.1:
            overall_sentiment = 'positive'
        elif overall_polarity < -0.1:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Calculate overall confidence
        overall_confidence = np.mean(list(language_confidence.values())) if language_confidence else 0.0
        
        processing_time = time.time() - start_time
        
        return MultiLanguageResult(
            original_text=text,
            detected_languages=detected_languages,
            language_confidence=language_confidence,
            dominant_language=dominant_language,
            sentiment_by_language=sentiment_by_language,
            overall_sentiment=overall_sentiment,
            overall_polarity=overall_polarity,
            confidence=overall_confidence,
            processing_time=processing_time,
            detailed_analysis={
                'language_polarities': language_polarities,
                'textblob_polarity': TextBlob(text).sentiment.polarity if detected_languages else 0.0
            }
        )
    
    def _analyze_language_sentiment(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze sentiment for a specific language"""
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
            polarity = sentiment_score / word_count
        else:
            # Fallback to TextBlob
            polarity = TextBlob(text).sentiment.polarity
        
        # Determine sentiment
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'polarity': polarity
        }

# Interactive Demo
def run_interactive_demo():
    """Run interactive demo for multi-language sentiment analysis"""
    print("🌍 MULTI-LANGUAGE SENTIMENT ANALYSIS DEMO")
    print("=" * 60)
    print("✨ Detects multiple languages in code-switched text!")
    print("✨ Try: 'I amour you', 'C'est amazing!', 'Das ist really gut!'")
    print("✨ Type 'quit' to exit")
    print("=" * 60)
    
    # Initialize analyzer
    print("\n🔄 Loading comprehensive datasets...")
    analyzer = MultiLanguageSentimentAnalyzer()
    print("✅ Ready! Start typing...\n")
    
    count = 1
    while True:
        try:
            # Get user input
            text = input(f"📝 Text #{count}: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not text:
                print("⚠️ Please enter some text")
                continue
            
            # Analyze sentiment
            result = analyzer.analyze_sentiment(text)
            
            # Display results
            print(f"\n🔍 ANALYSIS RESULTS:")
            print(f"   📝 Text: \"{text}\"")
            print(f"   🌐 Languages detected: {', '.join([lang.upper() for lang in result.detected_languages])}")
            print(f"   🎯 Dominant language: {result.dominant_language.upper()}")
            print(f"   📊 Language confidence: {result.language_confidence}")
            print(f"   😊 Overall sentiment: {result.overall_sentiment.upper()} (polarity: {result.overall_polarity:.2f})")
            print(f"   🔍 Sentiment by language: {result.sentiment_by_language}")
            print(f"   ⚡ Processing time: {result.processing_time:.3f}s")
            
            # Debug: Show word-level analysis
            words = re.findall(r'\b\w+\b', text.lower())
            print(f"   🔬 Debug - Words analyzed: {words}")
            for word in words:
                if len(word) > 1:
                    word_scores = {}
                    for lang in ['en', 'es', 'fr', 'de']:
                        if lang in analyzer.language_detector.language_patterns:
                            score = analyzer.language_detector.calculate_word_membership(word, lang)
                            if score > 0.1:
                                word_scores[lang] = score
                    if word_scores:
                        print(f"     '{word}': {word_scores}")
            print()
            
            count += 1
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()

if __name__ == "__main__":
    run_interactive_demo()
