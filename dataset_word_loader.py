# Dataset Word Loader for Multilingual Sentiment System
# Automatically loads comprehensive word datasets from various sources

import nltk
import requests
import json
import os
from typing import Dict, List, Set
from collections import defaultdict

class DatasetWordLoader:
    """Loads comprehensive word datasets for multiple languages"""
    
    def __init__(self):
        self.download_nltk_data()
        self.word_datasets = {}
        
    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('brown', quiet=True)
            nltk.download('words', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)  # Open Multilingual Wordnet
            print("✅ NLTK data downloaded successfully")
        except Exception as e:
            print(f"⚠️ NLTK download warning: {e}")
    
    def load_nltk_words(self) -> Dict[str, Set[str]]:
        """Load words from NLTK corpora"""
        words = defaultdict(set)
        
        try:
            # Load stopwords for all languages
            from nltk.corpus import stopwords
            
            # English stopwords
            try:
                words['en'].update(stopwords.words('english'))
            except:
                pass
            
            # Spanish stopwords
            try:
                words['es'].update(stopwords.words('spanish'))
            except:
                pass
                
            # French stopwords
            try:
                words['fr'].update(stopwords.words('french'))
            except:
                pass
                
            # German stopwords
            try:
                words['de'].update(stopwords.words('german'))
            except:
                pass
                
        except Exception as e:
            print(f"⚠️ NLTK words loading warning: {e}")
            
        return dict(words)
    
    def load_web_datasets(self) -> Dict[str, Set[str]]:
        """Load word datasets from web sources"""
        words = defaultdict(set)
        
        # Common word lists from GitHub repositories
        datasets = {
            'en': [
                'https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt',
                'https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt'
            ],
            'es': [
                'https://raw.githubusercontent.com/words/an-array-of-spanish-words/master/words.json'
            ],
            'fr': [
                'https://raw.githubusercontent.com/words/an-array-of-french-words/master/words.json'
            ],
            'de': [
                'https://raw.githubusercontent.com/words/an-array-of-german-words/master/words.json'
            ]
        }
        
        for lang, urls in datasets.items():
            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        if url.endswith('.json'):
                            data = response.json()
                            if isinstance(data, list):
                                words[lang].update(data)
                        else:
                            lines = response.text.strip().split('\n')
                            words[lang].update(line.strip().lower() for line in lines if line.strip())
                        print(f"✅ Loaded {len(words[lang])} {lang} words from {url}")
                except Exception as e:
                    print(f"⚠️ Failed to load from {url}: {e}")
        
        return dict(words)
    
    def load_local_datasets(self) -> Dict[str, Set[str]]:
        """Load word datasets from local files"""
        words = defaultdict(set)
        
        # Check for local word files
        local_files = {
            'en': ['english_words.txt', 'en_words.txt'],
            'es': ['spanish_words.txt', 'es_words.txt'],
            'fr': ['french_words.txt', 'fr_words.txt'],
            'de': ['german_words.txt', 'de_words.txt']
        }
        
        for lang, filenames in local_files.items():
            for filename in filenames:
                if os.path.exists(filename):
                    try:
                        with open(filename, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            words[lang].update(line.strip().lower() for line in lines if line.strip())
                        print(f"✅ Loaded {len(words[lang])} {lang} words from {filename}")
                    except Exception as e:
                        print(f"⚠️ Failed to load {filename}: {e}")
        
        return dict(words)
    
    def load_comprehensive_datasets(self) -> Dict[str, Dict[str, List[str]]]:
        """Load comprehensive word datasets and organize by category"""
        print("🔄 Loading comprehensive word datasets...")
        
        # Load from all sources
        nltk_words = self.load_nltk_words()
        web_words = self.load_web_datasets()
        local_words = self.load_local_datasets()
        
        # Load sentiment lexicons
        sentiment_lexicons = self.load_sentiment_lexicons()
        
        # Combine all sources
        all_words = defaultdict(set)
        for source in [nltk_words, web_words, local_words]:
            for lang, word_set in source.items():
                all_words[lang].update(word_set)
        
        # Add sentiment words
        for lang, sentiment_data in sentiment_lexicons.items():
            if lang not in all_words:
                all_words[lang] = set()
            all_words[lang].update(sentiment_data['positive'])
            all_words[lang].update(sentiment_data['negative'])
        
        # Organize by categories
        organized_words = {}
        
        for lang in ['en', 'es', 'fr', 'de']:
            if lang not in all_words:
                continue
                
            word_list = list(all_words[lang])
            organized_words[lang] = self.categorize_words(word_list, lang)
            
            # Add sentiment words to organized structure
            if lang in sentiment_lexicons:
                organized_words[lang]['positive'] = sentiment_lexicons[lang]['positive']
                organized_words[lang]['negative'] = sentiment_lexicons[lang]['negative']
            
            print(f"📚 {lang.upper()}: {len(word_list)} total words")
            for category, words in organized_words[lang].items():
                print(f"   {category}: {len(words)} words")
        
        return organized_words
    
    def load_sentiment_lexicons(self) -> Dict[str, Dict[str, List[str]]]:
        """Load sentiment lexicons for ES, FR, DE"""
        try:
            with open('sentiment_lexicons.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("📝 Creating sentiment lexicons...")
            # Import and run the sentiment lexicon creator
            try:
                from create_sentiment_lexicons import create_sentiment_lexicons
                return create_sentiment_lexicons()
            except ImportError:
                print("⚠️ Could not create sentiment lexicons, using basic patterns")
                return {}
        except Exception as e:
            print(f"⚠️ Error loading sentiment lexicons: {e}")
            return {}
    
    def categorize_words(self, words: List[str], language: str) -> Dict[str, List[str]]:
        """Categorize words by type and sentiment"""
        categories = defaultdict(list)
        
        # Common patterns for categorization
        patterns = {
            'en': {
                'function_words': ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'],
                'pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'],
                'positive': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 'perfect'],
                'negative': ['bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'worst', 'disappointing'],
                'neutral': ['house', 'car', 'book', 'table', 'chair', 'window', 'door', 'wall', 'floor']
            },
            'es': {
                'function_words': ['el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'pero', 'en', 'de', 'con', 'por'],
                'pronouns': ['yo', 'tú', 'él', 'ella', 'nosotros', 'vosotros', 'ellos', 'ellas'],
                'positive': ['bueno', 'excelente', 'maravilloso', 'fantástico', 'increíble', 'genial', 'perfecto'],
                'negative': ['malo', 'terrible', 'horrible', 'asqueroso', 'odiar', 'peor', 'decepcionante'],
                'neutral': ['casa', 'coche', 'libro', 'mesa', 'silla', 'ventana', 'puerta', 'pared', 'suelo']
            },
            'fr': {
                'function_words': ['le', 'la', 'les', 'un', 'une', 'et', 'ou', 'mais', 'dans', 'de', 'avec', 'pour'],
                'pronouns': ['je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles'],
                'positive': ['bon', 'excellent', 'merveilleux', 'fantastique', 'incroyable', 'génial', 'parfait'],
                'negative': ['mauvais', 'terrible', 'horrible', 'dégoûtant', 'détester', 'pire', 'décevant'],
                'neutral': ['maison', 'voiture', 'livre', 'table', 'chaise', 'fenêtre', 'porte', 'mur', 'sol']
            },
            'de': {
                'function_words': ['der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'aber', 'in', 'auf', 'unter'],
                'pronouns': ['ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'sie'],
                'positive': ['gut', 'ausgezeichnet', 'wunderbar', 'fantastisch', 'unglaublich', 'großartig', 'perfekt'],
                'negative': ['schlecht', 'schrecklich', 'furchtbar', 'ekelhaft', 'hassen', 'schlimmste', 'enttäuschend'],
                'neutral': ['Haus', 'Auto', 'Buch', 'Tisch', 'Stuhl', 'Fenster', 'Tür', 'Wand', 'Boden']
            }
        }
        
        lang_patterns = patterns.get(language, {})
        
        for word in words:
            word_lower = word.lower()
            
            # Categorize based on patterns
            categorized = False
            for category, pattern_words in lang_patterns.items():
                if word_lower in pattern_words:
                    categories[category].append(word)
                    categorized = True
                    break
            
            # If not categorized, add to general words
            if not categorized:
                categories['general'].append(word)
        
        return dict(categories)
    
    def save_datasets(self, datasets: Dict[str, Dict[str, List[str]]], filename: str = 'comprehensive_word_datasets.json'):
        """Save loaded datasets to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(datasets, f, ensure_ascii=False, indent=2)
            print(f"💾 Datasets saved to {filename}")
        except Exception as e:
            print(f"⚠️ Failed to save datasets: {e}")
    
    def load_saved_datasets(self, filename: str = 'comprehensive_word_datasets.json') -> Dict[str, Dict[str, List[str]]]:
        """Load previously saved datasets"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                datasets = json.load(f)
            print(f"📂 Loaded datasets from {filename}")
            return datasets
        except FileNotFoundError:
            print(f"📂 No saved datasets found, will create new ones")
            return {}
        except Exception as e:
            print(f"⚠️ Failed to load saved datasets: {e}")
            return {}

# Test the dataset loader
if __name__ == "__main__":
    print("🚀 Comprehensive Word Dataset Loader")
    print("=" * 50)
    
    loader = DatasetWordLoader()
    
    # Try to load saved datasets first
    datasets = loader.load_saved_datasets()
    
    # If no saved datasets, create new ones
    if not datasets:
        datasets = loader.load_comprehensive_datasets()
        loader.save_datasets(datasets)
    
    # Display summary
    total_words = 0
    for lang, categories in datasets.items():
        lang_total = sum(len(words) for words in categories.values())
        total_words += lang_total
        print(f"\n📚 {lang.upper()}: {lang_total} words")
        for category, words in categories.items():
            print(f"   {category}: {len(words)} words")
    
    print(f"\n✅ Total words loaded: {total_words}")
