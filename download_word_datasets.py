# Download Word Datasets Script
# Downloads comprehensive word lists for EN, ES, FR, DE

import requests
import json
import os
from typing import Dict, List

def download_file(url: str, filename: str) -> bool:
    """Download a file from URL"""
    try:
        print(f"📥 Downloading {filename}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def download_word_datasets():
    """Download comprehensive word datasets"""
    
    # Create datasets directory
    os.makedirs('word_datasets', exist_ok=True)
    
    # Define datasets to download
    datasets = {
        # English datasets
        'english_words.txt': 'https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt',
        'english_common.txt': 'https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt',
        'english_positive.txt': 'https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/positive-words.txt',
        'english_negative.txt': 'https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/negative-words.txt',
        
        # Spanish datasets
        'spanish_words.txt': 'https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/es/es_50k.txt',
        'spanish_positive.txt': 'https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-es-165.txt',
        'spanish_common.txt': 'https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/es/es_10k.txt',
        
        # French datasets  
        'french_words.txt': 'https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/fr/fr_50k.txt',
        'french_positive.txt': 'https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-fr-165.txt',
        'french_common.txt': 'https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/fr/fr_10k.txt',
        
        # German datasets
        'german_words.txt': 'https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/de/de_50k.txt',
        'german_positive.txt': 'https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-de-165.txt',
        'german_common.txt': 'https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/de/de_10k.txt',
    }
    
    print("🚀 Downloading Comprehensive Word Datasets")
    print("=" * 50)
    
    success_count = 0
    total_count = len(datasets)
    
    for filename, url in datasets.items():
        filepath = os.path.join('word_datasets', filename)
        if download_file(url, filepath):
            success_count += 1
    
    print(f"\n📊 Download Summary:")
    print(f"   Successfully downloaded: {success_count}/{total_count} files")
    print(f"   Files saved to: word_datasets/")
    
    return success_count == total_count

def process_downloaded_datasets():
    """Process downloaded datasets into organized format"""
    
    print("\n🔄 Processing downloaded datasets...")
    
    processed_data = {
        'en': {'positive': [], 'negative': [], 'general': []},
        'es': {'positive': [], 'negative': [], 'general': []},
        'fr': {'positive': [], 'negative': [], 'general': []},
        'de': {'positive': [], 'negative': [], 'general': []}
    }
    
    # Process English datasets
    try:
        # English general words
        with open('word_datasets/english_words.txt', 'r', encoding='utf-8') as f:
            words = [line.strip().lower() for line in f if line.strip() and len(line.strip()) > 2]
            processed_data['en']['general'].extend(words[:10000])  # Limit to 10k most common
        
        # English positive words
        with open('word_datasets/english_positive.txt', 'r', encoding='utf-8') as f:
            words = [line.strip().lower() for line in f if line.strip() and not line.startswith(';')]
            processed_data['en']['positive'].extend(words)
        
        # English negative words
        with open('word_datasets/english_negative.txt', 'r', encoding='utf-8') as f:
            words = [line.strip().lower() for line in f if line.strip() and not line.startswith(';')]
            processed_data['en']['negative'].extend(words)
            
    except Exception as e:
        print(f"⚠️ Error processing English datasets: {e}")
    
    # Process Spanish datasets
    try:
        # Spanish frequency words (format: word frequency)
        with open('word_datasets/spanish_words.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            words = [line.split()[0].lower() for line in lines if line.strip() and len(line.split()) > 0]
            processed_data['es']['general'].extend([w for w in words if len(w) > 2][:10000])
        
        # Spanish common words
        with open('word_datasets/spanish_common.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            words = [line.split()[0].lower() for line in lines if line.strip() and len(line.split()) > 0]
            processed_data['es']['general'].extend([w for w in words if len(w) > 2][:5000])
    except Exception as e:
        print(f"⚠️ Error processing Spanish datasets: {e}")
    
    # Process French datasets
    try:
        # French frequency words (format: word frequency)
        with open('word_datasets/french_words.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            words = [line.split()[0].lower() for line in lines if line.strip() and len(line.split()) > 0]
            processed_data['fr']['general'].extend([w for w in words if len(w) > 2][:10000])
        
        # French common words
        with open('word_datasets/french_common.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            words = [line.split()[0].lower() for line in lines if line.strip() and len(line.split()) > 0]
            processed_data['fr']['general'].extend([w for w in words if len(w) > 2][:5000])
    except Exception as e:
        print(f"⚠️ Error processing French datasets: {e}")
    
    # Process German datasets
    try:
        # German frequency words (format: word frequency)
        with open('word_datasets/german_words.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            words = [line.split()[0].lower() for line in lines if line.strip() and len(line.split()) > 0]
            processed_data['de']['general'].extend([w for w in words if len(w) > 2][:10000])
        
        # German common words
        with open('word_datasets/german_common.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            words = [line.split()[0].lower() for line in lines if line.strip() and len(line.split()) > 0]
            processed_data['de']['general'].extend([w for w in words if len(w) > 2][:5000])
    except Exception as e:
        print(f"⚠️ Error processing German datasets: {e}")
    
    # Save processed data
    with open('processed_word_datasets.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Processed datasets saved to processed_word_datasets.json")
    
    # Display summary
    for lang, categories in processed_data.items():
        total_words = sum(len(words) for words in categories.values())
        print(f"   {lang.upper()}: {total_words} words")
        for category, words in categories.items():
            print(f"     {category}: {len(words)} words")

if __name__ == "__main__":
    print("🚀 Word Dataset Downloader")
    print("=" * 40)
    
    # Download datasets
    if download_word_datasets():
        # Process downloaded datasets
        process_downloaded_datasets()
        print("\n✅ All datasets downloaded and processed successfully!")
    else:
        print("\n⚠️ Some datasets failed to download, but processing will continue with available files.")
        process_downloaded_datasets()
