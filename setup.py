import subprocess
import sys

# Install dependencies first
print("📦 Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Now import and setup NLTK
import nltk
print("📚 Downloading NLTK data...")
nltk.download('punkt')
nltk.download('brown')
print("✅ Setup complete!")