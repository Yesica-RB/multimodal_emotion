# src/nlp_classic/preprocessing.py
# This module cleans and prepares tweet text before classification.
# Steps: lowercase → remove URLs/mentions/hashtags → tokenize
#        → remove stopwords → apply Porter stemming

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Download required NLTK resources (only runs once)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """Cleans and normalises raw tweet text for NLP classification."""

    def __init__(self):
        self.stemmer    = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def clean(self, text: str) -> str:
        """Remove noise: URLs, mentions, hashtags, punctuation."""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)   # remove URLs
        text = re.sub(r'@\w+|#\w+',        '', text)   # remove mentions and hashtags
        text = re.sub(r'[^\w\s]',           '', text)   # remove punctuation
        text = re.sub(r'\s+',              ' ', text).strip()  # remove extra spaces
        return text

    def tokenize(self, text: str) -> list:
        """Split text into individual word tokens."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list) -> list:
        """Remove common words that carry no emotional meaning (e.g. 'the', 'a')."""
        return [t for t in tokens if t not in self.stop_words]

    def stem(self, tokens: list) -> list:
        """Reduce each word to its root form using Porter Stemmer.
        Example: 'running', 'runs', 'ran' → 'run'
        """
        return [self.stemmer.stem(t) for t in tokens]

    def preprocess(self, text: str) -> str:
        """Full pipeline: clean → tokenize → remove stopwords → stem.
        Returns a single cleaned string ready for TF-IDF vectorisation.
        """
        text   = self.clean(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem(tokens)
        return ' '.join(tokens)
