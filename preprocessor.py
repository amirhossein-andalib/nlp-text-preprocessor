import re
import nltk
import hazm
from hazm import Normalizer, Stemmer, Lemmatizer, stopwords_list
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')


# ─────────────────────────────────────────
#           PERSIAN PREPROCESSOR
# ─────────────────────────────────────────

class PersianPreprocessor:
    def __init__(self):
        self.normalizer = Normalizer()
        self.stemmer = Stemmer()
        self.lemmatizer = Lemmatizer()
        self.persian_stopwords = stopwords_list()

    def normalize(self, text):
        """Normalize Persian text"""
        return self.normalizer.normalize(text)

    def tokenize(self, text):
        """Tokenize Persian text into words (uses Hazm's tokenizer, not NLTK's)"""
        return hazm.word_tokenize(text)

    def remove_stopwords(self, tokens):
        """Remove Persian stopwords"""
        return [token for token in tokens if token not in self.persian_stopwords]

    def stem(self, tokens):
        """Stem Persian words"""
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize(self, tokens):
        """Lemmatize Persian words"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def remove_punctuation(self, text):
        """Remove punctuation from Persian text"""
        return re.sub(r'[،؛؟!.«»\(\)\[\]{}]', ' ', text)

    def remove_arabic_diacritics(self, text):
        """Remove Arabic diacritics (harakat)"""
        diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F]')
        return diacritics.sub('', text)

    def preprocess(self, text, do_stem=False, do_lemmatize=True):
        """Full Persian preprocessing pipeline"""
        print("\n[PERSIAN PIPELINE]")
        print(f"Original     : {text}")

        text = self.normalize(text)
        print(f"Normalized   : {text}")

        text = self.remove_arabic_diacritics(text)
        text = self.remove_punctuation(text)

        tokens = self.tokenize(text)
        print(f"Tokens       : {tokens}")

        tokens = self.remove_stopwords(tokens)
        print(f"No Stopwords : {tokens}")

        if do_lemmatize:
            tokens = self.lemmatize(tokens)
            print(f"Lemmatized   : {tokens}")

        if do_stem:
            tokens = self.stem(tokens)
            print(f"Stemmed      : {tokens}")

        return tokens


# ─────────────────────────────────────────
#           ENGLISH PREPROCESSOR
# ─────────────────────────────────────────

class EnglishPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.english_stopwords = set(stopwords.words('english'))

    def to_lowercase(self, text):
        """Convert text to lowercase"""
        return text.lower()

    def remove_punctuation(self, text):
        """Remove punctuation"""
        return re.sub(r'[^\w\s]', ' ', text)

    def remove_numbers(self, text):
        """Remove numbers"""
        return re.sub(r'\d+', '', text)

    def tokenize(self, text):
        """Tokenize English text"""
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        """Remove English stopwords"""
        return [token for token in tokens if token not in self.english_stopwords]

    def lemmatize(self, tokens):
        """Lemmatize English words"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def remove_extra_spaces(self, text):
        """Remove extra whitespace"""
        return re.sub(r'\s+', ' ', text).strip()

    def preprocess(self, text, do_lemmatize=True):
        """Full English preprocessing pipeline"""
        print("\n[ENGLISH PIPELINE]")
        print(f"Original     : {text}")

        text = self.to_lowercase(text)
        print(f"Lowercase    : {text}")

        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_extra_spaces(text)

        tokens = self.tokenize(text)
        print(f"Tokens       : {tokens}")

        tokens = self.remove_stopwords(tokens)
        print(f"No Stopwords : {tokens}")

        if do_lemmatize:
            tokens = self.lemmatize(tokens)
            print(f"Lemmatized   : {tokens}")

        return tokens
