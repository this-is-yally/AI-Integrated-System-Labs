"""
Naive Bayes Classifier for spam detection
Практичне заняття 4: Ймовірнісні моделі
"""

import re
import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.spam_words = defaultdict(int)
        self.ham_words = defaultdict(int)
        self.spam_count = 0
        self.ham_count = 0
        self.vocabulary = set()
        
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        return words
    
    def train(self, messages, labels):
        """Train the classifier"""
        for message, label in zip(messages, labels):
            words = self.preprocess_text(message)
            
            for word in words:
                self.vocabulary.add(word)
                if label == 'spam':
                    self.spam_words[word] += 1
                else:
                    self.ham_words[word] += 1
            
            if label == 'spam':
                self.spam_count += 1
            else:
                self.ham_count += 1
    
    def calculate_probability(self, message):
        """Calculate probability using Bayes theorem"""
        words = self.preprocess_text(message)
        
        # Prior probabilities
        p_spam = self.spam_count / (self.spam_count + self.ham_count)
        p_ham = self.ham_count / (self.spam_count + self.ham_count)
        
        # Initialize logarithmic probabilities
        log_p_spam = math.log(p_spam) if p_spam > 0 else -float('inf')
        log_p_ham = math.log(p_ham) if p_ham > 0 else -float('inf')
        
        # Calculate probabilities for each word
        for word in words:
            # Laplace smoothing
            p_word_spam = (self.spam_words.get(word, 0) + 1) / (self.spam_count + len(self.vocabulary))
            p_word_ham = (self.ham_words.get(word, 0) + 1) / (self.ham_count + len(self.vocabulary))
            
            log_p_spam += math.log(p_word_spam)
            log_p_ham += math.log(p_word_ham)
        
        return log_p_spam, log_p_ham
    
    def classify(self, message):
        """Classify message as spam or ham"""
        log_p_spam, log_p_ham = self.calculate_probability(message)
        
        if log_p_spam > log_p_ham:
            return 'spam', log_p_spam, log_p_ham
        else:
            return 'ham', log_p_spam, log_p_ham

def load_data_from_csv(filename):
    """Load data from CSV file"""
    import csv
    messages = []
    labels = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            messages.append(row['message'])
            labels.append(row['label'])
    
    return messages, labels

def demo_bayes_classifier():
    """Demonstrate the Bayes classifier"""
    # Load data
    try:
        messages, labels = load_data_from_csv('data/spam_dataset.csv')
    except:
        # Fallback data if file not found
        messages = [
            "Виграй iPhone безкоштовно", "Купи зараз з великою знижкою",
            "Привіт, як справи?", "Зустрінемось о 15:00"
        ]
        labels = ['spam', 'spam', 'ham', 'ham']
    
    # Train classifier
    classifier = NaiveBayesClassifier()
    classifier.train(messages, labels)
    
    # Test classifier
    test_messages = [
        "Купи зараз iPhone зі знижкою!",
        "Надішли, будь ласка, презентацію до завтра",
        "Виграв автомобіль у конкурсі"
    ]
    
    print("НАЇВНИЙ БАЄСІВСЬКИЙ КЛАСИФІКАТОР")
    print("=" * 50)
    
    for message in test_messages:
        result, p_spam, p_ham = classifier.classify(message)
        print(f"\nПовідомлення: '{message}'")
        print(f"Результат: {result.upper()}")
        print(f"Ймовірність спаму: {p_spam:.2f}")
        print(f"Ймовірність не спаму: {p_ham:.2f}")
        print("-" * 40)

if __name__ == "__main__":
    demo_bayes_classifier()