"""
Machine Learning Models Comparison
Практичне заняття 5: Класичне машинне навчання
"""

import math
import random
from collections import defaultdict

class MLComparison:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def prepare_data(self):
        """Prepare spam classification data"""
        messages = [
            # SPAM
            "Виграй iPhone безкоштовно", "Купи зараз з великою знижкою",
            "Термінова розпродаж тільки сьогодні", "Виграв мільйон у лотерею",
            "Супер пропозиція тільки для вас", "Гарантований виграш приз",
            "Отримай безкоштовний подарунок", "Акція знижки 50%",
            "Швидкий заробіток в інтернеті", "Кредит з низькою ставкою",
            # HAM
            "Привіт, як справи?", "Зустрінемось о 15:00",
            "Надсилаю документи для перегляду", "Дякую за допомогу",
            "Зателефонуй мені будь ласка", "Презентація готова до показу",
            "Чи можна перенести зустріч?", "Нагадую про завтрашню нараду",
            "Надішли звіт до кінця дня", "Обговорення проекту завтра"
        ]
        
        labels = ['spam'] * 10 + ['ham'] * 10
        return messages, labels
    
    def extract_features(self, messages):
        """Extract features from messages"""
        features = []
        for message in messages:
            words = message.lower().split()
            features.append([
                len(message),
                1 if '!' in message else 0,
                1 if any(word in message.lower() for word in ['виграй', 'виграв', 'приз']) else 0,
                1 if any(word in message.lower() for word in ['купи', 'знижк', 'акція']) else 0,
                len(words),
                1 if any(word in message.lower() for word in ['терміново', 'швидко']) else 0,
                1 if any(word in message.lower() for word in ['безкоштовно', 'безплатно']) else 0,
            ])
        return features
    
    def train_test_split(self, X, y, test_size=0.2):
        """Split data into train and test sets"""
        n_test = int(len(X) * test_size)
        indices = list(range(len(X)))
        random.shuffle(indices)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_train = [y[i] for i in train_indices]
        y_test = [y[i] for i in test_indices]
        
        return X_train, X_test, y_train, y_test
    
    def normalize_features(self, X_train, X_test):
        """Normalize features for KNN"""
        if not X_train:
            return X_train, X_test
            
        mins = [min(feature) for feature in zip(*X_train)]
        maxs = [max(feature) for feature in zip(*X_train)]
        
        X_train_norm = []
        for sample in X_train:
            normalized = [(x - min_val) / (max_val - min_val) if max_val != min_val else 0 
                         for x, min_val, max_val in zip(sample, mins, maxs)]
            X_train_norm.append(normalized)
        
        X_test_norm = []
        for sample in X_test:
            normalized = [(x - min_val) / (max_val - min_val) if max_val != min_val else 0 
                         for x, min_val, max_val in zip(sample, mins, maxs)]
            X_test_norm.append(normalized)
        
        return X_train_norm, X_test_norm
    
    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def knn_predict(self, X_train, y_train, X_test, k=3):
        """K-Nearest Neighbors implementation"""
        predictions = []
        
        for test_sample in X_test:
            distances = []
            for i, train_sample in enumerate(X_train):
                distance = self.euclidean_distance(test_sample, train_sample)
                distances.append((distance, y_train[i]))
            
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:k]
            
            votes = {}
            for _, label in k_nearest:
                votes[label] = votes.get(label, 0) + 1
            
            predicted_label = max(votes.items(), key=lambda x: x[1])[0]
            predictions.append(predicted_label)
        
        return predictions
    
    def decision_tree_predict(self, X_train, y_train, X_test):
        """Decision Tree implementation"""
        predictions = []
        
        for sample in X_test:
            length, exclamation, win_words, buy_words, num_words, urgency, free = sample
            
            if win_words == 1 or free == 1:
                predictions.append('spam')
            elif buy_words == 1 and urgency == 1:
                predictions.append('spam')
            elif exclamation == 1 and num_words < 4:
                predictions.append('spam')
            else:
                predictions.append('ham')
        
        return predictions
    
    def logistic_regression_predict(self, X_train, y_train, X_test):
        """Logistic Regression implementation"""
        predictions = []
        
        for sample in X_test:
            length, exclamation, win_words, buy_words, num_words, urgency, free = sample
            
            score = (win_words * 3 + buy_words * 2 + free * 2 + 
                    urgency * 1.5 + exclamation * 1 - num_words * 0.1)
            
            predictions.append('spam' if score > 2.5 else 'ham')
        
        return predictions
    
    def calculate_accuracy(self, y_true, y_pred):
        """Calculate accuracy score"""
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true)
    
    def train_and_evaluate(self):
        """Train and evaluate all models"""
        messages, labels = self.prepare_data()
        X = self.extract_features(messages)
        X_train, X_test, y_train, y_test = self.train_test_split(X, labels)
        X_train_norm, X_test_norm = self.normalize_features(X_train, X_test)
        
        print("ПОРІВНЯННЯ АЛГОРИТМІВ МАШИННОГО НАВЧАННЯ")
        print("=" * 60)
        print(f"Розмір даних: {len(messages)} | Тренувальних: {len(X_train)} | Тестових: {len(X_test)}")
        print("=" * 60)
        
        models = {
            'K-Nearest Neighbors': (self.knn_predict, True),
            'Decision Tree': (self.decision_tree_predict, False),
            'Logistic Regression': (self.logistic_regression_predict, False)
        }
        
        for name, (model_func, use_normalized) in models.items():
            if use_normalized:
                predictions = model_func(X_train_norm, y_train, X_test_norm, 3)
            else:
                predictions = model_func(X_train, y_train, X_test)
            
            accuracy = self.calculate_accuracy(y_test, predictions)
            self.results[name] = {'accuracy': accuracy, 'predictions': predictions}
            
            print(f"\n{name}:")
            print(f"Точність: {accuracy:.3f}")
            print(f"Правильних: {sum(1 for t, p in zip(y_test, predictions) if t == p)}/{len(y_test)}")
    
    def print_comparison(self):
        """Print comparison results"""
        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТИ ПОРІВНЯННЯ")
        print("=" * 50)
        
        best_accuracy = 0
        best_model = ""
        
        for name, result in self.results.items():
            accuracy = result['accuracy']
            bar = '█' * int(accuracy * 30) + ' ' * (30 - int(accuracy * 30))
            print(f"{name:22} [{bar}] {accuracy:.3f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = name
        
        print(f"\nНайкраща модель: {best_model} ({best_accuracy:.3f})")
        return best_model

def demo_ml_models():
    """Demonstrate ML models"""
    print("МАШИННЕ НАВЧАННЯ: ПОРІВНЯННЯ АЛГОРИТМІВ")
    ml_system = MLComparison()
    ml_system.train_and_evaluate()
    best_model = ml_system.print_comparison()
    return ml_system, best_model

if __name__ == "__main__":
    demo_ml_models()