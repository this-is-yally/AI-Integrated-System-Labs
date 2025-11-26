"""
Rule-based system for spam detection
Практичні заняття 2-3: Експертна система на основі правил
"""

class RulesEngine:
    def __init__(self):
        self.spam_keywords = [
            'виграй', 'виграв', 'безкоштовно', 'знижк', 'акція', 
            'терміново', 'кредит', 'заробіток', 'подарунок', 'приз'
        ]
        self.urgency_words = ['терміново', 'швидко', 'негайно', 'тільки сьогодні']
        
    def check_spam_rules(self, message):
        """Check message against spam rules"""
        message_lower = message.lower()
        
        # Rule 1: Contains spam keywords
        spam_keywords_count = sum(1 for keyword in self.spam_keywords if keyword in message_lower)
        
        # Rule 2: Contains urgency words
        urgency_count = sum(1 for word in self.urgency_words if word in message_lower)
        
        # Rule 3: Multiple exclamation marks
        exclamation_count = message.count('!')
        
        # Rule 4: Short message with commercial intent
        words = message.split()
        is_short_commercial = len(words) <= 5 and any(word in message_lower for word in ['купи', 'продаж'])
        
        # Decision making based on rules
        score = 0
        score += spam_keywords_count * 2
        score += urgency_count * 1.5
        score += min(exclamation_count, 3) * 1
        score += 3 if is_short_commercial else 0
        
        if score >= 3:
            return "spam", score
        else:
            return "ham", score
    
    def analyze_message(self, message):
        """Complete analysis of message"""
        result, score = self.check_spam_rules(message)
        
        analysis = {
            'message': message,
            'result': result,
            'confidence_score': score,
            'details': {
                'length': len(message),
                'word_count': len(message.split()),
                'has_exclamation': '!' in message,
                'spam_keywords_found': [word for word in self.spam_keywords if word in message.lower()]
            }
        }
        
        return analysis

def demo_rules_engine():
    """Demonstrate the rules engine"""
    engine = RulesEngine()
    
    test_messages = [
        "Виграй iPhone безкоштовно!",
        "Зустрінемось о 15:00",
        "Терміново купи зі знижкою",
        "Надішли звіт будь ласка"
    ]
    
    print("ПРАВИЛА ДЛЯ ВИЯВЛЕННЯ СПАМУ")
    print("=" * 50)
    
    for message in test_messages:
        analysis = engine.analyze_message(message)
        print(f"\nПовідомлення: {message}")
        print(f"Результат: {analysis['result'].upper()}")
        print(f"Оцінка впевненості: {analysis['confidence_score']}")
        print(f"Знайдені ключові слова: {analysis['details']['spam_keywords_found']}")
        print("-" * 40)

if __name__ == "__main__":
    demo_rules_engine()