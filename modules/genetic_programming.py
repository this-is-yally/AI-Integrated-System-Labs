"""
Genetic Programming Module
Практичне заняття 8: Генетичне програмування
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class GeneticProgramming:
    def __init__(self):
        self.model = None
        self.found_formula = None
        self.history = None
        
    def check_gplearn(self):
        """Перевірити наявність gplearn"""
        try:
            from gplearn.genetic import SymbolicRegressor
            return True
        except ImportError:
            print("Бібліотека gplearn не встановлена!")
            print("Встановіть: pip install gplearn")
            return False
        
    def generate_sample_data(self, n_samples=100, noise=0.5):
        """Генерація тестових даних з квадратичною залежністю"""
        np.random.seed(42)
        X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
        # Справжня формула: y = 2x² + 3x + 1
        y_true = 2*X**2 + 3*X + 1
        # Додаємо шум
        y = y_true + np.random.normal(0, noise, size=X.shape)
        return X, y.flatten(), y_true.flatten()
    
    def create_model(self, population_size=500, generations=30, 
                    tournament_size=20, stopping_criteria=0.01):
        """Створення моделі генетичного програмування"""
        if not self.check_gplearn():
            return None
            
        from gplearn.genetic import SymbolicRegressor
        
        self.model = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            verbose=1,
            random_state=42,
            function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 
                         'abs', 'neg', 'inv', 'sin', 'cos')
        )
        return self.model
    
    def train_model(self, X, y):
        """Навчання моделі"""
        if self.model is None:
            if not self.create_model():
                return None, None, None, 0, 0
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        
        # Розділення на тренувальну та тестову вибірки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Навчання моделі
        print("Початок навчання генетичного програмування...")
        self.model.fit(X_train, y_train)
        
        # Прогнозування
        y_pred = self.model.predict(X_test)
        
        # Оцінка якості
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # Збереження знайденої формули
        self.found_formula = str(self.model._program)
        
        return X_test, y_test, y_pred, r2, mse
    
    def plot_results(self, X, y_true, X_test, y_test, y_pred, r2, mse):
        """Побудова графіків результатів як у завданні"""
        # Створюємо велике вікно для всіх графіків
        plt.figure(figsize=(16, 12))
        
        # Графік 1: Основне порівняння (як у завданні)
        plt.subplot(2, 2, 1)
        plt.scatter(X, y_true, label="Справжні дані", color="blue", alpha=0.7, s=50)
        
        # Прогноз для всіх точок для плавної кривої
        X_smooth = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
        y_smooth = self.model.predict(X_smooth)
        plt.plot(X_smooth, y_smooth, label="Прогноз моделі", color="red", linewidth=3)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("Символьна регресія: знайдена формула", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Графік 2: Детальний прогноз на тестових даних
        plt.subplot(2, 2, 2)
        plt.scatter(X_test, y_test, color='green', alpha=0.7, s=60, label='Тестові дані')
        plt.scatter(X_test, y_pred, color='red', alpha=0.7, s=30, label='Прогнози')
        
        # Лінії помилок
        for i in range(len(X_test)):
            plt.plot([X_test[i], X_test[i]], [y_test[i], y_pred[i]], 
                    'gray', alpha=0.5, linewidth=1)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Прогнози на тестових даних', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Графік 3: Похибки прогнозу
        plt.subplot(2, 2, 3)
        errors = y_test - y_pred
        plt.scatter(y_test, errors, alpha=0.7, color='purple', s=50)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Справжні значення')
        plt.ylabel('Похибки прогнозу')
        plt.title('Графік похибок прогнозу', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Додаємо статистику похибок
        mean_error = np.mean(np.abs(errors))
        plt.text(0.05, 0.95, f'Середня похибка: {mean_error:.3f}\nСтандартне відхилення: {np.std(errors):.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'),
                verticalalignment='top')
        
        # Графік 4: Інформація про модель
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        # Виводимо інформацію про модель
        info_text = f"РЕЗУЛЬТАТИ ГЕНЕТИЧНОГО ПРОГРАМУВАННЯ\n\n"
        info_text += f"Знайдена формула:\n{self.found_formula}\n\n"
        info_text += f"Якість моделі:\n"
        info_text += f"R² score: {r2:.4f}\n"
        info_text += f"MSE: {mse:.4f}\n\n"
        info_text += f"Параметри навчання:\n"
        info_text += f"Популяція: 500 формул\n"
        info_text += f"Покоління: 30\n"
        info_text += f"Розмір вибірки: {len(X)} точок"
        
        plt.text(0.1, 0.9, info_text, transform=plt.gca().transAxes, fontsize=11,
                bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8),
                verticalalignment='top')
        
        plt.suptitle('ГЕНЕТИЧНЕ ПРОГРАМУВАННЯ: СИМВОЛЬНА РЕГРЕСІЯ', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def print_detailed_results(self, r2, mse):
        """Детальний вивід результатів"""
        print("\n" + "="*70)
        print("РЕЗУЛЬТАТИ ГЕНЕТИЧНОГО ПРОГРАМУВАННЯ")
        print("="*70)
        print(f"Знайдена формула: {self.found_formula}")
        print(f"R² score: {r2:.4f}")
        print(f"Середньоквадратична помилка: {mse:.4f}")
        print("="*70)
        
        # Спрощення формули для читабельності
        simplified = self.found_formula.replace('add', '+')\
                                      .replace('sub', '-')\
                                      .replace('mul', '*')\
                                      .replace('div', '/')\
                                      .replace('sqrt', '√')\
                                      .replace('abs', '|')\
                                      .replace('neg', '-')\
                                      .replace('inv', '1/')\
                                      .replace('sin', 'sin')\
                                      .replace('cos', 'cos')\
                                      .replace('log', 'log')
        print(f"Спрощена формула: {simplified}")
        
        # Висновки про якість
        print("\nВИСНОВКИ:")
        if r2 > 0.9:
            print("✓ Модель демонструє відмінну якість прогнозування")
        elif r2 > 0.7:
            print("✓ Модель демонструє хорошу якість прогнозування") 
        elif r2 > 0.5:
            print("⚠ Модель демонструє задовільну якість прогнозування")
        else:
            print("✗ Модель потребує покращення")
        
        print("✓ Алгоритм успішно знайшов математичну формулу")
        print("✓ Формула може бути використана для прогнозування")
        print("✓ Результат інтерпретований та зрозумілий")

def demo_genetic_programming():
    """Демонстрація роботи генетичного програмування з графіками"""
    print("ГЕНЕТИЧНЕ ПРОГРАМУВАННЯ: СИМВОЛЬНА РЕГРЕСІЯ")
    print("=" * 70)
    
    # Створення об'єкта генетичного програмування
    gp = GeneticProgramming()
    
    # Перевірка наявності бібліотеки
    if not gp.check_gplearn():
        print("\nДля роботи потрібна бібліотека gplearn!")
        print("Встановіть: pip install gplearn")
        return gp
    
    # Генерація тестових даних
    print("Генерація тестових даних...")
    X, y, y_true = gp.generate_sample_data(n_samples=100, noise=1.0)
    print(f"Розмір даних: {X.shape[0]} точок")
    print(f"Справжня формула: y = 2x² + 3x + 1")
    print(f"Діапазон x: [{X.min():.1f}, {X.max():.1f}]")
    
    # Створення та навчання моделі
    print("\nСтворення моделі генетичного програмування...")
    print("Параметри моделі:")
    print("- Розмір популяції: 500 формул")
    print("- Кількість поколінь: 30") 
    print("- Турнірний відбір: 20 формул")
    print("- Ймовірність кросоверу: 0.7")
    print("- Ймовірність мутації: 0.25")
    
    gp.create_model(population_size=500, generations=30)
    
    # Навчання моделі
    print("\n" + "="*50)
    print("ПОЧАТОК НАВЧАННЯ")
    print("="*50)
    X_test, y_test, y_pred, r2, mse = gp.train_model(X, y)
    
    # Побудова графіків
    print("\n" + "="*50)
    print("ПОБУДОВА ГРАФІКІВ РЕЗУЛЬТАТІВ")
    print("="*50)
    gp.plot_results(X, y, X_test, y_test, y_pred, r2, mse)
    
    # Детальний вивід результатів
    gp.print_detailed_results(r2, mse)
    
    return gp

def demonstrate_business_example():
    """Демонстрація бізнес-застосування"""
    print("\n" + "="*70)
    print("ПРАКТИЧНЕ ЗАСТОСУВАННЯ: ПРОГНОЗУВАННЯ ПРОДАЖІВ")
    print("="*70)
    
    # Створюємо бізнес-дані (бюджет реклами -> продажі)
    np.random.seed(123)
    ad_budget = np.linspace(1, 10, 20).reshape(-1, 1)  # Бюджет реклами (тис. грн)
    # Реальна залежність: продажі = 50 + 20*бюджет + 3*бюджет²
    true_sales = 50 + 20*ad_budget + 3*ad_budget**2
    sales = true_sales + np.random.normal(0, 10, ad_budget.shape)  # Додаємо шум
    
    print("Бізнес-дані:")
    print("Бюджет реклами (тис. грн) -> Продажі (од.)")
    for i in range(5):  # Показуємо перші 5 записів
        print(f"{ad_budget[i][0]:.1f} -> {sales[i][0]:.1f}")
    print("...")
    
    gp_business = GeneticProgramming()
    if gp_business.check_gplearn():
        gp_business.create_model(population_size=300, generations=20)
        X_test, y_test, y_pred, r2, mse = gp_business.train_model(ad_budget, sales.flatten())
        
        print(f"\nБІЗНЕС-РЕЗУЛЬТАТИ:")
        print(f"Знайдена формула: {gp_business.found_formula}")
        print(f"Якість прогнозу (R²): {r2:.4f}")
        
        # Простий графік для бізнес-прикладу
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(ad_budget, sales, alpha=0.7, label='Реальні дані', color='blue')
        budget_smooth = np.linspace(ad_budget.min(), ad_budget.max(), 100).reshape(-1, 1)
        sales_smooth = gp_business.model.predict(budget_smooth)
        plt.plot(budget_smooth, sales_smooth, 'r-', linewidth=2, label='Знайдена формула')
        plt.xlabel('Бюджет реклами (тис. грн)')
        plt.ylabel('Продажі (од.)')
        plt.title('Прогнозування продажів від бюджету реклами', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.axis('off')
        info_text = f"БІЗНЕС-ЗАСТОСУВАННЯ\n\n"
        info_text += f"Задача: Прогнозування продажів\n"
        info_text += f"Формула: {gp_business.found_formula}\n\n"
        info_text += f"Якість: R² = {r2:.4f}\n"
        info_text += f"Застосування:\n"
        info_text += f"- Оптимізація рекламного бюджету\n"
        info_text += f"- Прогнозування доходів\n"
        info_text += f"- Планування маркетингу"
        
        plt.text(0.1, 0.9, info_text, transform=plt.gca().transAxes, fontsize=11,
                bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.8),
                verticalalignment='top')
        
        plt.suptitle('ГЕНЕТИЧНЕ ПРОГРАМУВАННЯ В БІЗНЕСІ', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Основна демонстрація
    gp_model = demo_genetic_programming()
    
    # Демонстрація бізнес-застосування
    demonstrate_business_example()