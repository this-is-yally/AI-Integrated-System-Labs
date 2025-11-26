"""
Practical Assignment 9: Selection Methods in Evolutionary Algorithms
Порівняння методів відбору: Рулетка, Турнір, Ранжування
"""

import numpy as np
import matplotlib.pyplot as plt
import random

class SelectionComparison:
    def __init__(self):
        self.bounds = [-5, 5]
        self.pop_size = 30
        self.generations = 30
        self.mutation_rate = 0.1
        self.mutation_strength = 0.5
        
    def target_function(self, x):
        """
        Функція для мінімізації: f(x) = x^2 + 4sin(3x)
        """
        return x**2 + 4 * np.sin(3 * x)

    def initialize_population(self):
        """Створення початкової популяції"""
        return np.random.uniform(self.bounds[0], self.bounds[1], self.pop_size)

    # --- МЕТОДИ ВІДБОРУ ---

    def roulette_wheel_selection(self, population, fitness_values):
        """
        Відбір за рулеткою (для мінімізації).
        Ми інвертуємо фітнес, щоб менші значення мали більшу ймовірність.
        """
        # Зсув значень, щоб всі були додатні (для коректності ймовірностей)
        # Формула перетворення для мінімізації: 
        # Score = (Max_Fitness - Current_Fitness) + Epsilon
        max_val = np.max(fitness_values)
        adjusted_fitness = max_val - fitness_values + 1e-6
        
        total_fit = np.sum(adjusted_fitness)
        probs = adjusted_fitness / total_fit
        
        # Вибір індексів на основі ймовірностей
        selected_indices = np.random.choice(len(population), size=len(population), p=probs)
        return population[selected_indices]

    def tournament_selection(self, population, fitness_values, k=3):
        """
        Турнірний відбір.
        Обираємо k випадкових, беремо найкращого (з найменшим f(x)).
        """
        new_population = []
        for _ in range(len(population)):
            # Випадкові індекси для турніру
            candidates_indices = np.random.choice(len(population), size=k, replace=False)
            candidates = population[candidates_indices]
            candidates_fitness = fitness_values[candidates_indices]
            
            # Переможець - той, у кого найменше значення функції
            winner_idx = np.argmin(candidates_fitness)
            new_population.append(candidates[winner_idx])
            
        return np.array(new_population)

    def rank_selection(self, population, fitness_values):
        """
        Ранговий відбір.
        Сортуємо за фітнесом, присвоюємо ймовірність на основі рангу.
        """
        # Отримуємо індекси відсортованого масиву (від найкращого до найгіршого)
        # argsort сортує від меншого до більшого (що нам і треба для мінімізації)
        sorted_indices = np.argsort(fitness_values)
        
        # Ранги: найгірший (останній у списку) = 1, найкращий (перший) = N
        # Але argsort дає менші значення спочатку. Тому найкращий має індекс 0.
        # Для ймовірності нам треба, щоб найкращий мав вищий ранг.
        # Тому реверсуємо: Найкращий (min f(x)) -> Ранг N, Найгірший -> Ранг 1
        n = len(population)
        ranks = np.arange(n, 0, -1) # [30, 29, ..., 1] для відсортованого списку
        
        total_rank = np.sum(ranks)
        probs = ranks / total_rank
        
        # Вибираємо індекси зі списку 'sorted_indices' на основі ймовірностей 'probs'
        # Тут ми обираємо, який за порядком ранг виграв
        chosen_ranks_indices = np.random.choice(len(population), size=len(population), p=probs)
        
        # Перетворюємо назад у реальні індекси популяції
        final_indices = sorted_indices[chosen_ranks_indices]
        
        return population[final_indices]

    # --- ОПЕРАТОРИ ---

    def crossover(self, parents):
        """Середнє арифметичне двох батьків"""
        offspring = []
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            # Якщо непарна кількість, беремо першого як пару
            p2 = parents[i+1] if i+1 < len(parents) else parents[0]
            
            # Кросовер (середина відрізка)
            child1 = (p1 + p2) / 2.0
            child2 = (p1 + p2) / 2.0 # Можна додати випадковість, але за умовою - середнє
            
            offspring.extend([child1, child2])
        return np.array(offspring[:len(parents)])

    def mutate(self, population):
        """Випадкова зміна (Gaussian mutation)"""
        for i in range(len(population)):
            if np.random.random() < self.mutation_rate:
                noise = np.random.normal(0, self.mutation_strength)
                population[i] += noise
                # Обмеження межами [-5, 5]
                population[i] = np.clip(population[i], self.bounds[0], self.bounds[1])
        return population

    # --- ГОЛОВНИЙ ЦИКЛ ---

    def run_algorithm(self, selection_method_name):
        population = self.initialize_population()
        best_history = []
        avg_history = []
        
        for generation in range(self.generations):
            # 1. Оцінювання
            fitness_values = self.target_function(population)
            
            # Збереження статистики
            best_history.append(np.min(fitness_values))
            avg_history.append(np.mean(fitness_values))
            
            # 2. Відбір
            if selection_method_name == 'Roulette':
                population = self.roulette_wheel_selection(population, fitness_values)
            elif selection_method_name == 'Tournament':
                population = self.tournament_selection(population, fitness_values, k=3)
            elif selection_method_name == 'Rank':
                population = self.rank_selection(population, fitness_values)
                
            # 3. Кросовер
            population = self.crossover(population)
            
            # 4. Мутація
            population = self.mutate(population)
            
        return best_history, avg_history

    def visualize_results(self, results):
        """Побудова графіків для порівняння"""
        generations = range(self.generations)
        
        plt.figure(figsize=(12, 5))
        
        # Графік 1: Найкращий фітнес (Best Fitness)
        plt.subplot(1, 2, 1)
        for name, data in results.items():
            plt.plot(generations, data['best'], label=name, linewidth=2)
        plt.title('Збіжність: Найкраще рішення')
        plt.xlabel('Покоління')
        plt.ylabel('f(x) (мінімізація)')
        plt.legend()
        plt.grid(True)
        
        # Графік 2: Середній фітнес (Avg Fitness)
        plt.subplot(1, 2, 2)
        for name, data in results.items():
            plt.plot(generations, data['avg'], label=name, linestyle='--')
        plt.title('Різноманітність: Середнє значення')
        plt.xlabel('Покоління')
        plt.ylabel('Середній f(x)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Функція для запуску з main.py
def demo_selection_methods():
    print("\n" + "="*60)
    print("ПРАКТИЧНЕ ЗАНЯТТЯ 9: МЕТОДИ ВІДБОРУ")
    print("="*60)
    print("Цільова функція: f(x) = x^2 + 4sin(3x) на інтервалі [-5, 5]")
    print("Завдання: Знайти мінімум (глобальний мінімум близько -4.5)")
    print("-" * 60)
    
    lab = SelectionComparison()
    methods = ['Roulette', 'Tournament', 'Rank']
    results = {}
    
    for method in methods:
        print(f"Запуск методу: {method}...")
        best_hist, avg_hist = lab.run_algorithm(method)
        results[method] = {'best': best_hist, 'avg': avg_hist}
        print(f"  -> Знайдений мінімум: {best_hist[-1]:.4f}")
    
    print("-" * 60)
    print("Візуалізація результатів...")
    lab.visualize_results(results)

if __name__ == "__main__":
    demo_selection_methods()