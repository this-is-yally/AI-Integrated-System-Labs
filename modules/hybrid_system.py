"""
Practical Assignment 12: Hybrid Methods (Neuro-Evolution)
Гібридна система: Генетичний алгоритм оптимізує гіперпараметри Нейронної Мережі.
Задача: Апроксимація функції sin(x).
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings

# Ігноруємо попередження про збіжність (для чистоти виводу при швидкому навчанні)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class HybridOptimizer:
    def __init__(self):
        # Діапазони пошуку
        self.neuron_range = [3, 20]  # Кількість нейронів
        self.lr_range = [0.001, 0.1] # Швидкість навчання
        
        # Параметри ГА
        self.pop_size = 15
        self.generations = 10
        self.top_k = 6 # Скільки найкращих переходить далі
        self.mutation_rate = 0.2
        
        # Підготовка даних (Sin(x))
        self.X = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
        self.y = np.sin(self.X).ravel()

    def create_population(self):
        """Створення популяції: [neurons (int), learning_rate (float)]"""
        pop = []
        for _ in range(self.pop_size):
            neurons = random.randint(self.neuron_range[0], self.neuron_range[1])
            lr = random.uniform(self.lr_range[0], self.lr_range[1])
            pop.append([neurons, lr])
        return pop

    def evaluate_model(self, genome):
        """
        Тренування нейромережі з параметрами з геному.
        Повертає MSE (Mean Squared Error).
        """
        neurons, lr = genome
        # Створення простої мережі (1 прихований шар)
        model = MLPRegressor(
            hidden_layer_sizes=(int(neurons),),
            activation='tanh', # Tanh добре підходить для sin(x)
            solver='adam',
            learning_rate_init=lr,
            max_iter=500, # Кількість епох навчання
            random_state=42
        )
        
        model.fit(self.X, self.y)
        prediction = model.predict(self.X)
        
        # Обчислення помилки (MSE)
        mse = np.mean((self.y - prediction) ** 2)
        return mse, model

    def crossover(self, parent1, parent2):
        """Схрещування параметрів"""
        # Нейрони: середнє арифметичне (округлене)
        child_neurons = int((parent1[0] + parent2[0]) / 2)
        
        # Learning Rate: середнє арифметичне
        child_lr = (parent1[1] + parent2[1]) / 2
        
        return [child_neurons, child_lr]

    def mutate(self, genome):
        """Мутація параметрів"""
        neurons, lr = genome
        
        if random.random() < self.mutation_rate:
            # Зміна нейронів на +/- 1..2
            neurons += random.randint(-2, 2)
            neurons = max(self.neuron_range[0], min(self.neuron_range[1], neurons))
            
        if random.random() < self.mutation_rate:
            # Зміна LR на 20%
            lr *= random.uniform(0.8, 1.2)
            lr = max(self.lr_range[0], min(self.lr_range[1], lr))
            
        return [neurons, lr]

    def run_optimization(self):
        print("Ініціалізація популяції...")
        population = self.create_population()
        best_history = []
        best_genome = None
        best_model = None
        min_mse = float('inf')

        for gen in range(self.generations):
            print(f"Покоління {gen+1}/{self.generations}...", end="")
            
            # 1. Оцінювання
            scored_pop = []
            for genome in population:
                mse, _ = self.evaluate_model(genome)
                scored_pop.append((genome, mse))
            
            # Сортування (найменша помилка - найкраща)
            scored_pop.sort(key=lambda x: x[1])
            
            # Збереження найкращого
            current_best_mse = scored_pop[0][1]
            best_history.append(current_best_mse)
            print(f" Найкраща MSE: {current_best_mse:.6f} | Params: {scored_pop[0][0]}")
            
            if current_best_mse < min_mse:
                min_mse = current_best_mse
                best_genome = scored_pop[0][0]
            
            # 2. Відбір (Top K)
            parents = [x[0] for x in scored_pop[:self.top_k]]
            
            # 3. Еволюція (Кросовер + Мутація)
            new_pop = parents[:] # Елітаризм
            
            while len(new_pop) < self.pop_size:
                p1, p2 = random.sample(parents, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_pop.append(child)
                
            population = new_pop

        # Фінальне тренування найкращої моделі
        print("\nТренування фінальної моделі...")
        final_mse, final_model = self.evaluate_model(best_genome)
        
        return best_genome, final_mse, final_model, best_history

    def run_manual_baseline(self):
        """Запуск моделі з параметрами "вручну" (за завданням)"""
        print("Тренування базової моделі (Manual)...")
        # 10 нейронів, LR 0.01
        genome = [10, 0.01]
        mse, model = self.evaluate_model(genome)
        return genome, mse, model

    def visualize_results(self, best_model, manual_model, history, best_params):
        plt.figure(figsize=(14, 5))
        
        # Графік 1: Прогнози
        plt.subplot(1, 2, 1)
        plt.plot(self.X, self.y, 'k--', label='True Sin(x)', linewidth=2)
        
        pred_best = best_model.predict(self.X)
        pred_manual = manual_model.predict(self.X)
        
        plt.plot(self.X, pred_best, 'g-', label=f'Hybrid (GA)\nN={best_params[0]}, LR={best_params[1]:.4f}')
        plt.plot(self.X, pred_manual, 'r:', label='Manual (Baseline)\nN=10, LR=0.01')
        
        plt.title('Апроксимація Sin(x): Гібрид vs База')
        plt.legend()
        plt.grid(True)
        
        # Графік 2: Збіжність
        plt.subplot(1, 2, 2)
        plt.plot(history, 'b-o')
        plt.title('Еволюція помилки (MSE)')
        plt.xlabel('Покоління')
        plt.ylabel('MSE (Lower is better)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def demo_hybrid_system():
    print("\n" + "="*60)
    print("ПРАКТИЧНЕ ЗАНЯТТЯ 12: ГІБРИДНІ МЕТОДИ (NeuroEvolution)")
    print("="*60)
    print("Ціль: Підібрати архітектуру нейромережі для функції Sin(x)")
    
    optimizer = HybridOptimizer()
    
    # 1. Запуск Генетичного Алгоритму
    print("\n--- ЕТАП 1: Генетична оптимізація ---")
    best_params, best_mse, best_model, history = optimizer.run_optimization()
    
    # 2. Запуск Базової моделі (для порівняння)
    print("\n--- ЕТАП 2: Базова модель (порівняння) ---")
    manual_params, manual_mse, manual_model = optimizer.run_manual_baseline()
    
    # 3. Вивід результатів
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТИ ПОРІВНЯННЯ")
    print("="*60)
    print(f"{'Модель':<15} | {'Нейрони':<10} | {'LR':<10} | {'MSE (Помилка)':<15}")
    print("-" * 60)
    print(f"{'Manual':<15} | {manual_params[0]:<10} | {manual_params[1]:<10.4f} | {manual_mse:.6f}")
    print(f"{'Hybrid (GA)':<15} | {best_params[0]:<10} | {best_params[1]:<10.4f} | {best_mse:.6f}")
    print("-" * 60)
    
    improvement = ((manual_mse - best_mse) / manual_mse) * 100
    if improvement > 0:
        print(f"✅ Гібридний метод покращив результат на {improvement:.2f}%")
    else:
        print(f"⚠️ Гібридний метод не перевершив базу (це нормально для стохастики)")
        
    # 4. Візуалізація
    print("Візуалізація результатів...")
    optimizer.visualize_results(best_model, manual_model, history, best_params)

if __name__ == "__main__":
    demo_hybrid_system()