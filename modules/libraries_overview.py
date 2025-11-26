"""
Practical Assignment 15: AI Libraries Overview
Порівняння DEAP, PyGAD та TensorFlow на задачі апроксимації функції y = x^2.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import random

# Імпорти бібліотек
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pygad
from deap import base, creator, tools, algorithms

class LibrariesComparison:
    def __init__(self):
        # Дані для навчання: парабола y = x^2 на проміжку [-10, 10]
        self.X = np.linspace(-10, 10, 100)
        self.y = self.X ** 2
        
        # Результати
        self.results = {}

    def calculate_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # --- 1. DEAP (Генетичний алгоритм) ---
    def run_deap(self):
        print("\n[DEAP] Запуск оптимізації коефіцієнтів a*x^2 + b*x + c...")
        start_time = time.time()

        # Налаштування DEAP
        # Ми шукаємо 3 коефіцієнти: a, b, c
        if hasattr(creator, "FitnessMin"): del creator.FitnessMin
        if hasattr(creator, "Individual"): del creator.Individual
            
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, -10, 10)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def eval_func(individual):
            a, b, c = individual
            y_pred = a * (self.X**2) + b * self.X + c
            return (self.calculate_mse(self.y, y_pred),)

        toolbox.register("evaluate", eval_func)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=50)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)

        # Запуск на 50 поколінь
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, stats=stats, verbose=False)
        
        best_ind = tools.selBest(pop, 1)[0]
        duration = time.time() - start_time
        
        # Прогноз
        a, b, c = best_ind
        y_pred = a * (self.X**2) + b * self.X + c
        mse = self.calculate_mse(self.y, y_pred)
        
        print(f"[DEAP] Знайдені коефіцієнти: a={a:.2f}, b={b:.2f}, c={c:.2f}")
        self.results['DEAP'] = {'time': duration, 'mse': mse, 'pred': y_pred}

    # --- 2. PyGAD (Еволюційний алгоритм) ---
    def run_pygad(self):
        print("\n[PyGAD] Запуск оптимізації...")
        start_time = time.time()

        def fitness_func(ga_instance, solution, solution_idx):
            a, b, c = solution
            y_pred = a * (self.X**2) + b * self.X + c
            mse = np.mean((self.y - y_pred) ** 2)
            # PyGAD максимізує фітнес, тому беремо обернене
            return 1.0 / (mse + 1e-8)

        ga_instance = pygad.GA(num_generations=50,
                               num_parents_mating=5,
                               fitness_func=fitness_func,
                               sol_per_pop=50,
                               num_genes=3, # a, b, c
                               init_range_low=-10,
                               init_range_high=10,
                               mutation_percent_genes=20,
                               suppress_warnings=True)

        ga_instance.run()
        
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        duration = time.time() - start_time
        
        a, b, c = solution
        y_pred = a * (self.X**2) + b * self.X + c
        mse = self.calculate_mse(self.y, y_pred)
        
        print(f"[PyGAD] Знайдені коефіцієнти: a={a:.2f}, b={b:.2f}, c={c:.2f}")
        self.results['PyGAD'] = {'time': duration, 'mse': mse, 'pred': y_pred}

    # --- 3. TensorFlow (Нейронна мережа) ---
    def run_tensorflow(self):
        print("\n[TensorFlow] Навчання нейронної мережі...")
        start_time = time.time()
        
        # Проста модель: вхід 1 -> 64 -> 64 -> вихід 1
        model = Sequential([
            Dense(64, activation='relu', input_shape=(1,)),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Навчання (вимкнули verbose для чистоти консолі)
        model.fit(self.X, self.y, epochs=100, batch_size=16, verbose=0)
        
        duration = time.time() - start_time
        y_pred = model.predict(self.X, verbose=0).flatten()
        mse = self.calculate_mse(self.y, y_pred)
        
        print(f"[TensorFlow] Фінальна помилка навчання (MSE): {mse:.4f}")
        self.results['TensorFlow'] = {'time': duration, 'mse': mse, 'pred': y_pred}

    def print_comparison(self):
        print("\n" + "="*60)
        print("ПОРІВНЯЛЬНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ")
        print("="*60)
        print(f"{'Бібліотека':<15} | {'Час (сек)':<10} | {'MSE (Якість)':<15} | {'Складність коду'}")
        print("-" * 60)
        
        complexity = {
            'DEAP': 'Висока (багато налаштувань)', 
            'PyGAD': 'Середня (зручний API)', 
            'TensorFlow': 'Низька (готовий .fit)'
        }
        
        for name, res in self.results.items():
            print(f"{name:<15} | {res['time']:<10.4f} | {res['mse']:<15.4f} | {complexity[name]}")
            
    def visualize(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.X, self.y, 'k--', linewidth=2, label='True: x^2')
        
        colors = {'DEAP': 'blue', 'PyGAD': 'green', 'TensorFlow': 'red'}
        for name, res in self.results.items():
            plt.plot(self.X, res['pred'], label=f"{name} (MSE={res['mse']:.2f})", color=colors[name], alpha=0.7)
            
        plt.title('Апроксимація y=x^2 різними методами')
        plt.legend()
        plt.grid(True)
        plt.show()

def demo_libraries_overview():
    print("\n" + "="*60)
    print("ПРАКТИЧНЕ ЗАНЯТТЯ 15: ОГЛЯД БІБЛІОТЕК")
    print("="*60)
    
    comp = LibrariesComparison()
    
    # Запуск усіх методів
    comp.run_deap()
    comp.run_pygad()
    comp.run_tensorflow()
    
    # Вивід результатів
    comp.print_comparison()
    comp.visualize()

if __name__ == "__main__":
    demo_libraries_overview()