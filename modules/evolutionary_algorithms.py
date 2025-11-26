"""
Evolutionary Algorithms Module
Практичне заняття 7: Еволюційні алгоритми
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import statistics

class EvolutionaryOptimizer:
    def __init__(self):
        self.stats = None
        self.logbook = None
        self.hof = None
        
    def setup_evolutionary_algorithm(self):
        """Налаштування еволюційного алгоритму"""
        # Визначення типу задачі - мінімізація
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Створення toolbox з операторами
        toolbox = base.Toolbox()
        
        # Реєстрація атрибутів та операторів
        toolbox.register("attr_float", random.uniform, -5.12, 5.12)
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                        toolbox.attr_float, n=2)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Функція для оптимізації - Растрігіна
        toolbox.register("evaluate", self.rastrigin_function)
        
        # Генетичні оператори
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        return toolbox
    
    def rastrigin_function(self, individual):
        """Функція Растрігіна - тестова багатоекстремальна функція"""
        A = 10
        result = A * len(individual) + sum(
            [(x**2 - A * np.cos(2 * np.pi * x)) for x in individual]
        )
        return result,
    
    def run_optimization(self, population_size=50, generations=50, 
                        crossover_prob=0.5, mutation_prob=0.2):
        """Запуск еволюційної оптимізації"""
        toolbox = self.setup_evolutionary_algorithm()
        
        # Створення початкової популяції
        population = toolbox.population(n=population_size)
        
        # Налаштування статистики
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
        # Hall of Fame - зберігає найкращі рішення
        self.hof = tools.HallOfFame(1)
        
        # Запуск алгоритму
        population, self.logbook = algorithms.eaSimple(
            population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob,
            ngen=generations, stats=self.stats, halloffame=self.hof, verbose=True
        )
        
        return population
    
    def print_results(self):
        """Вивід результатів оптимізації"""
        if self.hof is None:
            print("Оптимізація не була виконана!")
            return
            
        best_individual = self.hof[0]
        best_fitness = best_individual.fitness.values[0]
        
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТИ ЕВОЛЮЦІЙНОЇ ОПТИМІЗАЦІЇ")
        print("="*60)
        print(f"Найкраще рішення: {best_individual}")
        print(f"Значення функції: {best_fitness:.6f}")
        print(f"Координати: x1 = {best_individual[0]:.6f}, x2 = {best_individual[1]:.6f}")
        print("="*60)
        
        # Аналіз збіжності
        if self.logbook:
            gen = self.logbook.select("gen")
            fits_min = self.logbook.select("min")
            fits_avg = self.logbook.select("avg")
            
            final_min = fits_min[-1]
            final_avg = fits_avg[-1]
            improvement = ((fits_min[0] - final_min) / fits_min[0]) * 100
            
            print(f"Початкове найкраще значення: {fits_min[0]:.2f}")
            print(f"Кінцеве найкраще значення: {final_min:.6f}")
            print(f"Покращення: {improvement:.1f}%")
            print(f"Кінцеве середнє значення: {final_avg:.6f}")
    
    def plot_convergence(self):
        """Побудова графіка збіжності"""
        if self.logbook is None:
            print("Немає даних для побудови графіка!")
            return
            
        gen = self.logbook.select("gen")
        fits_min = self.logbook.select("min")
        fits_avg = self.logbook.select("avg")
        fits_std = self.logbook.select("std")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Графік значень фітнесу
        ax1.plot(gen, fits_min, 'b-', label="Найкраще значення")
        ax1.plot(gen, fits_avg, 'r-', label="Середнє значення")
        ax1.fill_between(gen, 
                        np.array(fits_avg) - np.array(fits_std),
                        np.array(fits_avg) + np.array(fits_std),
                        alpha=0.2, color='red')
        
        ax1.set_xlabel("Покоління")
        ax1.set_ylabel("Значення функції")
        ax1.set_title("Еволюція функції Растрігіна")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Графік покращення
        improvement = [(fits_min[0] - value) / fits_min[0] * 100 for value in fits_min]
        ax2.plot(gen, improvement, 'g-', linewidth=2)
        ax2.set_xlabel("Покоління")
        ax2.set_ylabel("Покращення (%)")
        ax2.set_title("Відносне покращення якості")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_function_landscape(self):
        """Візуалізація ландшафту функції Растрігіна"""
        x = np.linspace(-5.12, 5.12, 100)
        y = np.linspace(-5.12, 5.12, 100)
        X, Y = np.meshgrid(x, y)
        
        Z = 20 + (X**2 - 10 * np.cos(2 * np.pi * X)) + \
                (Y**2 - 10 * np.cos(2 * np.pi * Y))
        
        plt.figure(figsize=(10, 8))
        contour = plt.contour(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(contour, label='Значення функції')
        
        if self.hof:
            best_point = self.hof[0]
            plt.plot(best_point[0], best_point[1], 'ro', markersize=10, 
                    label='Знайдений оптимум')
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Ландшафт функції Растрігіна')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_parameters(self):
        """Аналіз впливу параметрів на ефективність"""
        print("\n" + "="*60)
        print("АНАЛІЗ ПАРАМЕТРІВ АЛГОРИТМУ")
        print("="*60)
        
        if self.logbook is None:
            print("Немає даних для аналізу!")
            return
            
        fits_min = self.logbook.select("min")
        fits_avg = self.logbook.select("avg")
        fits_std = self.logbook.select("std")
        
        # Аналіз збіжності
        convergence_gen = None
        for i in range(1, len(fits_min)):
            if abs(fits_min[i] - fits_min[i-1]) < 0.001:
                convergence_gen = i
                break
        
        print(f"Покоління збіжності: {convergence_gen if convergence_gen else 'Не досягнуто'}")
        print(f"Фінальне стандартне відхилення: {fits_std[-1]:.6f}")
        print(f"Коливання якості: {(max(fits_avg) - min(fits_avg)):.4f}")
        
        # Рекомендації щодо параметрів
        print("\nРЕКОМЕНДАЦІЇ:")
        if convergence_gen and convergence_gen < 10:
            print("• Зменшити розмір популяції або кількість поколінь")
        elif not convergence_gen:
            print("• Збільшити кількість поколінь або ймовірність мутації")
        
        if fits_std[-1] > 5:
            print("• Збільшити інтенсивність відбору (tournsize)")
        
        if max(fits_avg) - min(fits_avg) < 1:
            print("• Збільшити різноманітність (мутація/кросовер)")

def demo_evolutionary_algorithm():
    """Демонстрація роботи еволюційного алгоритму"""
    print("ЕВОЛЮЦІЙНИЙ АЛГОРИТМ: ОПТИМІЗАЦІЯ ФУНКЦІЇ РАСТРІГІНА")
    print("=" * 70)
    
    # Створення оптимізатора
    optimizer = EvolutionaryOptimizer()
    
    # Запуск оптимізації
    print("Запуск еволюційної оптимізації...")
    population = optimizer.run_optimization(
        population_size=50,
        generations=50,
        crossover_prob=0.5,
        mutation_prob=0.2
    )
    
    # Вивід результатів
    optimizer.print_results()
    
    # Візуалізація
    optimizer.plot_convergence()
    optimizer.plot_function_landscape()
    
    # Аналіз параметрів
    optimizer.analyze_parameters()
    
    return optimizer

def compare_parameters():
    """Порівняння різних параметрів алгоритму"""
    print("\n" + "="*70)
    print("ПОРІВНЯННЯ РІЗНИХ ПАРАМЕТРІВ АЛГОРИТМУ")
    print("="*70)
    
    parameters = [
        {"pop_size": 30, "cx_prob": 0.7, "mut_prob": 0.1, "name": "Великий кросовер"},
        {"pop_size": 30, "cx_prob": 0.3, "mut_prob": 0.3, "name": "Велика мутація"},
        {"pop_size": 100, "cx_prob": 0.5, "mut_prob": 0.2, "name": "Велика популяція"},
    ]
    
    results = []
    
    for params in parameters:
        print(f"\nТестування: {params['name']}")
        optimizer = EvolutionaryOptimizer()
        population = optimizer.run_optimization(
            population_size=params["pop_size"],
            generations=30,  # Менше поколінь для швидкості
            crossover_prob=params["cx_prob"],
            mutation_prob=params["mut_prob"]
        )
        
        best_fitness = optimizer.hof[0].fitness.values[0]
        results.append({
            "name": params["name"],
            "fitness": best_fitness,
            "params": params
        })
        
        print(f"Результат: {best_fitness:.6f}")
    
    # Вивід порівняння
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТИ ПОРІВНЯННЯ")
    print("="*50)
    
    for result in sorted(results, key=lambda x: x["fitness"]):
        print(f"{result['name']:20} : {result['fitness']:.6f}")
    
    return results

if __name__ == "__main__":
    # Основна демонстрація
    optimizer = demo_evolutionary_algorithm()
    
    # Додаткове порівняння параметрів
    compare_parameters()