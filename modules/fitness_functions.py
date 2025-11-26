"""
Practical Assignment 11: Fitness Functions in Logistics
Оптимізація маршруту доставки з різними фітнес-функціями.
Задача: Знайти оптимальний порядок об'їзду 10 точок.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math

class LogisticsOptimizer:
    def __init__(self):
        self.num_points = 10
        self.pop_size = 50
        self.generations = 50
        self.crossover_prob = 0.8
        self.mutation_prob = 0.05
        
        # Параметри логістики
        self.average_speed = 60.0  # км/год
        self.alpha = 0.6 # Вага відстані
        self.beta = 0.4  # Вага часу
        
        # Генерація карти (фіксована для відтворюваності)
        np.random.seed(42)
        self.points = np.random.rand(self.num_points, 2) * 100 # Координати 0-100 км
        
        # Генерація затримок на точках (наприклад, час розвантаження 5-30 хв)
        # У годинах: 5/60 = 0.08, 30/60 = 0.5
        self.delays = np.random.uniform(0.08, 0.5, self.num_points)

    def calculate_distance(self, route):
        """Евклідова відстань всього маршруту"""
        dist = 0
        for i in range(len(route) - 1):
            p1 = self.points[route[i]]
            p2 = self.points[route[i+1]]
            dist += np.sqrt(np.sum((p1 - p2)**2))
        return dist

    def calculate_time(self, route, distance):
        """Час = Відстань / Швидкість + Сума затримок"""
        # Час у дорозі
        travel_time = distance / self.average_speed
        # Час на розвантаження у точках маршруту
        service_time = np.sum(self.delays) 
        return travel_time + service_time

    # --- ФІТНЕС-ФУНКЦІЇ ---

    def fitness_distance(self, route):
        """Критерій 1: Мінімізація відстані"""
        dist = self.calculate_distance(route)
        # +1 щоб уникнути ділення на нуль
        return 1 / (1 + dist), dist, 0

    def fitness_time(self, route):
        """Критерій 2: Мінімізація часу"""
        dist = self.calculate_distance(route)
        time = self.calculate_time(route, dist)
        return 1 / (1 + time), dist, time

    def fitness_balanced(self, route):
        """Критерій 3: Баланс (Alpha * Dist + Beta * Time)"""
        dist = self.calculate_distance(route)
        time = self.calculate_time(route, dist)
        
        # Комбінована вартість. 
        # Примітка: Відстань (~300) і Час (~6) мають різні масштаби.
        # Для коректної роботи часто потрібна нормалізація, 
        # але використаємо формулу з завдання.
        cost = (self.alpha * dist) + (self.beta * time * 10) # *10 для балансу масштабу
        return 1 / (1 + cost), dist, time

    # --- ГЕНЕТИЧНІ ОПЕРАТОРИ (PERMUTATION BASED) ---

    def create_population(self):
        """Створення випадкових маршрутів"""
        base_route = list(range(self.num_points))
        population = []
        for _ in range(self.pop_size):
            route = base_route[:]
            random.shuffle(route)
            population.append(route)
        return population

    def ordered_crossover(self, parent1, parent2):
        """
        Впорядкований кросовер (OX1) для задач маршрутизації.
        Зберігає порядок міст і уникає дублікатів.
        """
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [-1] * size
        # Копіюємо частину першого батька
        child[start:end] = parent1[start:end]
        
        # Заповнюємо решту з другого батька, зберігаючи порядок
        current_pos = end
        for city in parent2:
            if city not in child:
                if current_pos >= size:
                    current_pos = 0
                child[current_pos] = city
                current_pos += 1
                
        return child

    def swap_mutation(self, route):
        """Обмін двох міст місцями"""
        if random.random() < self.mutation_prob:
            idx1, idx2 = random.sample(range(len(route)), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
        return route

    # --- ГОЛОВНИЙ ЦИКЛ ---

    def run_scenario(self, fitness_type):
        population = self.create_population()
        best_fitness_history = []
        best_route = None
        min_metric = float('inf') # Зберігає найкращу дистанцію або час

        for gen in range(self.generations):
            # 1. Оцінка
            scored_pop = []
            for ind in population:
                if fitness_type == 'distance':
                    fit, d, t = self.fitness_distance(ind)
                    metric = d
                elif fitness_type == 'time':
                    fit, d, t = self.fitness_time(ind)
                    metric = t
                else:
                    fit, d, t = self.fitness_balanced(ind)
                    metric = (self.alpha * d) + (self.beta * t * 10)

                scored_pop.append((ind, fit, d, t))
                
                if metric < min_metric:
                    min_metric = metric
                    best_route = ind

            # Сортування за фітнесом (спадання)
            scored_pop.sort(key=lambda x: x[1], reverse=True)
            best_fitness_history.append(scored_pop[0][1])

            # 2. Відбір (Турнірний)
            new_pop = []
            elite_size = 2 # Елітаризм
            new_pop.extend([x[0] for x in scored_pop[:elite_size]])
            
            while len(new_pop) < self.pop_size:
                # Турнір 3
                candidates = random.sample(scored_pop, 3)
                winner = max(candidates, key=lambda x: x[1])[0]
                new_pop.append(winner)
            
            # 3. Кросовер
            offspring = []
            # Зберігаємо еліту без змін
            offspring.extend(new_pop[:elite_size])
            
            for i in range(elite_size, self.pop_size):
                if random.random() < self.crossover_prob:
                    parent2 = new_pop[random.randint(0, self.pop_size-1)]
                    child = self.ordered_crossover(new_pop[i], parent2)
                    offspring.append(child)
                else:
                    offspring.append(new_pop[i])
            
            # 4. Мутація
            for i in range(elite_size, self.pop_size):
                offspring[i] = self.swap_mutation(offspring[i])
                
            population = offspring

        return best_fitness_history, best_route

    def plot_route(self, route, title, ax):
        """Малювання карти маршруту"""
        # Додаємо повернення в початок для замкнутого маршруту
        plot_route = route + [route[0]]
        
        x = self.points[plot_route, 0]
        y = self.points[plot_route, 1]
        
        ax.plot(x, y, 'o-', mfc='r')
        ax.set_title(title)
        
        # Підпис точок
        for i, txt in enumerate(route):
            ax.annotate(str(txt), (self.points[txt, 0]+1, self.points[txt, 1]+1))

def demo_fitness_functions():
    print("\n" + "="*60)
    print("ПРАКТИЧНЕ ЗАНЯТТЯ 11: ФІТНЕС-ФУНКЦІЇ (ЛОГІСТИКА)")
    print("="*60)
    
    optimizer = LogisticsOptimizer()
    scenarios = ['distance', 'time', 'balanced']
    results = {}
    routes = {}
    
    # Створення вікна графіків
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Оптимізація логістики: Порівняння фітнес-функцій")
    
    for i, scenario in enumerate(scenarios):
        print(f"Запуск сценарію: {scenario.upper()}...")
        history, best_route = optimizer.run_scenario(scenario)
        results[scenario] = history
        routes[scenario] = best_route
        
        # Обчислення фінальних метрик
        dist = optimizer.calculate_distance(best_route)
        time = optimizer.calculate_time(best_route, dist)
        
        print(f"  -> Найкращий фітнес: {history[-1]:.6f}")
        print(f"  -> Дистанція: {dist:.2f} км")
        print(f"  -> Час: {time:.2f} год")
        
        # Графік збіжності
        axes[0, i].plot(history, color='blue')
        axes[0, i].set_title(f"Збіжність ({scenario})")
        axes[0, i].set_xlabel("Покоління")
        axes[0, i].set_ylabel("Fitness")
        axes[0, i].grid(True)
        
        # Графік маршруту
        optimizer.plot_route(best_route, f"Маршрут ({scenario})\n{dist:.1f}km, {time:.1f}h", axes[1, i])

    print("-" * 60)
    print("Візуалізація результатів...")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo_fitness_functions()