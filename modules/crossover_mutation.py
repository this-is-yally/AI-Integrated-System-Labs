"""
Practical Assignment 10: Crossover and Mutation Methods
Порівняння методів кросоверу та мутації.
Цільова функція: f(x) = x^2 + 4sin(2x)
"""

import numpy as np
import matplotlib.pyplot as plt
import random

class CrossoverMutationLab:
    def __init__(self):
        self.bounds = [-5, 5]
        self.pop_size = 30
        self.generations = 30
        self.mutation_rate = 0.1
        self.gene_length = 20  # Довжина бінарного рядка (точність)

    def target_function(self, x):
        """f(x) = x^2 + 4sin(2x)"""
        return x**2 + 4 * np.sin(2 * x)

    # --- ДОПОМІЖНІ ФУНКЦІЇ ДЛЯ ДВІЙКОВОГО КОДУВАННЯ ---
    
    def binary_to_float(self, binary_list):
        """Конвертація бітів у дійсне число в межах bounds"""
        # Перетворюємо список бітів у рядок, потім у ціле число
        binary_string = "".join(str(bit) for bit in binary_list)
        decimal_value = int(binary_string, 2)
        # Масштабуємо у діапазон [-5, 5]
        max_decimal = 2**self.gene_length - 1
        normalized = decimal_value / max_decimal
        return self.bounds[0] + normalized * (self.bounds[1] - self.bounds[0])

    def create_binary_population(self):
        return [np.random.randint(0, 2, self.gene_length).tolist() for _ in range(self.pop_size)]

    def create_real_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], self.pop_size)

    # --- ОПЕРАТОРИ КРОСОВЕРУ ---

    def single_point_crossover(self, p1, p2):
        """Одноточковий кросовер (для бінарних)"""
        point = random.randint(1, self.gene_length - 1)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return c1, c2

    def two_point_crossover(self, p1, p2):
        """Двоточковий кросовер (для бінарних)"""
        pt1 = random.randint(1, self.gene_length - 2)
        pt2 = random.randint(pt1, self.gene_length - 1)
        c1 = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
        c2 = p2[:pt1] + p1[pt1:pt2] + p2[pt2:]
        return c1, c2

    def arithmetic_crossover(self, p1, p2, alpha=0.5):
        """Арифметичний кросовер (для дійсних чисел)"""
        # Замінює однорідний, бо для 1 зміної однорідний не має сенсу
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = (1 - alpha) * p1 + alpha * p2
        return c1, c2

    # --- ОПЕРАТОРИ МУТАЦІЇ ---

    def bit_flip_mutation(self, individual):
        """Бітова мутація"""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i] # Інверсія 0 <-> 1
        return individual

    def swap_mutation(self, individual):
        """Swap-мутація (обмін двох бітів)"""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def gaussian_mutation(self, x, sigma=0.5):
        """Гаусівська мутація (для дійсних чисел)"""
        if random.random() < self.mutation_rate:
            x += np.random.normal(0, sigma)
            # Обмеження межами
            x = np.clip(x, self.bounds[0], self.bounds[1])
        return x

    # --- ОСНОВНИЙ ЦИКЛ ---

    def run_experiment(self, strategy_name):
        """Запуск експерименту з обраною стратегією"""
        
        # Визначаємо тип кодування
        is_binary = strategy_name in ["SinglePoint_BitFlip", "TwoPoint_Swap"]
        
        # Ініціалізація
        if is_binary:
            population = self.create_binary_population()
        else:
            population = self.create_real_population()
            
        best_fitness_history = []
        avg_fitness_history = []

        for gen in range(self.generations):
            # 1. Декодування та Оцінка
            if is_binary:
                decoded_pop = [self.binary_to_float(ind) for ind in population]
                fitness_values = [self.target_function(x) for x in decoded_pop]
            else:
                fitness_values = [self.target_function(x) for x in population]

            # Збереження статистики
            best_fitness_history.append(np.min(fitness_values))
            avg_fitness_history.append(np.mean(fitness_values))

            # 2. Відбір (Турнірний як базовий для всіх)
            new_population = []
            for _ in range(self.pop_size // 2):
                # Простий турнір для вибору батьків
                if is_binary:
                    # Для бінарних треба вибирати і геном, і фітнес
                    tournament = random.sample(list(zip(population, fitness_values)), 3)
                    p1 = min(tournament, key=lambda x: x[1])[0]
                    tournament = random.sample(list(zip(population, fitness_values)), 3)
                    p2 = min(tournament, key=lambda x: x[1])[0]
                else:
                    # Для дійсних простіше (індексація масиву numpy)
                    indices = np.random.choice(len(population), 3, replace=False)
                    p1 = population[indices[np.argmin(np.array(fitness_values)[indices])]]
                    indices = np.random.choice(len(population), 3, replace=False)
                    p2 = population[indices[np.argmin(np.array(fitness_values)[indices])]]

                # 3. Кросовер
                if strategy_name == "SinglePoint_BitFlip":
                    c1, c2 = self.single_point_crossover(p1, p2)
                elif strategy_name == "TwoPoint_Swap":
                    c1, c2 = self.two_point_crossover(p1, p2)
                else: # Real Valued (Arithmetic + Gaussian)
                    c1, c2 = self.arithmetic_crossover(p1, p2)

                # 4. Мутація
                if strategy_name == "SinglePoint_BitFlip":
                    c1 = self.bit_flip_mutation(c1)
                    c2 = self.bit_flip_mutation(c2)
                elif strategy_name == "TwoPoint_Swap":
                    c1 = self.swap_mutation(c1)
                    c2 = self.swap_mutation(c2)
                else: # Real Valued
                    c1 = self.gaussian_mutation(c1)
                    c2 = self.gaussian_mutation(c2)

                new_population.extend([c1, c2])
            
            population = new_population[:self.pop_size]

        return best_fitness_history, avg_fitness_history

    def visualize_comparison(self, results):
        """Побудова порівняльних графіків"""
        plt.figure(figsize=(12, 6))
        
        colors = {'SinglePoint_BitFlip': 'blue', 'TwoPoint_Swap': 'green', 'Real_Gaussian': 'red'}
        labels = {
            'SinglePoint_BitFlip': '1-Point + BitFlip (Binary)',
            'TwoPoint_Swap': '2-Point + Swap (Binary)',
            'Real_Gaussian': 'Arithmetic + Gaussian (Real)'
        }

        for name, data in results.items():
            plt.plot(data['best'], label=f"{labels[name]}", color=colors.get(name), linewidth=2)
            # Можна додати пунктир для середнього, але графік буде перевантажений
            
        plt.title('Порівняння стратегій кросоверу та мутації')
        plt.xlabel('Покоління')
        plt.ylabel('Найкращий Fitness (f(x))')
        plt.legend()
        plt.grid(True)
        plt.show()

def demo_crossover_mutation():
    print("\n" + "="*60)
    print("ПРАКТИЧНЕ ЗАНЯТТЯ 10: КРОСОВЕР ТА МУТАЦІЯ")
    print("="*60)
    print("Задача: Мінімізація f(x) = x^2 + 4sin(2x) на [-5, 5]")
    print("-" * 60)

    lab = CrossoverMutationLab()
    strategies = ["SinglePoint_BitFlip", "TwoPoint_Swap", "Real_Gaussian"]
    results = {}

    for strat in strategies:
        print(f"Тестування стратегії: {strat}...")
        best, avg = lab.run_experiment(strat)
        results[strat] = {'best': best, 'avg': avg}
        print(f"  -> Фінальний результат: {best[-1]:.4f}")

    print("-" * 60)
    print("Візуалізація результатів...")
    lab.visualize_comparison(results)

if __name__ == "__main__":
    demo_crossover_mutation()