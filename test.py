import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def plot_real_evolutionary():
    """Графік з реальними даними еволюційних алгоритмів"""
    try:
        from evolutionary_algorithms import EvolutionaryOptimizer
        ea = EvolutionaryOptimizer()
        
        # Запускаємо оптимізацію з реальними параметрами
        ea.run_optimization(population_size=30, generations=10)
        
        if ea.logbook:
            gen = ea.logbook.select("gen")
            fits_min = ea.logbook.select("min")
            fits_avg = ea.logbook.select("avg")
            fits_std = ea.logbook.select("std")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Графік 1: Значення фітнесу
            ax1.plot(gen, fits_min, 'b-', linewidth=3, marker='o', label='Найкраще значення')
            ax1.plot(gen, fits_avg, 'r-', linewidth=2, marker='s', label='Середнє значення')
            ax1.fill_between(gen, 
                            np.array(fits_avg) - np.array(fits_std),
                            np.array(fits_avg) + np.array(fits_std),
                            alpha=0.2, color='red', label='±1 стандартне відхилення')
            
            ax1.set_xlabel('Покоління')
            ax1.set_ylabel('Значення функції Растрігіна')
            ax1.set_title('ЕВОЛЮЦІЯ ФУНКЦІЇ РАСТРІГІНА\n(Реальні дані оптимізації)', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Графік 2: Покращення
            improvement = [(fits_min[0] - value) / fits_min[0] * 100 for value in fits_min]
            ax2.plot(gen, improvement, 'g-', linewidth=3, marker='D')
            ax2.set_xlabel('Покоління')
            ax2.set_ylabel('Покращення (%)')
            ax2.set_title('ВІДНОСНЕ ПОКРАЩЕННЯ ЯКОСТІ', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Анотації
            final_improvement = improvement[-1]
            ax2.annotate(f'Кінцеве покращення: {final_improvement:.1f}%', 
                        xy=(gen[-1], improvement[-1]), 
                        xytext=(gen[-1]-3, improvement[-1]-10),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            
            # Виводимо найкращий результат
            best_fitness = fits_min[-1]
            best_individual = ea.hof[0] if ea.hof else "Не знайдено"
            
            plt.suptitle(f'ЕВОЛЮЦІЙНА ОПТИМІЗАЦІЯ\nНайкраще рішення: {best_individual}\nЗначення функції: {best_fitness:.4f}', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            print(f"\nРЕАЛЬНІ РЕЗУЛЬТАТИ EVOLUTIONARY:")
            print(f"Початкове значення: {fits_min[0]:.4f}")
            print(f"Кінцеве значення: {best_fitness:.4f}")
            print(f"Покращення: {final_improvement:.1f}%")
            print(f"Найкраще рішення: {best_individual}")
            
    except Exception as e:
        print(f"Помилка: {e}")

if __name__ == "__main__":
    plot_real_evolutionary()