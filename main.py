"""
Головний файл проєкту AI
Інтегрована система штучного інтелекту
"""

import sys
import os

def safe_import_tensorflow():
    """Безпечний імпорт TensorFlow"""
    try:
        import tensorflow as tf
        return True, tf.__version__
    except AttributeError as e:
        if "numpy" in str(e):
            return "numpy_conflict", "Конфлікт версій NumPy"
        return False, str(e)
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def check_dependencies():
    """Перевірити наявність бібліотек"""
    print("\nПЕРЕВІРКА БІБЛІОТЕК")
    print("=" * 50)
    
    # Список бібліотек для перевірки
    libraries = [
        ('numpy', 'np'),
        ('matplotlib', 'plt'),
        ('sklearn', 'sklearn'),
        ('deap', 'deap'),
        ('gplearn', 'gplearn'),
        ('pandas', 'pd'),
        ('seaborn', 'sns'),
        ('pygad', 'pygad')
    ]
    
    for lib_name, import_name in libraries:
        try:
            if lib_name == 'numpy':
                import numpy as np
                version = np.__version__
                print(f"✓ {lib_name}: {version}")
            elif lib_name == 'matplotlib':
                import matplotlib
                version = matplotlib.__version__
                print(f"✓ {lib_name}: {version}")
            elif lib_name == 'sklearn':
                import sklearn
                version = sklearn.__version__
                print(f"✓ {lib_name}: {version}")
            elif lib_name == 'deap':
                import deap
                version = deap.__version__
                print(f"✓ {lib_name}: {version}")
            elif lib_name == 'gplearn':
                import gplearn
                version = gplearn.__version__
                print(f"✓ {lib_name}: {version}")
            elif lib_name == 'pandas':
                import pandas
                version = pandas.__version__
                print(f"✓ {lib_name}: {version}")
            elif lib_name == 'seaborn':
                import seaborn
                version = seaborn.__version__
                print(f"✓ {lib_name}: {version}")
            elif lib_name == 'pygad':
                import pygad
                version = pygad.__version__
                print(f"✓ {lib_name}: {version}")
        except ImportError:
            print(f"✗ {lib_name}: НЕ ВСТАНОВЛЕНО (pip install {lib_name})")
        except Exception as e:
            print(f"⚠ {lib_name}: ПРОБЛЕМА - {e}")
    
    # TensorFlow - окрема перевірка
    print("\nTensorFlow:")
    tf_status, tf_info = safe_import_tensorflow()
    if tf_status is True:
        print(f"✓ tensorflow: {tf_info}")
    elif tf_status == "numpy_conflict":
        print(f"⚠ tensorflow: КОНФЛІКТ ВЕРСІЙ NUMPY")
        print("  Рішення: pip install numpy==1.24.3")
    else:
        print(f"✗ tensorflow: НЕ ВСТАНОВЛЕНО - {tf_info}")
    
    print("=" * 50)

def load_module_directly(module_name):
    """Безпосереднє завантаження модуля"""
    try:
        # Базові модулі
        if module_name == 'rules_engine':
            import modules.rules_engine as module
            return module.demo_rules_engine
        elif module_name == 'bayes_classifier':
            import modules.bayes_classifier as module
            return module.demo_bayes_classifier
        elif module_name == 'ml_models':
            import modules.ml_models as module
            return module.demo_ml_models
        elif module_name == 'neural_network':
            try:
                import modules.neural_network as module
                return module.demo_neural_network
            except Exception as e:
                return f"error:{e}"
        # Еволюційні модулі
        elif module_name == 'evolutionary_algorithms':
            import modules.evolutionary_algorithms as module
            return module.demo_evolutionary_algorithm
        elif module_name == 'genetic_programming':
            import modules.genetic_programming as module
            return module.demo_genetic_programming
        elif module_name == 'selection_methods':
            import modules.selection_methods as module
            return module.demo_selection_methods
        elif module_name == 'crossover_mutation':
            import modules.crossover_mutation as module
            return module.demo_crossover_mutation
        elif module_name == 'fitness_functions':
            import modules.fitness_functions as module
            return module.demo_fitness_functions
        elif module_name == 'hybrid_system':
            import modules.hybrid_system as module
            return module.demo_hybrid_system
        # Модулі застосування
        elif module_name == 'business_forecast':
            import modules.business_forecast as module
            return module.demo_business_forecast
        elif module_name == 'science_energy':
            import modules.science_energy as module
            return module.demo_science_energy
        elif module_name == 'libraries_overview':
            import modules.libraries_overview as module
            return module.demo_libraries_overview
            
    except ImportError as e:
        return f"import_error:{e}"
    except Exception as e:
        return f"error:{e}"

def show_system_info():
    """Показати інформацію про систему"""
    print("\n" + "="*60)
    print("ІНТЕГРОВАНА СИСТЕМА ШТУЧНОГО ІНТЕЛЕКТУ")
    print("="*60)
    print("Версія: 4.0.0 (FINAL)")
    print("Розробник: Студент")
    print("\nДоступні модулі:")
    print("1.  Rule-based System (Практичні 2-3) ✓")
    print("2.  Naive Bayes Classifier (Практичне 4) ✓") 
    print("3.  Machine Learning Models (Практичне 5) ✓")
    print("4.  Neural Network (Практичне 6) ✓")
    print("5.  Evolutionary Algorithms (Практичне 7) ✓")
    print("6.  Genetic Programming (Практичне 8) ✓")
    print("7.  Selection Methods (Практичне 9) ✓")
    print("8.  Crossover & Mutation (Практичне 10) ✓")
    print("9.  Fitness Functions (Практичне 11) ✓")
    print("10. Hybrid Systems (Практичне 12) ✓")
    print("11. Business Forecasting (Практичне 13) ✓")
    print("12. Science & Tech Energy (Практичне 14) ✓")
    print("13. AI Libraries Overview (Практичне 15) NEW! ✓")
    print("14. Усі модулі послідовно")
    print("15. Інформація про модулі")
    print("16. Інструкція з встановлення")
    print("0.  Вихід")
    print("="*60)

def run_individual_module(choice):
    """Запуск окремого модуля"""
    modules_map = {
        '1': ('rules_engine', 'RULE-BASED SYSTEM'),
        '2': ('bayes_classifier', 'BAYES CLASSIFIER'),
        '3': ('ml_models', 'ML MODELS'),
        '4': ('neural_network', 'NEURAL NETWORK'),
        '5': ('evolutionary_algorithms', 'EVOLUTIONARY ALGORITHMS'),
        '6': ('genetic_programming', 'GENETIC PROGRAMMING'),
        '7': ('selection_methods', 'SELECTION METHODS (Prac 9)'),
        '8': ('crossover_mutation', 'CROSSOVER & MUTATION (Prac 10)'),
        '9': ('fitness_functions', 'FITNESS FUNCTIONS (Prac 11)'),
        '10': ('hybrid_system', 'HYBRID SYSTEM (Prac 12)'),
        '11': ('business_forecast', 'BUSINESS FORECAST (Prac 13)'),
        '12': ('science_energy', 'SCIENCE ENERGY FORECAST (Prac 14)'),
        '13': ('libraries_overview', 'LIBRARIES OVERVIEW (Prac 15)')
    }

    if choice in modules_map:
        module_name, display_name = modules_map[choice]
        print("\n" + "="*50)
        print(f"ЗАПУСК {display_name}")
        print("="*50)
        demo_func = load_module_directly(module_name)
        
        if callable(demo_func):
            demo_func()
        elif isinstance(demo_func, str):
            print(f"Помилка: {demo_func}")
        else:
            print(f"Не вдалося завантажити модуль {module_name}!")

def run_all_modules():
    """Запуск всіх модулів послідовно"""
    print("\n" + "="*50)
    print("ПОВНИЙ ТЕСТ СИСТЕМИ ШТУЧНОГО ІНТЕЛЕКТУ")
    print("="*50)
    
    modules_list = [
        ('1. RULE-BASED SYSTEM', 'rules_engine'),
        ('2. BAYES CLASSIFIER', 'bayes_classifier'),
        ('3. ML MODELS', 'ml_models'), 
        ('4. NEURAL NETWORK', 'neural_network'),
        ('5. EVOLUTIONARY ALGORITHMS', 'evolutionary_algorithms'),
        ('6. GENETIC PROGRAMMING', 'genetic_programming'),
        ('7. SELECTION METHODS', 'selection_methods'),
        ('8. CROSSOVER & MUTATION', 'crossover_mutation'),
        ('9. FITNESS FUNCTIONS', 'fitness_functions'),
        ('10. HYBRID SYSTEM', 'hybrid_system'),
        ('11. BUSINESS FORECAST', 'business_forecast'),
        ('12. SCIENCE ENERGY', 'science_energy'),
        ('13. LIBRARIES OVERVIEW', 'libraries_overview')
    ]
    
    for name, module_name in modules_list:
        print(f"\n{name}")
        print("-" * 50)
        
        demo_func = load_module_directly(module_name)
        if callable(demo_func):
            try:
                demo_func()
            except Exception as e:
                print(f"Помилка виконання: {e}")
        elif module_name == 'neural_network':
             pass 
        else:
            print(f"Модуль {module_name} не знайдено або помилка завантаження!")
        
        print("-" * 50)
        if module_name != modules_list[-1][1]:
            input("Натисніть Enter для продовження до наступного модуля...")
    
    print("\n" + "="*50)
    print("ТЕСТУВАННЯ ВСІХ МОДУЛІВ ЗАВЕРШЕНО!")
    print("="*50)

def show_installation_guide():
    """Показати інструкцію з встановлення"""
    print("\n" + "="*60)
    print("ІНСТРУКЦІЯ З ВСТАНОВЛЕННЯ ТА ВИКОРИСТАННЯ")
    print("="*60)
    print("Повна команда для встановлення всіх бібліотек:")
    print("\npip install numpy==1.24.3 matplotlib==3.7.1 scikit-learn pandas seaborn deap gplearn tensorflow==2.15.0 protobuf==3.20.3 pygad")
    print("="*60)

def main():
    """Головна функція проєкту"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    modules_dir = os.path.join(current_dir, 'modules')
    sys.path.insert(0, current_dir)
    sys.path.insert(0, modules_dir)
    
    show_system_info()
    check_dependencies()
    
    while True:
        print("\n" + "="*50)
        print("ОБЕРІТЬ МОДУЛЬ ДЛЯ ЗАПУСКУ:")
        print("1-13. Запуск відповідного практичного модуля")
        print("14.   Усі модулі послідовно")
        print("15.   Інформація про модулі")
        print("16.   Інструкція з встановлення")
        print("0.    Вихід")
        print("="*50)
        
        try:
            choice = input("\nВаш вибір (0-16): ").strip()
            
            if choice == '0':
                print("\nДо побачення! 👋")
                break
                
            elif choice in [str(i) for i in range(1, 14)]:
                run_individual_module(choice)
                
            elif choice == '14':
                run_all_modules()

            elif choice == '15':
                show_system_info()
                
            elif choice == '16':
                show_installation_guide()
                
            else:
                print("Некоректний вибір. Спробуйте ще раз.")
                
        except KeyboardInterrupt:
            print("\n\nПрограму перервано. До побачення! 👋")
            break
        except Exception as e:
            print(f"\nНеочікувана помилка: {e}")

if __name__ == "__main__":
    main()