"""
Neural Network Module for MNIST digit recognition
Практичне заняття 6: Нейронні мережі та глибинне навчання
"""

import numpy as np
import matplotlib.pyplot as plt
import os

class MNISTNeuralNetwork:
    def __init__(self):
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        
    def check_tensorflow(self):
        """Перевірити наявність TensorFlow"""
        try:
            import tensorflow as tf
            return True, tf
        except ImportError:
            print("TensorFlow не встановлено!")
            print("Встановіть: pip install tensorflow")
            return False, None
    
    def load_and_prepare_data(self):
        """Завантаження та підготовка даних MNIST"""
        tf_available, tf = self.check_tensorflow()
        if not tf_available:
            return None, None, None, None
            
        print("Завантаження даних MNIST...")
        from tensorflow.keras.datasets import mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        
        # Нормалізація пікселів до діапазону [0, 1]
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0
        
        print(f"Розмір тренувальних даних: {self.x_train.shape}")
        print(f"Розмір тестових даних: {self.x_test.shape}")
        print(f"Унікальні мітки: {np.unique(self.y_train)}")
        
        return self.x_train, self.y_train, self.x_test, self.y_test
    
    def build_model(self):
        """Побудова моделі нейронної мережі"""
        tf_available, tf = self.check_tensorflow()
        if not tf_available:
            return None
            
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Flatten
        
        self.model = Sequential([
            Flatten(input_shape=(28, 28)),  # Перетворення 28x28 в 784
            Dense(128, activation='relu'),  # Прихований шар з 128 нейронами
            Dense(10, activation='softmax') # Вихідний шар з 10 нейронами (0-9)
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Архітектура моделі:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, epochs=5):
        """Навчання моделі"""
        if self.model is None:
            print("Модель не побудована!")
            return None
            
        print(f"\nПочаток навчання на {epochs} епохах...")
        
        self.history = self.model.fit(
            self.x_train, 
            self.y_train,
            epochs=epochs,
            validation_data=(self.x_test, self.y_test),
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self):
        """Оцінка моделі на тестових даних"""
        if self.model is None:
            print("Модель не навчена!")
            return None
            
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"\nРезультати оцінки:")
        print(f"Втрати на тестових даних: {test_loss:.4f}")
        print(f"Точність на тестових даних: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy
    
    def plot_training_history(self):
        """Візуалізація процесу навчання"""
        if self.history is None:
            print("Історія навчання недоступна!")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Графік точності
        ax1.plot(self.history.history['accuracy'], label='Тренувальна точність')
        ax1.plot(self.history.history['val_accuracy'], label='Валідаційна точність')
        ax1.set_title('Точність моделі')
        ax1.set_xlabel('Епоха')
        ax1.set_ylabel('Точність')
        ax1.legend()
        ax1.grid(True)
        
        # Графік втрат
        ax2.plot(self.history.history['loss'], label='Тренувальні втрати')
        ax2.plot(self.history.history['val_loss'], label='Валідаційні втрати')
        ax2.set_title('Втрати моделі')
        ax2.set_xlabel('Епоха')
        ax2.set_ylabel('Втрати')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict_and_visualize(self, num_examples=10):
        """Прогнозування та візуалізація результатів"""
        if self.model is None:
            print("Модель не навчена!")
            return
            
        # Випадковий вибір тестових зображень
        indices = np.random.choice(len(self.x_test), num_examples, replace=False)
        test_images = self.x_test[indices]
        true_labels = self.y_test[indices]
        
        # Прогнозування
        predictions = self.model.predict(test_images)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Візуалізація
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(num_examples):
            axes[i].imshow(test_images[i], cmap='gray')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            
            color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
            axes[i].set_title(
                f"Прогноз: {predicted_labels[i]}\nСправжня: {true_labels[i]}", 
                color=color,
                fontsize=10
            )
        
        plt.tight_layout()
        plt.show()
        
        # Статистика
        correct_predictions = np.sum(predicted_labels == true_labels)
        accuracy = correct_predictions / num_examples
        print(f"Точність на {num_examples} випадкових прикладах: {accuracy:.2f}")
        
        return predicted_labels, true_labels
    
    def demo_without_tensorflow(self):
        """Демонстрація без TensorFlow"""
        print("ДЕМОНСТРАЦІЯ НЕЙРОННОЇ МЕРЕЖІ (без TensorFlow)")
        print("=" * 50)
        print("Для повної роботи потрібен TensorFlow!")
        print("Встановіть: pip install tensorflow")
        print("\nОчікувані результати:")
        print("• Точність на MNIST: >97%")
        print("• Час навчання: 2-5 хвилин")
        print("• Архітектура: 128 нейронів у прихованому шарі")
        print("=" * 50)

def demo_neural_network():
    """Demonstrate ML models"""
    print("НЕЙРОННА МЕРЕЖА ДЛЯ РОЗПІЗНАВАННЯ ЦИФР")
    nn = MNISTNeuralNetwork()
    
    # Перевірка TensorFlow
    tf_available, _ = nn.check_tensorflow()
    
    if tf_available:
        # Повна демонстрація з TensorFlow
        nn.load_and_prepare_data()
        nn.build_model()
        nn.train_model(epochs=2)  # Менше епох для швидкості
        nn.evaluate_model()
        nn.predict_and_visualize(num_examples=5)
    else:
        # Демонстрація без TensorFlow
        nn.demo_without_tensorflow()
    
    return nn

if __name__ == "__main__":
    demo_neural_network()