from transformers import pipeline, set_seed

def demo_text_generation():
    print("--- ЗАПУСК ДЕМО: ГЕНЕРАТИВНА МОДЕЛЬ (GPT-2) ---")
    
    # 1. Ініціалізація пайплайну (завантаження моделі)
    # Ми використовуємо gpt2, бо вона легка і швидка для демо
    print("Завантаження моделі... Це може зайняти хвилину.")
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42) # Фіксація результату для відтворюваності

    # 2. Вхідні дані
    input_text = "Artificial Intelligence represents the future because"
    print(f"\nВхідна фраза: '{input_text}'")
    print("-" * 50)

    # 3. Генерація тексту
    # max_length - скільки слів генерувати
    # num_return_sequences - скільки варіантів створити
    results = generator(input_text, max_length=50, num_return_sequences=2)

    # 4. Вивід результатів
    for i, result in enumerate(results):
        print(f"\nВаріант #{i+1}:")
        print(result['generated_text'])
    
    print("\n--- КІНЕЦЬ ДЕМО ---")

if __name__ == "__main__":
    demo_text_generation()