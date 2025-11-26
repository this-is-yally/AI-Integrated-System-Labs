"""
Practical Assignment 13: Business Forecasting (Retail Sales)
Прогнозування продажів магазину з використанням ML (Random Forest).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

class SalesForecaster:
    def __init__(self):
        self.seed = 42
        np.random.seed(self.seed)
        self.model_path = "sales_model.pkl"
        
    def generate_synthetic_data(self, days=365):
        """
        Генерація синтетичних даних продажів:
        - Тренд (ріст)
        - Сезонність (тижнева)
        - Промо-акції
        - Випадковий шум
        """
        date_range = pd.date_range(start='2024-01-01', periods=days, freq='D')
        
        # 1. Лінійний тренд
        trend = np.linspace(50, 100, days)
        
        # 2. Тижнева сезонність (синусоїда + піки на вихідних)
        # 0=Monday, 6=Sunday. Робимо пік у п'ятницю/суботу
        week_day = date_range.dayofweek
        seasonality = 10 * np.sin(2 * np.pi * week_day / 7)
        weekend_boost = np.where(week_day >= 5, 20, 0)
        
        # 3. Промо-акції (випадкові дні, близько 10% часу)
        promo = np.random.choice([0, 1], size=days, p=[0.9, 0.1])
        promo_effect = promo * 40 # Промо дає +40 продажів
        
        # 4. Шум
        noise = np.random.normal(0, 5, days)
        
        # Підсумкові продажі (забезпечуємо, щоб не було < 0)
        sales = trend + seasonality + weekend_boost + promo_effect + noise
        sales = np.maximum(sales, 0)
        
        df = pd.DataFrame({
            'date': date_range,
            'sales': sales,
            'promo': promo,
            'day_of_week': week_day
        })
        df.set_index('date', inplace=True)
        return df

    def create_features(self, df):
        """Feature Engineering: Створення ознак для ML"""
        df_feat = df.copy()
        
        # Лаги (значення в минулому)
        df_feat['lag_1'] = df_feat['sales'].shift(1) # Вчора
        df_feat['lag_7'] = df_feat['sales'].shift(7) # Тиждень тому
        
        # Ковзні середні (Rolling features)
        df_feat['rolling_mean_7'] = df_feat['sales'].shift(1).rolling(window=7).mean()
        
        # Календарні ознаки
        df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
        
        # Видалення NaN, які з'явилися через shift
        df_feat.dropna(inplace=True)
        return df_feat

    def calculate_metrics(self, y_true, y_pred):
        """Розрахунок бізнес-метрик"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE (Mean Absolute Percentage Error)
        # Уникаємо ділення на 0
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        return mae, rmse, mape

    def run_pipeline(self):
        print("1. Генерація даних (365 днів)...")
        df = self.generate_synthetic_data()
        
        print("2. Створення ознак (Lags, Rolling, Calendar)...")
        df_ml = self.create_features(df)
        
        # Розділення на Train/Test (Останні 30 днів - тест)
        test_days = 30
        train = df_ml.iloc[:-test_days]
        test = df_ml.iloc[-test_days:]
        
        features = ['lag_1', 'lag_7', 'rolling_mean_7', 'promo', 'day_of_week', 'is_weekend']
        target = 'sales'
        
        X_train, y_train = train[features], train[target]
        X_test, y_test = test[features], test[target]
        
        # --- BASELINES ---
        print("\n--- BASELINE METRICS ---")
        
        # Naive: Прогноз = значення вчора
        y_pred_naive = X_test['lag_1']
        mae_n, rmse_n, mape_n = self.calculate_metrics(y_test, y_pred_naive)
        print(f"Naive Forecast (Вчорашній день):")
        print(f"  MAE: {mae_n:.2f} | MAPE: {mape_n:.2f}%")
        
        # Seasonal Naive: Прогноз = значення тиждень тому
        y_pred_snaive = X_test['lag_7']
        mae_sn, rmse_sn, mape_sn = self.calculate_metrics(y_test, y_pred_snaive)
        print(f"Seasonal Naive (Минулий тиждень):")
        print(f"  MAE: {mae_sn:.2f} | MAPE: {mape_sn:.2f}%")

        # --- ML MODEL ---
        print("\n--- ML TRAINING (Random Forest) ---")
        model = RandomForestRegressor(n_estimators=100, random_state=self.seed)
        model.fit(X_train, y_train)
        
        # Прогноз
        y_pred_ml = model.predict(X_test)
        mae_ml, rmse_ml, mape_ml = self.calculate_metrics(y_test, y_pred_ml)
        
        print(f"Random Forest Forecast:")
        print(f"  MAE: {mae_ml:.2f} | MAPE: {mape_ml:.2f}%")
        
        # Порівняння
        improvement = ((mae_sn - mae_ml) / mae_sn) * 100
        print(f"\n✅ Покращення відносно сезонного бейзлайну: {improvement:.2f}%")

        # Збереження моделі
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"💾 Модель збережено у файл: {self.model_path}")

        # Важливість ознак
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        print("\nTOP-3 Важливі ознаки:")
        print(importances.head(3))

        return train, test, y_pred_ml, mae_ml

    def visualize(self, train, test, y_pred):
        plt.figure(figsize=(12, 6))
        
        # Показуємо тільки останній місяць навчання для ясності
        plt.plot(train.index[-60:], train['sales'][-60:], label='Історія (Train)', color='gray', alpha=0.5)
        plt.plot(test.index, test['sales'], label='Факт (Test)', color='blue', linewidth=2)
        plt.plot(test.index, y_pred, label='Прогноз ML', color='red', linestyle='--', linewidth=2)
        
        plt.title('Прогноз продажів: Факт vs Модель')
        plt.xlabel('Дата')
        plt.ylabel('Продажі (шт)')
        plt.legend()
        plt.grid(True)
        plt.show()

def demo_business_forecast():
    print("\n" + "="*60)
    print("ПРАКТИЧНЕ ЗАНЯТТЯ 13: БІЗНЕС-ПРОГНОЗУВАННЯ (SALES)")
    print("="*60)
    
    forecaster = SalesForecaster()
    train, test, y_pred, mae = forecaster.run_pipeline()
    
    print("-" * 60)
    print("Візуалізація результатів...")
    forecaster.visualize(train, test, y_pred)

if __name__ == "__main__":
    demo_business_forecast()