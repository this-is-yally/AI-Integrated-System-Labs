"""
Practical Assignment 14: Science & Tech (Energy Consumption Forecasting)
Прогнозування погодинного енергоспоживання (Трек A).
Джерело натхнення: PJM Hourly Energy Consumption Data (Kaggle).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings

# Фільтруємо попередження для чистоти виводу
warnings.filterwarnings('ignore')

class EnergyForecaster:
    def __init__(self):
        self.seed = 42
        np.random.seed(self.seed)
        
    def generate_energy_data(self, n_years=2):
        """
        Генерація реалістичного датасету енергоспоживання.
        Включає:
        - Річна сезонність (зима/літо піки)
        - Тижнева сезонність (будні > вихідні)
        - Добова сезонність (день > ніч)
        - Випадковий шум та аномалії
        """
        print("Генерація синтетичного датасету PJM Energy...")
        
        dates = pd.date_range(start='2022-01-01', periods=n_years*24*365, freq='H')
        n = len(dates)
        
        # 1. Річна сезонність (синусоїда з піком взимку та влітку)
        day_of_year = dates.dayofyear
        yearly = 5000 * np.cos(2 * np.pi * day_of_year / 365) 
        
        # 2. Тижнева сезонність
        weekday = dates.dayofweek
        weekly = np.where(weekday < 5, 2000, 0) # Будні +2000 МВт
        
        # 3. Добова сезонність (складний профіль: ранок пік, вечір пік)
        hour = dates.hour
        daily = 3000 * np.sin(2 * np.pi * (hour - 6) / 24) + \
                1000 * np.sin(2 * np.pi * (hour - 18) / 12)
        
        # 4. Базове навантаження + Тренд
        base_load = 30000 + np.linspace(0, 2000, n)
        
        # 5. Температурний фактор (проста імітація)
        # Взимку (дні 0-60, 300-365) холод збільшує споживання
        temp_factor = np.where((day_of_year < 60) | (day_of_year > 300), 2000, 0)
        
        # 6. Шум
        noise = np.random.normal(0, 1000, n)
        
        consumption = base_load + yearly + weekly + daily + temp_factor + noise
        
        df = pd.DataFrame({'datetime': dates, 'MW': consumption})
        df.set_index('datetime', inplace=True)
        return df

    def eda(self, df):
        """Exploratory Data Analysis"""
        print("\n--- EDA (Розвідувальний аналіз) ---")
        print(df.describe())
        
        plt.figure(figsize=(15, 10))
        
        # Графік 1: Загальний вигляд
        plt.subplot(2, 2, 1)
        plt.plot(df.index, df['MW'], alpha=0.5, linewidth=0.5)
        plt.title("Погодинне споживання енергії (MW)")
        plt.ylabel("МВт")
        
        # Графік 2: Тиждень (деталізація)
        plt.subplot(2, 2, 2)
        subset = df[:168*2] # 2 тижні
        plt.plot(subset.index, subset['MW'])
        plt.title("Деталізація: 2 тижні")
        plt.xticks(rotation=45)
        
        # Графік 3: Розподіл
        plt.subplot(2, 2, 3)
        sns.histplot(df['MW'], kde=True, bins=50)
        plt.title("Розподіл споживання")
        
        # Графік 4: Boxplot по годинах
        plt.subplot(2, 2, 4)
        df['hour'] = df.index.hour
        sns.boxplot(x='hour', y='MW', data=df)
        plt.title("Сезонність по годинах доби")
        
        plt.tight_layout()
        plt.show()

    def feature_engineering(self, df):
        """Створення ознак (Feature Engineering)"""
        print("\nСтворення ознак...")
        df_feat = df.copy()
        
        # Часові ознаки
        df_feat['hour'] = df_feat.index.hour
        df_feat['dayofweek'] = df_feat.index.dayofweek
        df_feat['quarter'] = df_feat.index.quarter
        df_feat['month'] = df_feat.index.month
        df_feat['year'] = df_feat.index.year
        df_feat['dayofyear'] = df_feat.index.dayofyear
        
        # Лагові ознаки (минуле)
        # lag_1: година тому, lag_24: добу тому, lag_168: тиждень тому
        df_feat['lag_1'] = df_feat['MW'].shift(1)
        df_feat['lag_24'] = df_feat['MW'].shift(24)
        df_feat['lag_168'] = df_feat['MW'].shift(168)
        
        # Ковзні вікна (Rolling Statistics)
        df_feat['rolling_mean_24'] = df_feat['MW'].shift(1).rolling(window=24).mean()
        df_feat['rolling_std_24'] = df_feat['MW'].shift(1).rolling(window=24).std()
        
        # Видаляємо NaN
        df_feat.dropna(inplace=True)
        return df_feat

    def evaluate_model(self, y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        print(f"{model_name:<20} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}% | R2: {r2:.4f}")
        return mae, rmse, mape, r2

    def run_pipeline(self):
        # 1. Дані
        df = self.generate_energy_data()
        
        # 2. EDA
        self.eda(df)
        
        # 3. Features
        df_ml = self.feature_engineering(df)
        
        # Спліт (Train/Test) - часовий!
        # Test = останні 3 місяці (~2000 годин)
        test_size = 24 * 30 * 3 
        train = df_ml.iloc[:-test_size]
        test = df_ml.iloc[-test_size:]
        
        features = ['hour', 'dayofweek', 'month', 'dayofyear', 
                   'lag_1', 'lag_24', 'lag_168', 'rolling_mean_24', 'rolling_std_24']
        target = 'MW'
        
        X_train, y_train = train[features], train[target]
        X_test, y_test = test[features], test[target]
        
        print(f"\nРозмір Train: {X_train.shape}, Test: {X_test.shape}")
        
        # 4. Baseline (Наївний)
        print("\n--- РЕЗУЛЬТАТИ ОЦІНЮВАННЯ ---")
        print(f"{'Model':<20} | {'MAE':<10} | {'RMSE':<10} | {'MAPE':<10} | {'R2':<10}")
        print("-" * 75)
        
        # Наївний 1: Прогноз = значенню 24 години тому (добова сезонність)
        y_pred_naive = X_test['lag_24']
        self.evaluate_model(y_test, y_pred_naive, "Baseline (Lag-24)")
        
        # 5. ML Model (Gradient Boosting)
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=self.seed)
        model.fit(X_train, y_train)
        y_pred_ml = model.predict(X_test)
        
        self.evaluate_model(y_test, y_pred_ml, "Gradient Boosting")
        
        # 6. Важливість ознак
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n--- FEATURE IMPORTANCE ---")
        print(feature_importance)
        
        # 7. Візуалізація прогнозу
        plt.figure(figsize=(15, 6))
        # Показуємо тільки частину тесту для ясності (перші 2 тижні)
        subset_test = y_test[:168*2]
        subset_pred = y_pred_ml[:168*2]
        subset_naive = y_pred_naive[:168*2]
        
        plt.plot(subset_test.index, subset_test, label='Факт (Real)', color='black', linewidth=1.5)
        plt.plot(subset_test.index, subset_pred, label='ML Forecast', color='green', linewidth=1.5)
        plt.plot(subset_test.index, subset_naive, label='Baseline', color='red', linestyle='--', alpha=0.5)
        
        plt.title("Прогноз споживання: Факт vs ML vs Baseline (фрагмент)")
        plt.ylabel("MW")
        plt.legend()
        plt.grid(True)
        plt.show()

def demo_science_energy():
    print("\n" + "="*60)
    print("ПРАКТИЧНЕ ЗАНЯТТЯ 14: НАУКА І ТЕХНІКА")
    print("="*60)
    print("Задача: Прогнозування енергоспоживання (Time Series Regression)")
    
    forecaster = EnergyForecaster()
    forecaster.run_pipeline()

if __name__ == "__main__":
    demo_science_energy()