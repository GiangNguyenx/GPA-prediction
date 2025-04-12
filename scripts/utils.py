from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

def evaluate_model(y_true, y_pred):
    mean_squared_error_value = mean_squared_error(y_true, y_pred)
    mean_absolute_error_value = mean_absolute_error(y_true, y_pred)
    r2_score_value = r2_score(y_true, y_pred)

    print(f"Mean Squared Error: {mean_squared_error_value}")
    print(f"Mean Absolute Error: {mean_absolute_error_value}")
    print(f"R2 Score: {r2_score_value}")

def plot_predictions(model, X_test, y_test):
    y_pred = np.round(model.predict(X_test))

    y_pred = y_pred.flatten()
    y_test = y_test.to_numpy()

    y_pred = np.clip(y_pred, 0, 20)

    # PLOT THE RESULT (Predicted vs Actual)
    y_pred_smooth = pd.Series(y_pred).rolling(window=5, center=True).mean()
    y_test_smooth = pd.Series(y_test).rolling(window=5, center=True).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_smooth, label='Actual Grades', linewidth=2)
    plt.plot(y_pred_smooth, label='Predicted Grades', linewidth=2)
    plt.title('Predicted vs Actual')
    plt.xlabel('Student No.')
    plt.ylabel('Final Grade')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

