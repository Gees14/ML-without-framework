import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ------------------------ Dataset ------------------------ #
def make_pwm_speed_dataset(n_samples=160, noise_std=0.25, pwm_min=0, pwm_max=100, rng=rng):
    a_true = 0.12
    b_true = -1.0
    pwm = rng.uniform(pwm_min, pwm_max, size=(n_samples, 1))
    noise = rng.normal(0, noise_std, size=(n_samples,))
    omega = b_true + a_true * pwm[:, 0] + noise
    return pwm, omega, a_true, b_true

def train_test_split(X, y, test_size=0.25, rng=rng):
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ------------------------ Main ------------------------ #
def main():
    # 1) Dataset
    X, y, a_true, b_true = make_pwm_speed_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 2) Estándarización
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # 3) Entrenar modelo LinearRegression
    model = LinearRegression()
    model.fit(X_train_std, y_train)

    # 4) Predicciones
    y_tr_pred = model.predict(X_train_std)
    y_te_pred = model.predict(X_test_std)

    # 5) Evaluar
    print("=== LINEAR REGRESSION (OLS) ===")
    print(f"Train MSE: {mean_squared_error(y_train, y_tr_pred):.4f} | "
          f"MAE: {mean_absolute_error(y_train, y_tr_pred):.4f} | "
          f"R²: {r2_score(y_train, y_tr_pred):.4f}")
    print(f"Test  MSE: {mean_squared_error(y_test, y_te_pred):.4f} | "
          f"MAE: {mean_absolute_error(y_test, y_te_pred):.4f} | "
          f"R²: {r2_score(y_test, y_te_pred):.4f}")

    # 6) Convertir a espacio real
    coef_std = float(model.coef_[0])
    intercept_std = float(model.intercept_)
    mean = float(scaler.mean_[0])
    std = float(scaler.scale_[0])

    a_real = coef_std / std
    b_real = intercept_std - coef_std * (mean / std)

    print("\nEcuación ajustada (espacio real):")
    print(f"ω ≈ {b_real:.4f} + ({a_real:.4f}) · PWM [rad/s]")
    print(f"Parámetros verdaderos: b_true = {b_true:.4f}, a_true = {a_true:.4f}")

    # 7) Graficar datos + recta ajustada
    pwm_grid = np.linspace(X.min(), X.max(), 200)
    omega_hat = b_real + a_real * pwm_grid

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, alpha=0.6, label="Datos (PWM vs ω)")
    plt.plot(pwm_grid, omega_hat, "r", linewidth=2.5, label="Recta ajustada (OLS)")
    plt.xlabel("PWM (%)")
    plt.ylabel("Velocidad ω (rad/s)")
    plt.title("Calibración PWM → Velocidad (LinearRegression)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("calibracion_pwm_vs_velocidad_OLS.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
