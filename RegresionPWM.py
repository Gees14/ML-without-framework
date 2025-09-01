"""
Jorge Ignacio Reyes Pérez - A00573981


Calibración PWM → Velocidad (rad/s) con Regresión Lineal desde cero
- Sin librerías de ML
- Descenso de gradiente batch
- Métricas: MSE, MAE, R²
- Gráficas: (1) datos + recta ajustada, (2) pérdida vs iteración
- Guarda PNG y (si hay GUI) muestra ventanas
"""

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ------------------------ Utilidades ------------------------ #
def train_test_split(X, y, test_size=0.25, rng=rng):
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standardize(X, mean=None, std=None, eps=1e-8):
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return (X - mean) / std, mean, std

def add_bias(X):
    
    return np.hstack([np.ones((X.shape[0], 1)), X])

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))

# ------------------- Modelo (GD desde cero) ------------------- #
class LinearRegressionGD:
    def __init__(self, lr=0.05, max_iter=50_000, tol=1e-10, verbose=True, log_every=500):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.log_every = log_every
        self.w = None              # (incluye bias en la 1a posición)
        self.loss_history = []     # para graficar la pérdida

    def fit(self, X, y):
        n, d = X.shape
        limit = math.sqrt(6.0 / (d + 1))
        self.w = rng.uniform(-limit, limit, size=(d,))
        prev_loss = np.inf

        if self.verbose:
            print(f"[INFO] Entrenando | muestras={n}, features(con bias)={d}")

        for it in range(1, self.max_iter + 1):
            y_hat = X @ self.w
            error = y_hat - y
            grad = (2.0 / n) * (X.T @ error)
            self.w -= self.lr * grad

            loss = mse(y, y_hat)
            self.loss_history.append(loss)

            if self.verbose and (it in (1, 2, 3, 5, 10, 20, 50, 100) or it % self.log_every == 0):
                print(f"[{it:5d}] MSE={loss:.6f}")

            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"[INFO] Early stopping en iter {it}, Δ={abs(prev_loss - loss):.2e}")
                break
            prev_loss = loss

        if self.verbose:
            print("[INFO] Entrenamiento finalizado.")

    def predict(self, X):
        return X @ self.w

# ----------------- Dataset práctico (PWM→ω) ------------------ #
def make_pwm_speed_dataset(n_samples=120, noise_std=0.25, pwm_min=0, pwm_max=100, rng=rng):
    """
    Simula datos de calibración PWM (% 0-100) → velocidad (rad/s).
    Modelo real: ω = b_true + a_true * PWM + ruido
    - a_true ~ 0.12 rad/s por %PWM (pendiente)
    - b_true ~ -1.0 rad/s (fricción/pérdidas -> umbral de arranque)
    """
    a_true = 0.12
    b_true = -1.0

    # Muestras de PWM en el rango [pwm_min, pwm_max]
    pwm = rng.uniform(pwm_min, pwm_max, size=(n_samples, 1))
    noise = rng.normal(0, noise_std, size=(n_samples,))
    omega = b_true + a_true * pwm[:, 0] + noise

    # Clip opcional para simular que por debajo de cierto PWM no gira (no lo aplicamos aquí)
    # omega = np.where(pwm[:, 0] < 8, 0.0 + noise, omega)

    X = pwm  # feature: PWM (%)
    y = omega  # target: velocidad (rad/s)
    return X, y, a_true, b_true

# --------------------------- Main ---------------------------- #
def main():
    print("Entrenando calibración PWM→Velocidad...\n")

    # 1) Dataset práctico (PWM→ω)
    X, y, a_true, b_true = make_pwm_speed_dataset(
        n_samples=160, noise_std=0.25, pwm_min=0, pwm_max=100
    )

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # 3) Estandarizar SOLO features (PWM)
    X_train_std, mean, std = standardize(X_train)
    X_test_std, _, _ = standardize(X_test, mean, std)

    # 4) Agregar bias
    X_train_b = add_bias(X_train_std)
    X_test_b  = add_bias(X_test_std)

    # 5) Entrenar
    model = LinearRegressionGD(lr=0.05, max_iter=30_000, tol=1e-10, verbose=True, log_every=1000)
    model.fit(X_train_b, y_train)

    # 6) Evaluar
    y_tr_pred = model.predict(X_train_b)
    y_te_pred = model.predict(X_test_b)

    print("\n=== RESULTADOS ===")
    print(f"Train MSE: {mse(y_train, y_tr_pred):.4f} | MAE: {mae(y_train, y_tr_pred):.4f} | R²: {r2_score(y_train, y_tr_pred):.4f}")
    print(f"Test  MSE: {mse(y_test,  y_te_pred):.4f} | MAE: {mae(y_test,  y_te_pred):.4f} | R²: {r2_score(y_test,  y_te_pred):.4f}")

    # 7) Convertir pesos del espacio estandarizado al espacio real (PWM en %)
    # X_b = [1, (PWM - mean)/std]
    w_std = model.w
    b_std = w_std[0]
    a_std = w_std[1]  # peso sobre la feature estandarizada

    # ω ≈ b_real + a_real * PWM
    a_real = a_std / std[0]
    b_real = b_std - a_std * (mean[0] / std[0])

    print("\nEcuación ajustada (espacio real):")
    print(f"ω ≈ {b_real:.4f} + ({a_real:.4f}) · PWM   [rad/s]")
    print("\nParámetros verdaderos (simulación):")
    print(f"b_true = {b_true:.4f}, a_true = {a_true:.4f}")

    # 8) Graficar: Datos + recta ajustada
    # Para una línea suave, ordenamos PWM y predecimos en orden
    pwm_all = X[:, 0]
    pwm_grid = np.linspace(pwm_all.min(), pwm_all.max(), 200)
    # Predicción en espacio real: ω_hat = b_real + a_real * PWM
    omega_hat_grid = b_real + a_real * pwm_grid

    plt.figure(figsize=(8, 5))
    plt.scatter(pwm_all, y, alpha=0.6, label="Datos (PWM vs ω)")
    plt.plot(pwm_grid, omega_hat_grid, linewidth=2.5, label="Recta ajustada")
    plt.xlabel("PWM (%)")
    plt.ylabel("Velocidad ω (rad/s)")
    plt.title("Calibración PWM → Velocidad (Regresión Lineal)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("calibracion_pwm_vs_velocidad.png", dpi=150)
    try:
        plt.show()
    except Exception as e:
        print(f"[WARN] No se pudo mostrar la figura (guardada como PNG). Detalle: {e}")

    # 9) Graficar curva de pérdida
    plt.figure(figsize=(8, 4.5))
    plt.plot(model.loss_history)
    plt.xlabel("Iteración")
    plt.ylabel("MSE (train)")
    plt.title("Curva de entrenamiento (pérdida vs iteración)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("perdida_entrenamiento.png", dpi=150)
    try:
        plt.show()
    except Exception as e:
        print(f"[WARN] No se pudo mostrar la figura (guardada como PNG). Detalle: {e}")

if __name__ == "__main__":
    main()
