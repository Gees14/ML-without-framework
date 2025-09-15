import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ------------------------ Dataset ------------------------ #
def make_pwm_speed_dataset(n_samples=120, noise_std=0.25, pwm_min=0, pwm_max=100, rng=rng):
    """
    Simula datos de calibración PWM (% 0-100) → velocidad (rad/s).
    Modelo real: ω = b_true + a_true * PWM + ruido
    - a_true ~ 0.12 rad/s por %PWM (pendiente)
    - b_true ~ -1.0 rad/s (fricción/pérdidas -> umbral de arranque)
    """
    a_true = 0.12
    b_true = -1.0
    pwm = rng.uniform(pwm_min, pwm_max, size=(n_samples, 1))
    noise = rng.normal(0, noise_std, size=(n_samples,))
    omega = b_true + a_true * pwm[:, 0] + noise
    X = pwm
    y = omega
    return X, y, a_true, b_true

def train_test_split(X, y, test_size=0.25, rng=rng):
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ------------------------ Entrenamiento ------------------------ #
def fit_sgd_batch(X_train, y_train, lr=0.05, max_epochs=3000, tol=1e-10, log_every=200):
    """
    Entrena un SGDRegressor en modo 'batch' (una actualización por época con todo el set).
    Devuelve: modelo entrenado, historial de pérdidas (MSE en train).
    """
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)

    # SGDRegressor con squared_error, sin regularización para acercarse al GD "puro".
    sgd = SGDRegressor(
        loss="squared_error",
        penalty=None,
        learning_rate="constant",
        eta0=lr,
        fit_intercept=True,
        max_iter=1,         
        tol=None,
        random_state=RNG_SEED,
        warm_start=False    
    )

    # Inicializar con una pasada para que cree coeficientes
    sgd.partial_fit(X_train_std, y_train)

    loss_hist = []
    prev_loss = np.inf
    for epoch in range(1, max_epochs + 1):
        # Batch Gradient: una sola actualización por época usando TODO el batch
        sgd.partial_fit(X_train_std, y_train)

        
        y_hat = sgd.predict(X_train_std)
        loss = mean_squared_error(y_train, y_hat)
        loss_hist.append(loss)

        if epoch in (1, 2, 3, 5, 10, 20, 50, 100) or (epoch % log_every == 0):
            print(f"[{epoch:5d}] MSE={loss:.6f}")

        if abs(prev_loss - loss) < tol:
            print(f"[INFO] Early stopping en época {epoch}, Δ={abs(prev_loss - loss):.2e}")
            break
        prev_loss = loss

    return sgd, scaler, loss_hist

# ------------------------ Main ------------------------ #
def main():
    print("Entrenando calibración PWM→Velocidad con scikit-learn...\n")

    # 1) Dataset 
    X, y, a_true, b_true = make_pwm_speed_dataset(
        n_samples=160, noise_std=0.25, pwm_min=0, pwm_max=100
    )

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # 3) Entrenar 
    model, scaler, loss_hist = fit_sgd_batch(
        X_train, y_train, lr=0.05, max_epochs=30000, tol=1e-10, log_every=1000
    )

    # 4) Evaluar en train/test 
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    y_tr_pred = model.predict(X_train_std)
    y_te_pred = model.predict(X_test_std)

    print("\n=== RESULTADOS ===")
    print(f"Train MSE: {mean_squared_error(y_train, y_tr_pred):.4f} | "
          f"MAE: {mean_absolute_error(y_train, y_tr_pred):.4f} | "
          f"R²:  {r2_score(y_train, y_tr_pred):.4f}")
    print(f"Test  MSE: {mean_squared_error(y_test,  y_te_pred):.4f} | "
          f"MAE: {mean_absolute_error(y_test,  y_te_pred):.4f} | "
          f"R²:  {r2_score(y_test,  y_te_pred):.4f}")

    # 5) Convertir coeficientes del espacio estandarizado al espacio real (PWM %)
    # y_hat = intercept_std + coef_std * ((PWM - mean)/std)
    coef_std = float(model.coef_[0])
    intercept_std = float(model.intercept_)
    mean = float(scaler.mean_[0])
    std = float(scaler.scale_[0])

    a_real = coef_std / std
    b_real = intercept_std - coef_std * (mean / std)

    print("\nEcuación ajustada (espacio real):")
    print(f"ω ≈ {b_real:.4f} + ({a_real:.4f}) · PWM   [rad/s]")
    print("\nParámetros verdaderos (simulación):")
    print(f"b_true = {b_true:.4f}, a_true = {a_true:.4f}")

    # 6) Graficar: Datos + recta ajustada (en espacio real)
    pwm_all = X[:, 0]
    pwm_grid = np.linspace(pwm_all.min(), pwm_all.max(), 200)
    omega_hat_grid = b_real + a_real * pwm_grid

    plt.figure(figsize=(8, 5))
    plt.scatter(pwm_all, y, alpha=0.6, label="Datos (PWM vs ω)")
    plt.plot(pwm_grid, omega_hat_grid, linewidth=2.5, label="Recta ajustada (sklearn)")
    plt.xlabel("PWM (%)")
    plt.ylabel("Velocidad ω (rad/s)")
    plt.title("Calibración PWM → Velocidad (Regresión Lineal con scikit-learn)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("calibracion_pwm_vs_velocidad_sklearn.png", dpi=150)
    try:
        plt.show()
    except Exception as e:
        print(f"[WARN] No se pudo mostrar la figura (guardada como PNG). Detalle: {e}")

    # 7) Graficar curva de pérdida (MSE en train vs época)
    plt.figure(figsize=(8, 4.5))
    plt.plot(loss_hist)
    plt.xlabel("Época")
    plt.ylabel("MSE (train)")
    plt.title("Curva de entrenamiento (pérdida vs época) - SGDRegressor")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("perdida_entrenamiento_sklearn.png", dpi=150)
    try:
        plt.show()
    except Exception as e:
        print(f"[WARN] No se pudo mostrar la figura (guardada como PNG). Detalle: {e}")

if __name__ == "__main__":
    main()
