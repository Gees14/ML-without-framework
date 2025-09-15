import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
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

def split_train_val_test(X, y, test_size=0.2, val_size=0.2, rng=rng):
    """Split estratificado simple por índices (regresión → aleatorio)."""
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(round(n * test_size))
    n_val  = int(round(n * val_size))

    test_idx = idx[:n_test]
    val_idx  = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    return (X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx],
            X[test_idx],  y[test_idx])

# ------------------------ Entrenamiento ------------------------ #
def train_sgd_with_early_stopping(
    X_tr, y_tr, X_val, y_val,
    lr=0.05, penalty="l2", alpha=1e-4,
    max_epochs=20000, tol=1e-10, patience=1000, log_every=1000
):
    """
    Entrena SGDRegressor en modo "batch" con early stopping sobre VALIDACIÓN.
    Devuelve: mejor modelo (deepcopy), scaler, historial (train_mse, val_mse)
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xva = scaler.transform(X_val)

    sgd = SGDRegressor(
        loss="squared_error",
        penalty=penalty,      
        alpha=alpha,          
        l1_ratio=0.15,        
        learning_rate="constant",
        eta0=lr,
        fit_intercept=True,
        max_iter=1, tol=None,
        random_state=RNG_SEED, warm_start=False
    )
    # una pasada para inicializar
    sgd.partial_fit(Xtr, y_tr)

    best_val = np.inf
    best_model = None
    best_epoch = 0

    hist_tr, hist_val = [], []
    prev_loss = np.inf
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # una actualización con TODO el batch → emula batch gradient
        sgd.partial_fit(Xtr, y_tr)

        yhat_tr = sgd.predict(Xtr)
        yhat_va = sgd.predict(Xva)
        tr_mse = mean_squared_error(y_tr, yhat_tr)
        va_mse = mean_squared_error(y_val, yhat_va)

        hist_tr.append(tr_mse)
        hist_val.append(va_mse)

        # early stopping por validación
        if va_mse + 1e-12 < best_val:
            best_val = va_mse
            best_model = deepcopy(sgd)
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if epoch in (1,2,3,5,10,20,50,100) or epoch % log_every == 0:
            print(f"[{epoch:5d}] train MSE={tr_mse:.6f} | val MSE={va_mse:.6f}")

        # criterio de convergencia muy estricto + paciencia
        if abs(prev_loss - va_mse) < tol or no_improve >= patience:
            print(f"[INFO] Early stopping en época {epoch} (mejor en {best_epoch})")
            break
        prev_loss = va_mse

    return best_model, scaler, (hist_tr, hist_val), best_epoch, best_val

# ------------------------ Diagnóstico ------------------------ #
def diagnose_bias_variance(r2_tr, r2_val, r2_te):
    """
    Heurísticas simples para clasificar bias/varianza.
    - Bias alto si R²_train < 0.8
    - Varianza alta si |R²_train - R²_val| > 0.05 o R²_val << R²_train
    """
    # Bias
    if r2_tr < 0.8:
        bias = "alto"
    elif r2_tr < 0.92:
        bias = "medio"
    else:
        bias = "bajo"

    # Varianza
    gap_tv = abs(r2_tr - r2_val)
    if gap_tv > 0.10:
        variance = "alta"
    elif gap_tv > 0.05:
        variance = "media"
    else:
        variance = "baja"

    # Ajuste global
    # underfit si todos los R² < 0.8; overfit si train muy alto y val/test bajos
    if r2_tr < 0.8 and r2_val < 0.8 and r2_te < 0.8:
        fit = "underfit"
    elif (r2_tr - r2_val > 0.10) or (r2_tr - r2_te > 0.10):
        fit = "overfit"
    else:
        fit = "buen ajuste"

    return bias, variance, fit

def report_metrics(tag, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"{tag:>6} → MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    return mse, mae, r2

# ------------------------ Búsqueda simple de hiperparámetros ------------------------ #
def tune_hyperparams(X_tr, y_tr, X_val, y_val):
    grid = {
        "penalty": ["l2", "elasticnet"],
        "alpha": [1e-6, 1e-5, 1e-4, 1e-3],
        "lr": [0.05, 0.02, 0.01],
    }
    results = []
    for penalty in grid["penalty"]:
        for alpha in grid["alpha"]:
            for lr in grid["lr"]:
                print(f"\n[TUNE] penalty={penalty} alpha={alpha} lr={lr}")
                model, scaler, _, _, best_val = train_sgd_with_early_stopping(
                    X_tr, y_tr, X_val, y_val,
                    lr=lr, penalty=penalty, alpha=alpha,
                    max_epochs=15000, tol=1e-11, patience=2000, log_every=5000
                )
                results.append((best_val, penalty, alpha, lr, model, scaler))
    # ordenar por menor MSE de validación
    results.sort(key=lambda x: x[0])
    best = results[0]
    print("\n[MEJOR CONFIGURACIÓN] "
          f"val MSE={best[0]:.6f} | penalty={best[1]} | alpha={best[2]} | lr={best[3]}")
    return best  # (val_mse, penalty, alpha, lr, model, scaler)

# ------------------------ Main ------------------------ #
def main(plot_curves=True):
    # 1) Datos y splits
    X, y, a_true, b_true = make_pwm_speed_dataset(n_samples=160, noise_std=0.25)
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_train_val_test(X, y, test_size=0.2, val_size=0.2)

    print("=== BASELINE (penalty=L2, alpha=1e-4, lr=0.05) ===")
    base_model, base_scaler, (tr_hist, val_hist), best_epoch, best_val = train_sgd_with_early_stopping(
        X_tr, y_tr, X_val, y_val, lr=0.05, penalty="l2", alpha=1e-4,
        max_epochs=30000, tol=1e-12, patience=4000, log_every=5000
    )

    # 2) Métricas en Train/Val/Test (baseline)
    Xtr_s = base_scaler.transform(X_tr)
    Xva_s = base_scaler.transform(X_val)
    Xte_s = base_scaler.transform(X_te)

    ytr_hat = base_model.predict(Xtr_s)
    yva_hat = base_model.predict(Xva_s)
    yte_hat = base_model.predict(Xte_s)

    print("\n--- MÉTRICAS Baseline ---")
    tr_mse, tr_mae, r2_tr = report_metrics("Train", y_tr, ytr_hat)
    va_mse, va_mae, r2_va = report_metrics(" Val ", y_val, yva_hat)
    te_mse, te_mae, r2_te = report_metrics("Test ", y_te, yte_hat)

    # 3) Diagnósticos
    bias, variance, fit = diagnose_bias_variance(r2_tr, r2_va, r2_te)
    print(f"\nDiagnóstico:")
    print(f"- Bias: {bias}")
    print(f"- Varianza: {variance}")
    print(f"- Ajuste global: {fit}")

    # 4) Tuning / Regularización
    print("\n=== TUNING / REGULARIZACIÓN ===")
    best_val_mse, best_pen, best_alpha, best_lr, best_model, best_scaler = tune_hyperparams(X_tr, y_tr, X_val, y_val)

    # Evaluamos el mejor en Test
    Xtr_b = best_scaler.transform(X_tr)
    Xva_b = best_scaler.transform(X_val)
    Xte_b = best_scaler.transform(X_te)

    ytr_b = best_model.predict(Xtr_b)
    yva_b = best_model.predict(Xva_b)
    yte_b = best_model.predict(Xte_b)

    print("\n--- MÉTRICAS TUNED ---")
    tr_mse_b, tr_mae_b, r2_tr_b = report_metrics("Train", y_tr, ytr_b)
    va_mse_b, va_mae_b, r2_va_b = report_metrics(" Val ", y_val, yva_b)
    te_mse_b, te_mae_b, r2_te_b = report_metrics("Test ", y_te, yte_b)

    # 5) Resumen 
    print("\n=== RESUMEN DE MEJORA (Baseline → Tuned) ===")
    print(f"Val MSE: {va_mse:.6f} → {va_mse_b:.6f}  (Δ={va_mse - va_mse_b:+.6f})")
    print(f"Test MSE: {te_mse:.6f} → {te_mse_b:.6f} (Δ={te_mse - te_mse_b:+.6f})")
    print(f"Test R² : {r2_te:.4f} → {r2_te_b:.4f}   (Δ={r2_te_b - r2_te:+.4f})")
    print(f"Mejor config: penalty={best_pen} | alpha={best_alpha} | lr={best_lr}")

    # 6) Gráficas: curvas de pérdida (Train vs Val)
    if plot_curves:
        plt.figure(figsize=(8,4.5))
        plt.plot(tr_hist, label="Train MSE")
        plt.plot(val_hist, label="Val MSE")
        plt.axvline(best_epoch, linestyle="--", alpha=0.6, label=f"Best epoch = {best_epoch}")
        plt.xlabel("Época")
        plt.ylabel("MSE")
        plt.title("Curvas de entrenamiento (Baseline)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("sgd_curvas_baseline.png", dpi=150)
        try:
            plt.show()
        except Exception as e:
            print(f"[WARN] No se pudo mostrar la figura (guardada). Detalle: {e}")

if __name__ == "__main__":
    main(plot_curves=True)
