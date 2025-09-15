# Calibración PWM → Velocidad (Regresión Lineal)

Este repositorio muestra diferentes formas de construir un modelo de **regresión lineal** para calibrar el mapeo **PWM (%) → velocidad angular (rad/s)**:

1. **`RegresionPWM.py`** → Implementación *from scratch* (sin librerías de ML).  
   - Descenso de gradiente batch programado a mano.  
   - Incluye métricas (MSE, MAE, R²) y gráficas de la recta ajustada y la curva de pérdida.  

2. **`pwmLinearRegression.py`** → Implementación con `LinearRegression` de *scikit-learn*.  
   - Calcula la solución cerrada de mínimos cuadrados ordinarios (OLS).  
   - Genera métricas y la gráfica de la recta ajustada.  

3. **`pwmSGDRegressor.py`** → Implementación con `SGDRegressor` de *scikit-learn*.  
   - Usa descenso de gradiente estocástico (batch) con `partial_fit`.  
   - Reporta métricas y genera gráficas de ajuste y pérdida.  

4. **`SGDFinal.py`** → Versión extendida con `SGDRegressor`.  
   - Divide en **Train/Validation/Test**.  
   - Entrena con **early stopping** en validación.  
   - Diagnóstico automático de **bias**, **varianza** y **nivel de ajuste** (underfit/overfit/buen ajuste).  
   - Realiza **tuning de hiperparámetros** (penalty, alpha, learning rate).  
   - Muestra un **resumen de mejora** (Baseline → Tuned).  

---

## Requisitos

- Python 3.11+ (probado en 3.11.9)  
- Librerías:
  - `numpy`
  - `matplotlib`
  - `scikit-learn` (excepto para `RegresionPWM.py`)

Instalación de dependencias:

```bash
pip install numpy matplotlib scikit-learn
