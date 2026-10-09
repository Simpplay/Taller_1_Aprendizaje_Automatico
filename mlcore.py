"""mlcore.py - modelos lineales hechos a mano con NumPy (Taller 1, Aprendizaje Automatico 2026-2S).

Autor: Juan Jose Jimenez Maya.

Idea general:
    - theta = (b, w): el intercepto b va en la primera posicion y los coeficientes w despues.
    - X~ = [1 X]: a la matriz X le pego una columna de unos para que b se multiplique por 1.
    - P = diag(0, 1, ..., 1): sirve para que la penalizacion NO toque a b.

Funciones objetivo (las mismas del notebook de clase):
    lineal:    F = 1/(2N) ||y - X~ theta||^2 + lam2/2 ||w||^2 + lam1 ||w||_1
    logistica: F = (1/S) sum_n a_n [log(1 + e^{z_n}) - y_n z_n] + lam2/2 ||w||^2,   S = sum_n a_n

Clases:
    OLSRegression         minimos cuadrados con np.linalg.lstsq
    RidgeRegression       resuelve (X~^T X~ + N lam2 P) theta = X~^T y con np.linalg.solve
    LinearRegressionGD    descenso por gradiente (con L2 opcional)
    LassoRegression       gradiente proximal con umbral suave (hereda de LinearRegressionGD)
    LogisticRegressionGD  descenso por gradiente con L2 y pesos de clase opcionales

Partes adaptadas de Regresion_lineal_logistica_regularizacion.ipynb (notebook de clase):
LinearRegressionScratch, LogisticRegressionScratch, sigmoid, binary_cross_entropy y
fit_regularized. Los cambios estan explicados en la seccion de atribucion del informe.
"""
import json
import warnings

import numpy as np


# =========================================================== funciones numericas
def log1pexp(z):
    """Calcula log(1 + e^z) sin desbordarse: max(z, 0) + log1p(e^{-|z|})."""
    z = np.asarray(z, dtype=float)
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def sigmoid(z):
    """Sigmoide estable: solo se calcula e^{-|z|}, que nunca es mas grande que 1."""
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def logistic_loss_terms(y, z):
    """Perdida de cada observacion: log(1 + e^z) - y z = log(1 + e^{(1-2y) z}). No recorta z."""
    return log1pexp((1.0 - 2.0 * np.asarray(y, dtype=float)) * np.asarray(z, dtype=float))


def soft_threshold(u, t):
    """Umbral suave: sign(u) * max(|u| - t, 0). Lo que esta entre -t y t queda en cero exacto."""
    return np.sign(u) * np.maximum(np.abs(u) - t, 0.0)


# =========================================================== revisiones de entrada
def revisar_datos(X, y=None, n_columnas=None, binaria=False):
    """Revisa que X sea 2D y finita, y que y tenga el mismo numero de filas."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 1:
        raise ValueError(f"X debe ser 2D con al menos una fila; forma recibida {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X tiene valores no finitos (NaN o infinito).")
    if n_columnas is not None and X.shape[1] != n_columnas:
        raise ValueError(f"X tiene {X.shape[1]} columnas y se esperaban {n_columnas}.")
    if y is None:
        return X
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError(f"y debe ser 1D con {X.shape[0]} elementos; forma recibida {y.shape}.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y tiene valores no finitos.")
    if binaria and not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y solo puede tener etiquetas 0 y 1.")
    return X, y


def revisar_hiperparametro(nombre, valor, positivo=False, entero=False):
    """Revisa que un hiperparametro sea un numero finito y no negativo (o positivo si se pide)."""
    if entero:
        es_numero = isinstance(valor, (int, np.integer))
    else:
        es_numero = isinstance(valor, (int, float, np.number))
    if not es_numero or not np.isfinite(valor) or valor < 0 or (positivo and valor == 0):
        raise ValueError(f"{nombre} invalido: {valor!r}.")
    return int(valor) if entero else float(valor)


# =========================================================== gradiente proximal
def gradiente_proximal(theta0, grad, obj, prox, eta, max_iter, tol, acelerar=False):
    """Bucle comun de los tres modelos iterativos.

    En cada iteracion: theta_nuevo = prox(v - eta * grad(v)).
      - En regresion lineal y logistica prox no hace nada (descenso por gradiente normal).
      - En Lasso prox es el umbral suave.
    Criterio de parada: ||theta_nuevo - v|| / eta < tol (sin prox es la norma del gradiente).
    Si acelerar=True uso momento tipo FISTA: v no es el punto anterior sino uno "empujado"
    en la direccion del ultimo paso. Si el empuje va en contra del descenso, reinicio el momento.

    Devuelve theta, numero de iteraciones, motivo de parada e historial
    (iteracion, objetivo, norma de parada).
    """
    theta = theta0.copy()
    v = theta0.copy()
    t = 1.0
    historial = []
    motivo = "max_iter"
    k = 0
    norma = np.inf
    for k in range(1, max_iter + 1):
        theta_nuevo = prox(v - eta * grad(v), eta)
        norma = np.linalg.norm(theta_nuevo - v) / eta
        if not np.all(np.isfinite(theta_nuevo)):
            motivo = "divergencia(no finito)"
            break
        historial.append((k, obj(theta_nuevo), norma))
        if norma < tol:
            theta = theta_nuevo
            motivo = "tol(norma del mapeo de gradiente)"
            break
        if acelerar:
            if np.dot(v - theta_nuevo, theta_nuevo - theta) > 0:   # el momento estorba: reinicio
                t = 1.0
            t_nuevo = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            v = theta_nuevo + ((t - 1) / t_nuevo) * (theta_nuevo - theta)
            t = t_nuevo
        else:
            v = theta_nuevo
        theta = theta_nuevo
    if not historial or historial[-1][0] != k:
        historial.append((k, obj(theta), norma))
    return theta, k, motivo, np.array(historial)


# =========================================================== clase base
class _Linear:
    """Lo que comparten todos los modelos: X~, la mascara P, predecir, objetivo y guardar."""
    lam1 = 0.0
    lam2 = 0.0

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def _augment(self, X):
        """Devuelve X~ = [1 X] (o X si no hay intercepto)."""
        if self.fit_intercept:
            return np.column_stack([np.ones(len(X)), X])
        return X

    def _mask(self, d):
        """Diagonal de P: 0 para b y 1 para cada w_j."""
        if self.fit_intercept:
            return np.r_[0.0, np.ones(d)]
        return np.ones(d)

    def _theta(self):
        """Junta (b, w) en un solo vector."""
        if self.fit_intercept:
            return np.r_[self.intercept_, self.coef_]
        return self.coef_.copy()

    def _set_theta(self, theta):
        """Separa el vector theta en intercept_ y coef_."""
        theta = np.asarray(theta, dtype=float)
        if self.fit_intercept:
            self.intercept_ = float(theta[0])
            self.coef_ = theta[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = theta.copy()

    def decision_function(self, X):
        if not hasattr(self, "coef_"):
            raise RuntimeError("El modelo no se ha ajustado; primero llame a fit.")
        return revisar_datos(X, n_columnas=len(self.coef_)) @ self.coef_ + self.intercept_

    def predict(self, X):
        return self.decision_function(X)

    # --- objetivo de regresion lineal (LogisticRegressionGD lo cambia por el suyo)
    def smooth_objective(self, X, y, theta=None):
        """Parte derivable del objetivo: 1/(2N)||r||^2 + lam2/2 ||w||^2."""
        X, y = revisar_datos(X, y)
        theta = self._theta() if theta is None else np.asarray(theta, float)
        r = y - self._augment(X) @ theta
        return 0.5 * np.mean(r ** 2) + 0.5 * self.lam2 * np.sum((self._mask(X.shape[1]) * theta) ** 2)

    def smooth_gradient(self, X, y, theta=None):
        """Gradiente de la parte derivable: X~^T (X~ theta - y) / N + lam2 P theta."""
        X, y = revisar_datos(X, y)
        Xt = self._augment(X)
        theta = self._theta() if theta is None else np.asarray(theta, float)
        return Xt.T @ (Xt @ theta - y) / len(y) + self.lam2 * self._mask(X.shape[1]) * theta

    def objective(self, X, y, theta=None):
        """Objetivo completo = parte derivable + lam1 ||w||_1."""
        theta = self._theta() if theta is None else np.asarray(theta, float)
        w = theta[1:] if self.fit_intercept else theta
        return self.smooth_objective(X, y, theta) + self.lam1 * np.sum(np.abs(w))

    def normal_equation_residual(self, X, y):
        """||X~^T r - N lam2 P theta||: vale cero en la solucion exacta de OLS y Ridge."""
        return float(np.linalg.norm(self.smooth_gradient(X, y)) * len(y))

    def to_json(self, path):
        """Guarda hiperparametros y parametros en un archivo JSON."""
        hiper = {k: v for k, v in vars(self).items() if not k.endswith("_")}
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"class": type(self).__name__, "hyperparameters": hiper,
                       "coef": self.coef_.tolist(), "intercept": self.intercept_}, f, indent=2)


def load_model(path):
    """Lee un modelo guardado con to_json."""
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    clases = {"OLSRegression": OLSRegression, "RidgeRegression": RidgeRegression,
              "LinearRegressionGD": LinearRegressionGD, "LassoRegression": LassoRegression,
              "LogisticRegressionGD": LogisticRegressionGD}
    m = clases[s["class"]](**s["hyperparameters"])
    m.coef_ = np.asarray(s["coef"], dtype=float)
    m.intercept_ = float(s["intercept"])
    return m


# =========================================================== soluciones cerradas
class OLSRegression(_Linear):
    """Minimos cuadrados con lstsq (usa SVD). Si hay columnas repetidas da la solucion de
    norma minima y guarda el rango en rank_."""

    def fit(self, X, y):
        X, y = revisar_datos(X, y)
        Xt = self._augment(X)
        theta, _, rango, _ = np.linalg.lstsq(Xt, y, rcond=None)
        self._set_theta(theta)
        self.rank_ = int(rango)
        self.n_iter_, self.termination_, self.history_ = 0, "closed_form(lstsq)", []
        return self


class RidgeRegression(_Linear):
    """Resuelve el sistema (X~^T X~ + N lam2 P) theta = X~^T y con solve (sin calcular inversas)."""

    def __init__(self, lam2=1.0, fit_intercept=True):
        super().__init__(fit_intercept)
        self.lam2 = lam2

    def fit(self, X, y):
        X, y = revisar_datos(X, y)
        lam2 = revisar_hiperparametro("lam2", self.lam2, positivo=True)
        Xt = self._augment(X)
        A = Xt.T @ Xt + len(y) * lam2 * np.diag(self._mask(X.shape[1]))
        self._set_theta(np.linalg.solve(A, Xt.T @ y))
        self.n_iter_, self.termination_, self.history_ = 0, "closed_form(solve)", []
        return self


# =========================================================== modelos iterativos
class LinearRegressionGD(_Linear):
    """Regresion lineal por descenso por gradiente. Si lam1 > 0 el paso es proximal (Lasso).

    Paso eta = 1/L con L = sigma_max(X~)^2 / N + lam2 (constante de Lipschitz del gradiente).
    Si warm_start=False, cada llamada a fit empieza desde theta = 0.
    """

    def __init__(self, lam2=0.0, max_iter=50_000, tol=1e-8, fit_intercept=True,
                 warm_start=False, accelerate=False):
        super().__init__(fit_intercept)
        self.lam2 = lam2
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.accelerate = accelerate

    def _start(self, n_parametros):
        """Punto de partida: ceros, o la solucion anterior si warm_start=True."""
        if self.warm_start and hasattr(self, "coef_") and len(self._theta()) == n_parametros:
            return self._theta()
        return np.zeros(n_parametros)

    def _finish(self, theta, n_iter, motivo, historial):
        """Guarda el resultado del bucle y avisa si no convergio."""
        self._set_theta(theta)
        self.n_iter_ = n_iter
        self.termination_ = motivo
        self.history_ = historial
        self.converged_ = motivo.startswith("tol")
        if not self.converged_:
            warnings.warn(f"{type(self).__name__}: termino por '{motivo}' tras {n_iter} iteraciones.",
                          RuntimeWarning, stacklevel=3)
        return self

    def fit(self, X, y):
        X, y = revisar_datos(X, y)
        lam1 = revisar_hiperparametro("lam1", self.lam1)
        lam2 = revisar_hiperparametro("lam2", self.lam2)
        max_iter = revisar_hiperparametro("max_iter", self.max_iter, positivo=True, entero=True)
        tol = revisar_hiperparametro("tol", self.tol, positivo=True)
        Xt = self._augment(X)
        n = len(y)
        L = np.linalg.norm(Xt, 2) ** 2 / n + lam2
        eta = 1.0 / L
        self.eta_, self.lipschitz_ = eta, L
        P = self._mask(X.shape[1])
        penalizado = P.astype(bool)          # True en las posiciones de w, False en b

        def grad(th):
            return Xt.T @ (Xt @ th - y) / n + lam2 * P * th

        def obj(th):
            return (0.5 * np.mean((y - Xt @ th) ** 2) + 0.5 * lam2 * np.sum((P * th) ** 2)
                    + lam1 * np.sum(np.abs(th[penalizado])))

        def prox(u, e):
            # umbral suave en cada iteracion, solo sobre w (b no se toca)
            if lam1 == 0:
                return u
            out = u.copy()
            out[penalizado] = soft_threshold(u[penalizado], e * lam1)
            return out

        resultado = gradiente_proximal(self._start(Xt.shape[1]), grad, obj, prox, eta,
                                       max_iter, tol, self.accelerate)
        return self._finish(*resultado)


class LassoRegression(LinearRegressionGD):
    """Lasso por gradiente proximal. Los ceros salen del umbral suave, no de redondear al final."""

    def __init__(self, lam1=1.0, lam2=0.0, max_iter=100_000, tol=1e-8, fit_intercept=True,
                 warm_start=False, accelerate=False):
        super().__init__(lam2, max_iter, tol, fit_intercept, warm_start, accelerate)
        self.lam1 = lam1


def lasso_kkt_violation(model, X, y):
    """Maxima violacion de las condiciones de optimalidad (KKT) de Lasso:
    gradiente de b = 0;  g_j + lam1*sign(w_j) = 0 si w_j != 0;  |g_j| <= lam1 si w_j = 0."""
    g = model.smooth_gradient(X, y)
    violaciones = [abs(g[0])] if model.fit_intercept else []
    if model.fit_intercept:
        g = g[1:]
    w = model.coef_
    no_cero = w != 0
    violaciones += list(np.abs(g[no_cero] + model.lam1 * np.sign(w[no_cero])))
    violaciones += list(np.maximum(np.abs(g[~no_cero]) - model.lam1, 0.0))
    return float(max(violaciones)) if violaciones else 0.0


class LogisticRegressionGD(LinearRegressionGD):
    """Regresion logistica binaria con L2 y pesos de clase opcionales ('balanced': a_c = N/(2 N_c)).

    Paso eta = 1/L con L = sigma_max(diag(sqrt(a)) X~)^2 / (4 S) + lam2.
    """

    def __init__(self, lam2=0.0, class_weight=None, max_iter=200_000, tol=1e-7,
                 fit_intercept=True, warm_start=False, accelerate=False):
        super().__init__(lam2, max_iter, tol, fit_intercept, warm_start, accelerate)
        self.class_weight = class_weight

    def _pesos(self, y):
        """Peso de cada observacion: todos 1, o N/(2 N_c) si class_weight='balanced'."""
        if self.class_weight is None:
            return np.ones_like(y)
        if self.class_weight != "balanced":
            raise ValueError("class_weight debe ser None o 'balanced'.")
        n, n1 = len(y), y.sum()
        if n1 == 0 or n1 == n:
            raise ValueError("class_weight='balanced' necesita las dos clases.")
        self.class_weights_ = {0: n / (2 * (n - n1)), 1: n / (2 * n1)}
        return np.where(y == 1, self.class_weights_[1], self.class_weights_[0])

    def _objetivo_y_gradiente(self, X, y):
        """Construye las funciones objetivo y gradiente para unos datos fijos."""
        Xt = self._augment(X)
        a = self._pesos(y)
        S = a.sum()
        P = self._mask(X.shape[1])
        signo = 1.0 - 2.0 * y

        def obj(th):
            return np.sum(a * log1pexp(signo * (Xt @ th))) / S + 0.5 * self.lam2 * np.sum((P * th) ** 2)

        def grad(th):
            return Xt.T @ (a * (sigmoid(Xt @ th) - y)) / S + self.lam2 * P * th

        return Xt, a, S, obj, grad

    def smooth_objective(self, X, y, theta=None):
        X, y = revisar_datos(X, y, binaria=True)
        _, _, _, obj, _ = self._objetivo_y_gradiente(X, y)
        return obj(self._theta() if theta is None else np.asarray(theta, float))

    objective = smooth_objective      # en la logistica no hay parte L1

    def smooth_gradient(self, X, y, theta=None):
        X, y = revisar_datos(X, y, binaria=True)
        _, _, _, _, grad = self._objetivo_y_gradiente(X, y)
        return grad(self._theta() if theta is None else np.asarray(theta, float))

    def fit(self, X, y):
        X, y = revisar_datos(X, y, binaria=True)
        revisar_hiperparametro("lam2", self.lam2)
        max_iter = revisar_hiperparametro("max_iter", self.max_iter, positivo=True, entero=True)
        tol = revisar_hiperparametro("tol", self.tol, positivo=True)
        Xt, a, S, obj, grad = self._objetivo_y_gradiente(X, y)
        L = np.linalg.norm(Xt * np.sqrt(a)[:, None], 2) ** 2 / (4 * S) + self.lam2
        eta = 1.0 / L
        self.eta_, self.lipschitz_ = eta, L
        self.classes_ = np.array([0, 1])
        sin_prox = lambda u, e: u
        resultado = gradiente_proximal(self._start(Xt.shape[1]), grad, obj, sin_prox, eta,
                                       max_iter, tol, self.accelerate)
        return self._finish(*resultado)

    def predict_proba(self, X):
        z = self.decision_function(X)
        return np.column_stack([sigmoid(-z), sigmoid(z)])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def log_loss(self, X, y):
        X, y = revisar_datos(X, y, binaria=True)
        return float(np.mean(logistic_loss_terms(y, self.decision_function(X))))


# =========================================================== verificacion numerica
def numerical_gradient(f, theta, h=1e-6):
    """Gradiente por diferencias centrales: (f(theta + h e_j) - f(theta - h e_j)) / (2h)."""
    theta = np.asarray(theta, dtype=float)
    g = np.zeros_like(theta)
    for j in range(len(theta)):
        e = np.zeros_like(theta)
        e[j] = h
        g[j] = (f(theta + e) - f(theta - e)) / (2 * h)
    return g


def relative_gradient_error(g_a, g_n):
    """delta_g = ||g_a - g_n|| / max(1, ||g_a||, ||g_n||)."""
    return float(np.linalg.norm(g_a - g_n) / max(1.0, np.linalg.norm(g_a), np.linalg.norm(g_n)))
