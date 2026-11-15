"""
paper/perfil_estacional.py — Modelo propuesto: perfil estacional global contraído (PEG).

Predice el índice de ventas (ventas del mes ÷ promedio de los 12 meses previos) como

    z_hat(s, t) = (1 - w) * P[mes(t)] + w * r12(s, t)

donde P[m] es el perfil estacional común: el valor típico (media o mediana) del índice en el mes m
entre todas las series de entrenamiento, y r12 son las ventas relativas del mismo mes del año
anterior de la propia serie. w en [0, 1] reparte el peso entre el patrón común y el de la serie
(w = 1 reproduce el pronóstico ingenuo estacional).

Tiene 12 parámetros (el perfil) y dos hiperparámetros (w y el estadístico), elegidos por validación
cruzada temporal igual que en los demás modelos. El argumento `perfil` solo existe para la ablación:
"serie" estima un perfil por serie (sin agrupar) y "ninguno" usa una constante (sin estacionalidad).
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

COLUMNAS = ["mes_sin", "mes_cos", "r_lag_12"]
SERIE = ["Category", "Region"]
REJILLA = {"w": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "estadistico": ["media", "mediana"]}


class PerfilEstacional(BaseEstimator, RegressorMixin):
    def __init__(self, w=0.2, estadistico="media", perfil="comun"):
        self.w = w
        self.estadistico = estadistico
        self.perfil = perfil

    @staticmethod
    def _mes(X):
        """Mes del año (0 = diciembre, 1 = enero, ...) a partir de su codificación seno/coseno."""
        return np.rint(np.arctan2(X[:, 0], X[:, 1]) * 6 / np.pi).astype(int) % 12

    def _estimar(self, y, mes, base):
        f = np.mean if self.estadistico == "media" else np.median
        p = np.array(base, float)
        for m in np.unique(mes):
            p[m] = f(y[mes == m])
        return p

    def fit(self, X, y):
        A, y = np.asarray(X[COLUMNAS], float), np.asarray(y, float)
        f = np.mean if self.estadistico == "media" else np.median
        mes = self._mes(A)
        constante = np.full(12, f(y))           # un mes sin datos usa el valor típico global
        self.perfil_ = constante if self.perfil == "ninguno" else self._estimar(y, mes, constante)
        if self.perfil == "serie":              # un perfil por serie; lo que falte se toma del común
            clave = (X[SERIE[0]].astype(str) + "|" + X[SERIE[1]].astype(str)).to_numpy()
            self.por_serie_ = {k: self._estimar(y[clave == k], mes[clave == k], self.perfil_) for k in np.unique(clave)}
        return self

    def predict(self, X):
        A = np.asarray(X[COLUMNAS], float)
        mes = self._mes(A)
        if self.perfil == "serie":
            clave = (X[SERIE[0]].astype(str) + "|" + X[SERIE[1]].astype(str)).to_numpy()
            p = np.array([self.por_serie_.get(k, self.perfil_)[m] for k, m in zip(clave, mes)])
        else:
            p = self.perfil_[mes]
        return (1 - self.w) * p + self.w * A[:, 2]
