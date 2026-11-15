"""
src/original_model.py — Modelo original: Perfil Estacional con Pesos Jerárquicos (PEJ).

Problema. Se pronostica un índice sin escala z(s, t) = ventas de la serie s en el mes t divididas por
el promedio de sus 12 meses anteriores. Con series cortas hay dos pronósticos sencillos disponibles:

    P[m(t)]   el perfil común: valor medio del índice en ese mes del año entre TODAS las series
              (estable, pero ignora lo particular de cada serie);
    r(s, t)   el rezago propio: el índice de la misma serie en el mismo mes del año anterior
              (particular, pero ruidoso: es una sola observación).

Regla de combinación (lo que aporta el modelo). Cada serie s recibe su propio peso w_s:

    z_hat(s, t) = P[m(t)] + w_s * ( r(s, t) - P[m(t)] )                                    (1)

Los pesos no se eligen en una rejilla: se aprenden minimizando

    J(w_1..w_S, w0) = sum_s sum_t ( a_st - w_s * b_st )^2  +  lam * sum_s ( w_s - w0 )^2     (2)

con a_st = z(s, t) - P[m(t)], b_st = r(s, t) - P[m(t)] y 0 <= w_s <= 1. El segundo término acerca
cada peso a un peso común w0, que también se aprende: una serie con poca evidencia hereda el peso
del grupo y una serie con mucha evidencia conserva el suyo (agrupamiento parcial).

Reglas de actualización (descenso por bloques; cada una es el mínimo exacto de J en su bloque):

    w_s <- recortar_[0,1]( (S_ab[s] + lam * w0) / (S_bb[s] + lam) )                         (3)
    w0  <- promedio_s( w_s )                                                                (4)

donde S_ab[s] = sum_t a_st * b_st y S_bb[s] = sum_t b_st^2. Se repite hasta que J deja de bajar.

Propiedades (demostradas en el capítulo 5 del libro y verificadas en tests/):
  * J no aumenta en ninguna iteración y el algoritmo converge al único mínimo de (2).
  * lam -> infinito: todas las series comparten un peso (combinación lineal global).
  * lam -> 0: cada serie usa su propio peso de mínimos cuadrados (sin agrupar).
  * Costo: O(n) para las sumas + O(S) por iteración; predicción O(n). Memoria O(S + 12).

Solo usa numpy. Hereda de BaseEstimator y RegressorMixin, de modo que funciona con Pipeline,
clone, GridSearchCV y cross_val_score.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

COL_MES = ["mes_sin", "mes_cos"]          # mes del año codificado en un círculo
COL_REZAGO = "r_lag_12"                   # índice de la misma serie, mismo mes del año anterior
COL_SERIE = ["Category", "Region"]        # columnas que identifican la serie
COLUMNAS = COL_SERIE + COL_MES + [COL_REZAGO]
PESOS = ("jerarquico", "global", "independiente", "promedio", "cero")
PERFILES = ("comun", "serie", "ninguno")


def mes_del_anio(mes_sin, mes_cos):
    """Recupera el mes (0 = diciembre, 1 = enero, ..., 11 = noviembre) de su codificación seno/coseno.

    El mes m se codificó como el ángulo 2*pi*m/12; arctan2 devuelve ese ángulo y dividirlo por
    30 grados (pi/6) devuelve m.
    """
    angulo = np.arctan2(np.asarray(mes_sin, float), np.asarray(mes_cos, float))
    return np.rint(angulo * 6 / np.pi).astype(int) % 12


class PerfilEstacionalJerarquico(BaseEstimator, RegressorMixin):
    """Perfil estacional común combinado con el rezago propio mediante pesos jerárquicos por serie.

    Parámetros
    ----------
    lam : float >= 0
        Fuerza con la que el peso de cada serie se acerca al peso común (ecuación 2).
    max_iter : int
        Máximo de iteraciones del descenso por bloques.
    tol : float
        Criterio de parada: se detiene cuando J baja menos de tol * (1 + J).
    pesos : {"jerarquico", "global", "independiente", "promedio", "cero"}
        Solo para la ablación. "jerarquico" es el modelo; "global" usa un único peso (lam infinito),
        "independiente" un peso por serie sin agrupar (lam = 0), "promedio" fija w = 0.5 (promedio
        simple de los dos pronósticos) y "cero" usa solo el perfil.
    perfil : {"comun", "serie", "ninguno"}
        Solo para la ablación. "serie" estima un perfil por serie y "ninguno" usa una constante.

    Atributos (después de fit)
    --------------------------
    perfil_ : arreglo de 12 valores, el perfil común P (posición 0 = diciembre).
    pesos_ : dict serie -> w_s.
    w_global_ : float, el peso común w0 (se usa para series no vistas al entrenar).
    historial_ : lista con el valor de J en cada iteración (curva de convergencia).
    n_iter_ : int, iteraciones ejecutadas.
    """

    def __init__(self, lam=8.0, max_iter=500, tol=1e-12, pesos="jerarquico", perfil="comun"):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.pesos = pesos
        self.perfil = perfil

    # ------------------------------------------------------------------ entrada
    def _leer(self, X, y=None):
        """Valida la tabla de entrada y devuelve (serie, mes, rezago[, y]) como arreglos."""
        faltan = [c for c in COLUMNAS if c not in getattr(X, "columns", [])]
        if faltan:
            raise ValueError(f"X debe ser una tabla con las columnas {COLUMNAS}; faltan {faltan}")
        num = np.asarray(X[COL_MES + [COL_REZAGO]], float)
        if not np.isfinite(num).all():
            raise ValueError("X contiene valores faltantes o infinitos")
        serie = (X[COL_SERIE[0]].astype(str) + "|" + X[COL_SERIE[1]].astype(str)).to_numpy()
        mes = mes_del_anio(num[:, 0], num[:, 1])
        if y is None:
            return serie, mes, num[:, 2]
        y = np.asarray(y, float)
        if y.shape[0] != num.shape[0]:
            raise ValueError("X e y tienen distinto número de filas")
        if not np.isfinite(y).all():
            raise ValueError("y contiene valores faltantes o infinitos")
        return serie, mes, num[:, 2], y

    # ------------------------------------------------------------------ perfil
    @staticmethod
    def _medias_por_mes(y, mes, base):
        """Media de y en cada mes del año; un mes sin datos conserva el valor de `base`."""
        p = np.array(base, float)
        for m in np.unique(mes):
            p[m] = y[mes == m].mean()
        return p

    def _perfil_de(self, serie, mes):
        """Valor del perfil que corresponde a cada fila."""
        if self.perfil == "serie":
            return np.array([self.perfiles_serie_.get(s, self.perfil_)[m] for s, m in zip(serie, mes)])
        return self.perfil_[mes]

    # ------------------------------------------------------------------ objetivo
    @staticmethod
    def objetivo(w, w0, Saa, Sab, Sbb, lam):
        """Valor de J (ecuación 2) a partir de las sumas por serie."""
        return float(np.sum(Saa - 2 * w * Sab + w ** 2 * Sbb) + lam * np.sum((w - w0) ** 2))

    def fit(self, X, y):
        if self.lam < 0:
            raise ValueError("lam debe ser mayor o igual que cero")
        if self.pesos not in PESOS or self.perfil not in PERFILES:
            raise ValueError(f"pesos debe estar en {PESOS} y perfil en {PERFILES}")
        serie, mes, r, y = self._leer(X, y)

        # 1) Perfil estacional: media del índice por mes del año entre todas las series
        constante = np.full(12, y.mean())
        self.perfil_ = constante if self.perfil == "ninguno" else self._medias_por_mes(y, mes, constante)
        if self.perfil == "serie":
            self.perfiles_serie_ = {s: self._medias_por_mes(y[serie == s], mes[serie == s], self.perfil_)
                                    for s in np.unique(serie)}
        p = self._perfil_de(serie, mes)

        # 2) Desviaciones respecto al perfil y sus sumas por serie (lo único que necesita el algoritmo)
        a, b = y - p, r - p
        self.series_, idx = np.unique(serie, return_inverse=True)
        S = self.series_.size
        Saa = np.bincount(idx, a * a, S)
        Sab = np.bincount(idx, a * b, S)
        Sbb = np.bincount(idx, b * b, S)

        # 3) Pesos
        tot = Sbb.sum()
        w0 = float(np.clip(Sab.sum() / tot, 0, 1)) if tot > 0 else 0.0     # peso global de mínimos cuadrados
        w = np.full(S, w0)
        self.historial_ = [self.objetivo(w, w0, Saa, Sab, Sbb, self.lam)]
        self.n_iter_ = 0
        if self.pesos == "jerarquico":
            for _ in range(self.max_iter):
                den = Sbb + self.lam
                # regla (3): mínimo exacto de J en cada w_s con w0 fijo, recortado a [0, 1]
                w = np.clip(np.divide(Sab + self.lam * w0, den, out=np.full(S, w0), where=den > 0), 0, 1)
                # regla (4): mínimo exacto de J en w0 con los w_s fijos
                w0 = float(w.mean())
                self.historial_.append(self.objetivo(w, w0, Saa, Sab, Sbb, self.lam))
                self.n_iter_ += 1
                if self.historial_[-2] - self.historial_[-1] <= self.tol * (1 + self.historial_[-1]):
                    break
        elif self.pesos == "independiente":
            w = np.clip(np.divide(Sab, Sbb, out=np.full(S, w0), where=Sbb > 0), 0, 1)
        elif self.pesos == "promedio":
            w, w0 = np.full(S, 0.5), 0.5
        elif self.pesos == "cero":
            w, w0 = np.zeros(S), 0.0
        self.pesos_ = dict(zip(self.series_, w))
        self.w_global_ = w0
        return self

    # ------------------------------------------------------------------ salida
    def _partes(self, X):
        check_is_fitted(self, "pesos_")
        serie, mes, r = self._leer(X)
        p = self._perfil_de(serie, mes)
        w = np.array([self.pesos_.get(s, self.w_global_) for s in serie])
        return p, w * (r - p)

    def predict(self, X):
        """Índice pronosticado (ecuación 1)."""
        p, ajuste = self._partes(X)
        return p + ajuste

    def explicar(self, X):
        """Descompone cada predicción en tres sumandos exactos: base + estacional + rezago.

        base       = media del perfil (lo que se pronostica sin saber nada)
        estacional = P[mes] - base (lo que aporta el mes del año)
        rezago     = w_s * (r - P[mes]) (lo que aporta la historia propia de la serie)
        """
        p, ajuste = self._partes(X)
        base = float(self.perfil_.mean())
        return {"base": np.full(p.size, base), "estacional": p - base, "rezago": ajuste}
