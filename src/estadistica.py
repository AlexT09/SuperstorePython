"""
src/estadistica.py — Pruebas de hipótesis y medidas de efecto para comparar pronósticos.

Todas reciben pérdidas ordenadas en el tiempo (una por mes de evaluación). Convención de signo en las
comparaciones por pares: d_t = pérdida del modelo A − pérdida del modelo B; negativo = A es mejor.
"""
import numpy as np
from scipy import stats


def indices_bootstrap_estacionario(T, n_rep, bloque, rng):
    """Índices del bootstrap estacionario (Politis y Romano, 1994).

    Cada remuestra se arma con bloques de longitud aleatoria (geométrica, media `bloque`) que se
    enrollan al final de la serie; así se conserva la dependencia entre meses cercanos.
    """
    idx = np.empty((n_rep, T), int)
    idx[:, 0] = rng.integers(0, T, n_rep)
    salto = rng.random((n_rep, T)) < 1 / bloque          # True = empieza un bloque nuevo
    nuevo = rng.integers(0, T, (n_rep, T))
    for t in range(1, T):
        idx[:, t] = np.where(salto[:, t], nuevo[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def ic_bootstrap(d, rng, n_rep=10000, bloque=3, nivel=0.95):
    """Intervalo de confianza de la media de d_t con bootstrap estacionario (percentil)."""
    d = np.asarray(d, float)
    medias = d[indices_bootstrap_estacionario(d.size, n_rep, bloque, rng)].mean(axis=1)
    return np.percentile(medias, [50 * (1 - nivel), 100 - 50 * (1 - nivel)])


def dm_hln(d, h=1):
    """Diebold–Mariano con la corrección de Harvey–Leybourne–Newbold. Devuelve (estadístico, p bilateral).

    H0: los dos modelos tienen la misma pérdida esperada. Supone que d_t es estacionaria.
    """
    d = np.asarray(d, float)
    T, media = d.size, d.mean()
    gamma = [np.sum((d[k:] - media) * (d[:T - k] - media)) / T for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / T
    est = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T) * media / np.sqrt(var)
    return float(est), float(2 * stats.t.sf(abs(est), df=T - 1))


def giacomini_white(d):
    """Giacomini–White (capacidad predictiva condicional, h = 1) con instrumentos (1, d_{t-1}).

    H0: conocida la diferencia del mes anterior, ningún modelo es mejor en el mes siguiente.
    Válida con modelos reestimados en cada origen. Devuelve (estadístico, p).
    """
    d = np.asarray(d, float)
    H = np.column_stack([np.ones(d.size - 1), d[:-1]])
    Z = H * d[1:, None]
    n, zm = Z.shape[0], Z.mean(axis=0)
    est = float(n * zm @ np.linalg.pinv(Z.T @ Z / n) @ zm)
    return est, float(stats.chi2.sf(est, df=H.shape[1]))


def mcs(perdidas, nombres, rng, n_rep=5000, bloque=3):
    """Model Confidence Set (Hansen, Lunde y Nason, 2011) con el estadístico de rango T_R.

    perdidas: matriz T x M (meses x modelos). Elimina de a un modelo, el peor, mientras se rechace
    la igualdad de capacidad predictiva. Devuelve dict nombre -> p-valor MCS; un modelo pertenece al
    conjunto de confianza de nivel 1 − alfa si su p-valor es >= alfa.
    """
    L = np.asarray(perdidas, float)
    T = L.shape[0]
    idx = indices_bootstrap_estacionario(T, n_rep, bloque, rng)
    medias_b = L[idx].mean(axis=1)                         # n_rep x M
    vivos, p_mcs, p_acum = list(range(L.shape[1])), {}, 0.0
    while len(vivos) > 1:
        m = L[:, vivos].mean(axis=0)
        mb = medias_b[:, vivos]
        dif = m[:, None] - m[None, :]                      # d_ij
        dif_b = mb[:, :, None] - mb[:, None, :]            # d*_ij
        se = np.sqrt(((dif_b - dif) ** 2).mean(axis=0))
        np.fill_diagonal(se, 1.0)
        t = dif / se
        t_b = np.abs(dif_b - dif) / se
        p = float((t_b.max(axis=(1, 2)) >= np.abs(t).max()).mean())
        p_acum = max(p_acum, p)
        peor = int(np.argmax(t.max(axis=1)))               # el de mayor pérdida relativa
        p_mcs[nombres[vivos[peor]]] = p_acum
        vivos.pop(peor)
    p_mcs[nombres[vivos[0]]] = 1.0
    return p_mcs


def holm(p):
    """p-valores ajustados por Holm–Bonferroni (mismo orden de entrada)."""
    p = np.asarray(p, float)
    orden = np.argsort(p)
    ajust = np.maximum.accumulate((p.size - np.arange(p.size)) * p[orden])
    out = np.empty_like(p)
    out[orden] = np.minimum(ajust, 1.0)
    return out


def d_cohen(d):
    """d de Cohen de diferencias pareadas: media / desviación estándar."""
    d = np.asarray(d, float)
    return float(d.mean() / d.std(ddof=1))


def delta_cliff(a, b):
    """Delta de Cliff: P(a > b) − P(a < b) sobre todos los pares. Negativo = a tiende a ser menor."""
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None, :]
    return float((a > b).mean() - (a < b).mean())


def friedman_iman_davenport(tabla):
    """Friedman con la corrección de Iman–Davenport. tabla: bloques (filas) x modelos (columnas), menor = mejor.

    Devuelve dict con los rangos medios, el estadístico F y su p-valor.
    """
    X = np.asarray(tabla, float)
    N, k = X.shape
    rangos = np.apply_along_axis(stats.rankdata, 1, X)
    R = rangos.mean(axis=0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    F = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) - chi2 > 0 else np.inf
    return {"rangos": R, "chi2": float(chi2), "F": float(F),
            "p": float(stats.f.sf(F, k - 1, (k - 1) * (N - 1)))}


def comparar(perd_a, perd_b, rng):
    """Batería de pruebas por pares sobre pérdidas mensuales (A = modelo original, B = referencia)."""
    d = np.asarray(perd_a, float) - np.asarray(perd_b, float)
    dm, p_dm = dm_hln(d)
    gw, p_gw = giacomini_white(d)
    lo, hi = ic_bootstrap(d, rng)
    return {"dif": float(d.mean()), "ic_lo": float(lo), "ic_hi": float(hi), "DM": dm, "p_DM": p_dm,
            "GW": gw, "p_GW": p_gw, "p_wilcoxon": float(stats.wilcoxon(d).pvalue),
            "d_cohen": d_cohen(d), "delta_cliff": delta_cliff(perd_a, perd_b),
            "meses_mejor": int((d < 0).sum()), "shapiro_p": float(stats.shapiro(d).pvalue)}
