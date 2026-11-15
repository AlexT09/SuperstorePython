"""Pruebas unitarias del modelo original (src/original_model.py). Ejecutar: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.original_model import PerfilEstacionalJerarquico, mes_del_anio  # noqa: E402


def sintetico(n_series=8, n_anios=12, ruido=0.02, semilla=0):
    """Datos donde la solución es conocida: z = P[mes] + w_s * (r - P[mes]) + ruido."""
    rng = np.random.default_rng(semilla)
    perfil = 1 + 0.4 * np.sin(2 * np.pi * np.arange(12) / 12)          # posición 0 = diciembre
    w_real = np.linspace(0.1, 0.8, n_series)
    filas = []
    for s in range(n_series):
        for t in range(12 * n_anios):
            mes = t % 12 + 1                                            # 1..12
            r = perfil[mes % 12] + rng.normal(0, 0.3)
            z = perfil[mes % 12] + w_real[s] * (r - perfil[mes % 12]) + rng.normal(0, ruido)
            filas.append((f"s{s}", "x", t, np.sin(2 * np.pi * mes / 12), np.cos(2 * np.pi * mes / 12), r, z))
    d = pd.DataFrame(filas, columns=["Category", "Region", "t", "mes_sin", "mes_cos", "r_lag_12", "z"])
    d = d.sort_values(["t", "Category"]).reset_index(drop=True)
    return d, perfil, w_real


@pytest.fixture(scope="module")
def datos():
    return sintetico()


def test_recupera_la_solucion_conocida(datos):
    d, perfil, w_real = datos
    m = PerfilEstacionalJerarquico(lam=0.01).fit(d, d.z)
    w = np.array([m.pesos_[f"s{s}|x"] for s in range(w_real.size)])
    assert np.abs(w - w_real).max() < 0.03          # recupera el peso de cada serie
    assert np.abs(m.perfil_ - perfil).max() < 0.06  # y el perfil estacional (error de muestreo ~0.015)
    assert np.abs(m.predict(d) - d.z).mean() < 0.03


def test_la_perdida_no_aumenta(datos):
    d, _, _ = datos
    for lam in [0.0, 0.5, 8.0, 100.0]:
        h = np.array(PerfilEstacionalJerarquico(lam=lam).fit(d, d.z).historial_)
        assert np.all(np.diff(h) <= 1e-9 * (1 + h[:-1]))   # descenso monótono (proposición 1)


def test_converge_al_minimo_del_objetivo(datos):
    """El punto final coincide con la solución exacta del sistema lineal de (2) cuando no hay recorte."""
    d, _, _ = datos
    lam = 5.0
    m = PerfilEstacionalJerarquico(lam=lam).fit(d, d.z)
    p = m.perfil_[mes_del_anio(d.mes_sin, d.mes_cos)]
    a, b = d.z.to_numpy() - p, d.r_lag_12.to_numpy() - p
    idx = d.Category.str[1:].astype(int).to_numpy()
    S = idx.max() + 1
    Sab, Sbb = np.bincount(idx, a * b, S), np.bincount(idx, b * b, S)
    # condiciones de primer orden: (Sbb_s + lam) w_s - lam w0 = Sab_s ;  w0 = promedio(w)
    A = np.diag(Sbb + lam) - lam / S
    w_exacto = np.linalg.solve(A, Sab)
    w = np.array([m.pesos_[f"s{s}|x"] for s in range(S)])
    np.testing.assert_allclose(w, w_exacto, atol=1e-6)
    assert abs(m.w_global_ - w_exacto.mean()) < 1e-6


def test_casos_limite(datos):
    d, _, _ = datos
    grande = PerfilEstacionalJerarquico(lam=1e9).fit(d, d.z)
    unico = PerfilEstacionalJerarquico(pesos="global").fit(d, d.z)
    assert np.ptp(list(grande.pesos_.values())) < 1e-6            # lam grande: un solo peso
    assert abs(grande.w_global_ - unico.w_global_) < 1e-3
    cero = PerfilEstacionalJerarquico(lam=0.0).fit(d, d.z)
    indep = PerfilEstacionalJerarquico(pesos="independiente").fit(d, d.z)
    np.testing.assert_allclose(list(cero.pesos_.values()), list(indep.pesos_.values()), atol=1e-8)


def test_pesos_dentro_de_cero_y_uno(datos):
    d, _, _ = datos
    m = PerfilEstacionalJerarquico(lam=1.0).fit(d, -d.z)   # objetivo invertido: fuerza el recorte
    w = np.array(list(m.pesos_.values()))
    assert w.min() >= 0 and w.max() <= 1


def test_reproducible(datos):
    d, _, _ = datos
    p1 = PerfilEstacionalJerarquico().fit(d, d.z).predict(d)
    p2 = PerfilEstacionalJerarquico().fit(d, d.z).predict(d)
    np.testing.assert_array_equal(p1, p2)                  # el algoritmo es determinista


def test_clone_conserva_hiperparametros():
    m = PerfilEstacionalJerarquico(lam=3.0, max_iter=10)
    assert clone(m).get_params() == m.get_params()


def test_entrada_invalida(datos):
    d, _, _ = datos
    malo = d.copy()
    malo.loc[0, "r_lag_12"] = np.nan
    with pytest.raises(ValueError):
        PerfilEstacionalJerarquico().fit(malo, d.z)
    with pytest.raises(ValueError):
        PerfilEstacionalJerarquico().fit(d.drop(columns="mes_sin"), d.z)
    with pytest.raises(ValueError):
        PerfilEstacionalJerarquico(lam=-1).fit(d, d.z)
    with pytest.raises(ValueError):
        PerfilEstacionalJerarquico().fit(d, d.z.iloc[:-1])


def test_serie_no_vista_usa_el_peso_comun(datos):
    d, _, _ = datos
    m = PerfilEstacionalJerarquico().fit(d, d.z)
    nueva = d.head(3).assign(Category="nueva")
    p = m.perfil_[mes_del_anio(nueva.mes_sin, nueva.mes_cos)]
    np.testing.assert_allclose(m.predict(nueva), p + m.w_global_ * (nueva.r_lag_12 - p))


def test_explicacion_suma_la_prediccion(datos):
    d, _, _ = datos
    m = PerfilEstacionalJerarquico().fit(d, d.z)
    e = m.explicar(d)
    np.testing.assert_allclose(e["base"] + e["estacional"] + e["rezago"], m.predict(d))


def test_compatible_con_scikit_learn(datos):
    d, _, _ = datos
    cv = TimeSeriesSplit(n_splits=3, test_size=8 * 12)
    puntajes = cross_val_score(PerfilEstacionalJerarquico(), d, d.z, cv=cv, scoring="neg_mean_absolute_error")
    assert np.isfinite(puntajes).all()
    g = GridSearchCV(PerfilEstacionalJerarquico(), {"lam": [0.1, 10]}, cv=cv, scoring="neg_mean_absolute_error")
    assert g.fit(d, d.z).best_params_["lam"] in (0.1, 10)
