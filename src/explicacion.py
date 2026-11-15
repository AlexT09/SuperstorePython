"""
src/explicacion.py — Explicaciones de una predicción: valores de Shapley exactos y un sustituto local tipo LIME.

Con 7 variables hay 2^7 = 128 coaliciones, así que los valores de Shapley se calculan de forma exacta
por enumeración (no hace falta la aproximación de KernelSHAP). El valor de una coalición S es la
predicción media cuando las variables de S toman el valor de la fila explicada y el resto se toma de
una muestra de fondo (valor esperado intervencional).
"""
import itertools
from math import factorial

import numpy as np
import pandas as pd


def shapley_exacto(predecir, fila, fondo, variables):
    """Valores de Shapley de una fila.

    predecir: función tabla -> arreglo de predicciones. fila: DataFrame de una fila. fondo: DataFrame
    de referencia. Devuelve (phi, base) con sum(phi) + base = predicción de la fila.
    """
    d, n = len(variables), len(fondo)
    coaliciones = list(itertools.product([0, 1], repeat=d))
    bloques = []
    for mascara in coaliciones:
        b = fondo[variables].copy()
        for usar, v in zip(mascara, variables):
            if usar:
                b[v] = fila[v].iloc[0]
        bloques.append(b)
    valores = np.asarray(predecir(pd.concat(bloques, ignore_index=True))).reshape(len(coaliciones), n).mean(axis=1)
    valor = dict(zip(coaliciones, valores))
    phi = np.zeros(d)
    for i in range(d):
        for mascara in coaliciones:
            if mascara[i]:
                continue
            s = sum(mascara)
            peso = factorial(s) * factorial(d - s - 1) / factorial(d)
            con_i = mascara[:i] + (1,) + mascara[i + 1:]
            phi[i] += peso * (valor[con_i] - valor[mascara])
    return phi, float(valor[(0,) * d])


def sustituto_local(predecir, fila, fondo, variables, n=500, semilla=42):
    """Explicación tipo LIME: regresión lineal ponderada sobre perturbaciones de la fila.

    Cada perturbación conserva algunas variables de la fila (1) y toma las demás de una fila de fondo
    al azar (0). Las perturbaciones más parecidas a la fila pesan más (núcleo exponencial). El
    coeficiente de cada variable es su efecto local aproximado. A diferencia de Shapley, no garantiza
    que los efectos sumen la predicción y depende del ancho del núcleo y del muestreo.
    """
    rng = np.random.default_rng(semilla)
    d = len(variables)
    Z = rng.integers(0, 2, (n, d))
    Z[0] = 1
    datos = fondo[variables].sample(n, replace=True, random_state=semilla).reset_index(drop=True)
    for j, v in enumerate(variables):
        datos.loc[Z[:, j] == 1, v] = fila[v].iloc[0]
    y = np.asarray(predecir(datos))
    peso = np.exp(-((d - Z.sum(axis=1)) / d) ** 2 / 0.25)
    A = np.column_stack([np.ones(n), Z]) * np.sqrt(peso)[:, None]
    coef = np.linalg.lstsq(A.T @ A + 1e-6 * np.eye(d + 1), A.T @ (y * np.sqrt(peso)), rcond=None)[0]
    return coef[1:], float(coef[0])
