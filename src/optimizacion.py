"""
src/optimizacion.py — Los cuatro métodos de optimización de hiperparámetros del proyecto.

Todos comparten la misma función objetivo: el MAE medio de validación cruzada temporal de un
estimador (normalmente un Pipeline con su preprocesamiento, que así se ajusta solo con los datos de
entrenamiento de cada pliegue: no hay fuga de información).

Un espacio de búsqueda es un dict  nombre -> especificación:
    ("log", lo, hi)    real en escala logarítmica        ("int", lo, hi)   entero
    ("real", lo, hi)   real en escala lineal             ("cat", [...])    categórica

Cada método devuelve un dict con: params (los mejores), puntaje (su MAE de validación), historial
(MAE de cada evaluación, en orden), segundos, y en el genético la diversidad por generación.
"""
import itertools
import time

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

METODOS = ("grid", "random", "bayes", "genetico")


def _objetivo(est, X, y, cv):
    """Devuelve f(params) = MAE medio de validación cruzada, con memoria de lo ya evaluado."""
    memoria = {}

    def f(params):
        clave = tuple(sorted((k, str(v)) for k, v in params.items()))
        if clave not in memoria:
            m = clone(est).set_params(**params)
            memoria[clave] = -cross_val_score(m, X, y, cv=cv, scoring="neg_mean_absolute_error").mean()
        return memoria[clave]
    return f


def decodificar(u, espacio):
    """Convierte un punto del cubo unitario [0, 1]^d en un dict de hiperparámetros."""
    params = {}
    for x, (nombre, spec) in zip(np.clip(u, 0, 1), espacio.items()):
        tipo = spec[0]
        if tipo == "log":
            params[nombre] = float(np.exp(np.log(spec[1]) + x * (np.log(spec[2]) - np.log(spec[1]))))
        elif tipo == "real":
            params[nombre] = float(spec[1] + x * (spec[2] - spec[1]))
        elif tipo == "int":
            params[nombre] = int(min(spec[2], spec[1] + int(x * (spec[2] - spec[1] + 1))))
        else:
            params[nombre] = spec[1][min(len(spec[1]) - 1, int(x * len(spec[1])))]
    return params


def _resultado(evaluados, t0, **extra):
    historial = [v for _, v in evaluados]
    mejor = int(np.argmin(historial))
    return {"params": evaluados[mejor][0], "puntaje": historial[mejor], "historial": historial,
            "segundos": time.perf_counter() - t0, **extra}


def grid(est, rejilla, X, y, cv):
    """Búsqueda exhaustiva: evalúa todas las combinaciones de la rejilla."""
    t0, f = time.perf_counter(), _objetivo(est, X, y, cv)
    combos = [dict(zip(rejilla, v)) for v in itertools.product(*rejilla.values())]
    return _resultado([(p, f(p)) for p in combos], t0)


def random(est, espacio, X, y, cv, n=16, semilla=42):
    """Búsqueda aleatoria: n puntos uniformes en el espacio (uniformes en log donde corresponde)."""
    t0, f = time.perf_counter(), _objetivo(est, X, y, cv)
    rng = np.random.default_rng(semilla)
    puntos = [decodificar(rng.random(len(espacio)), espacio) for _ in range(n)]
    return _resultado([(p, f(p)) for p in puntos], t0)


def bayes(est, espacio, X, y, cv, n=16, semilla=42):
    """Optimización bayesiana con Optuna.

    Modelo sustituto: TPE (Tree-structured Parzen Estimator), que modela la densidad de las
    configuraciones buenas l(x) y de las malas g(x). Función de adquisición: mejora esperada, que en
    TPE equivale a maximizar l(x)/g(x). Las primeras evaluaciones son aleatorias (exploración) y las
    siguientes se concentran donde l/g es alto (explotación).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0, f = time.perf_counter(), _objetivo(est, X, y, cv)
    evaluados = []

    def prueba(trial):
        params = {}
        for nombre, spec in espacio.items():
            if spec[0] == "log":
                params[nombre] = trial.suggest_float(nombre, spec[1], spec[2], log=True)
            elif spec[0] == "real":
                params[nombre] = trial.suggest_float(nombre, spec[1], spec[2])
            elif spec[0] == "int":
                params[nombre] = trial.suggest_int(nombre, spec[1], spec[2])
            else:
                params[nombre] = spec[1][trial.suggest_int(nombre, 0, len(spec[1]) - 1)]
        valor = f(params)
        evaluados.append((params, valor))
        return valor

    muestreador = optuna.samplers.TPESampler(seed=semilla, n_startup_trials=min(6, n // 2))
    optuna.create_study(direction="minimize", sampler=muestreador).optimize(prueba, n_trials=n)
    return _resultado(evaluados, t0)


def genetico(est, espacio, X, y, cv, n=16, semilla=42, poblacion=4, p_cruce=0.7, p_mut=0.3):
    """Algoritmo genético con DEAP sobre el cubo unitario.

    Selección por torneo (tamaño 2), cruce por mezcla (blend, alfa = 0.3) con probabilidad p_cruce,
    mutación gaussiana (sigma = 0.2 por gen, probabilidad 0.5 por gen) con probabilidad p_mut y
    elitismo de 1 individuo. Con poblacion = 4 y n = 16 se evalúan 4 generaciones de 4 individuos:
    el mismo presupuesto que las búsquedas aleatoria y bayesiana. La diversidad de cada generación
    es la desviación estándar media de los genes.
    """
    import random as _random

    from deap import base, creator, tools
    t0, f = time.perf_counter(), _objetivo(est, X, y, cv)
    _random.seed(semilla)
    rng = np.random.default_rng(semilla)
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individuo", list, fitness=creator.FitnessMin)
    d, evaluados, diversidad = len(espacio), [], []

    def evaluar(ind):
        params = decodificar(np.array(ind), espacio)
        ind.fitness.values = (f(params),)
        evaluados.append((params, ind.fitness.values[0]))

    pob = [creator.Individuo(rng.random(d).tolist()) for _ in range(poblacion)]
    for ind in pob:
        evaluar(ind)
    diversidad.append(float(np.std(np.array(pob), axis=0).mean()))
    while len(evaluados) + poblacion <= n:
        elite = tools.selBest(pob, 1)[0]
        hijos = [creator.Individuo(list(i)) for i in tools.selTournament(pob, poblacion, tournsize=2)]
        for a, b in zip(hijos[::2], hijos[1::2]):
            if _random.random() < p_cruce:
                tools.cxBlend(a, b, alpha=0.3)
        for h in hijos:
            if _random.random() < p_mut:
                tools.mutGaussian(h, mu=0, sigma=0.2, indpb=0.5)
            h[:] = [min(1.0, max(0.0, g)) for g in h]
            evaluar(h)
        pob = tools.selBest(hijos, poblacion - 1) + [elite]
        diversidad.append(float(np.std(np.array(pob), axis=0).mean()))
    return _resultado(evaluados, t0, diversidad=diversidad)


def optimizar(metodo, est, rejilla, espacio, X, y, cv, n=16, semilla=42):
    """Ejecuta el método pedido con el mismo presupuesto n (grid usa su rejilla completa)."""
    if metodo == "grid":
        return grid(est, rejilla, X, y, cv)
    return {"random": random, "bayes": bayes, "genetico": genetico}[metodo](est, espacio, X, y, cv, n=n, semilla=semilla)
