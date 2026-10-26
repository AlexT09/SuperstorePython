"""
dashboard/build_dashboard.py — Genera los datos del Panel de ventas (panel/panel_ventas.html).

El panel contiene el proyecto completo: planteamiento, ETL, EDA, ventas históricas, modelos,
comparación, proyección 2018 y conclusiones. Toda la información viaja en un solo JSON con
arreglos compactos; el navegador suma según los filtros (categoría y región) sin volver a
llamar al servidor.

Entradas:  data/superstore_transformado.csv, data/ventas_mensuales.csv, model/artifacts/*
Salida:    panel/panel_data.json

Ejecutar (después de model/train_model.py):  python dashboard/build_dashboard.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (REFERENCIA, ROOT, STATE_ABBR, load_artifact, load_data, load_mensual,  # noqa: E402
                    mejores_por_modelo)

PANEL_DIR = ROOT / "panel"
DATA_OUT = PANEL_DIR / "panel_data.json"

CATS = ["Furniture", "Office Supplies", "Technology"]
REGIONS = ["Central", "East", "South", "West"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
YEARS = [2014, 2015, 2016, 2017]


def _cubo(df, valor, ejes):
    """Arreglo anidado [eje1][eje2]...[mes|año] con ceros donde no hay ventas."""
    idx = pd.MultiIndex.from_product([e for _, e in ejes], names=[n for n, _ in ejes])
    s = df.groupby([n for n, _ in ejes])[valor].sum().reindex(idx, fill_value=0).round(0).astype(int)
    shape = [len(e) for _, e in ejes]
    flat = s.tolist()

    def anidar(datos, dims):
        if len(dims) == 1:
            return datos
        paso = len(datos) // dims[0]
        return [anidar(datos[i * paso:(i + 1) * paso], dims[1:]) for i in range(dims[0])]
    return anidar(flat, shape)


def _loess(serie):
    """Curva loess de una serie mensual (mismo ajuste que trendline="lowess" de plotly express)."""
    x = pd.to_datetime(serie.index).astype("int64") / 1e9
    return np.round(lowess(serie.to_numpy(), x, return_sorted=False), 0).astype(int).tolist()


def _eda(df, mensual):
    """Pedidos individuales (columnas) y resúmenes del análisis exploratorio."""
    s = df.Sales
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    total = mensual.groupby("Mes").Sales.sum()
    por = lambda col, vals: [_loess(mensual[mensual[col] == v].groupby("Mes").Sales.sum()) for v in vals]  # noqa: E731
    corr = df[["Sales", "Quantity", "Discount", "Profit"]].corr().round(2)
    return {
        "segs": SEGMENTS,
        # una entrada por línea de pedido: ventas, descuento e índices de categoría, región y segmento
        "pedidos": {"sales": s.round(2).tolist(), "disc": df.Discount.round(2).tolist(),
                    "cat": df.Category.map(CATS.index).tolist(), "reg": df.Region.map(REGIONS.index).tolist(),
                    "seg": df.Segment.map(SEGMENTS.index).tolist()},
        "stats": {"n": len(s), "media": s.mean(), "ds": s.std(), "mediana": s.median(), "q1": q1, "q3": q3,
                  "ric": q3 - q1, "min": s.min(), "max": s.max()},
        "corr": {"vars": corr.columns.tolist(), "z": corr.to_numpy().tolist()},
        "pedidosEstado": df.groupby("State").size().to_dict(),
        "loess": {"total": _loess(total), "cat": por("Category", CATS), "reg": por("Region", REGIONS)},
    }


def _modelos(resumen, modelos, todos, ref, imp):
    """Métricas, búsquedas de hiperparámetros, ajuste mensual y residuos de cada modelo."""
    pred = load_artifact("predicciones.csv")
    nombres = modelos.modelo.tolist() + [REFERENCIA]
    tot = pred.groupby("Mes")[["Sales"] + nombres].sum().round(0)
    prueba = pred[pred.conjunto == "Prueba"]
    best = resumen["mejor_modelo"]
    fila = lambda f: {"modelo": f.modelo, "busqueda": f.busqueda, "mae": f.MAE, "rmse": f.RMSE, "r2": f.R2,  # noqa: E731
                      "totalPct": f.total_MAE_pct, "params": json.loads(f.hiperparametros)}
    return {
        "tabla": [fila(f) for f in modelos.itertuples()],
        "ref": {"mae": ref.MAE, "rmse": ref.RMSE, "r2": ref.R2, "totalPct": ref.total_MAE_pct},
        "busquedas": [{"modelo": f.modelo, "busqueda": f.busqueda, "cv": f.cv_MAE_indice,
                       "evals": int(f.evaluaciones), "tiempo": f.tiempo_s} for f in todos.itertuples()],
        # total mensual de la empresa 2015–2017: real y proyección de cada modelo
        "ajuste": {"meses": [m.strftime("%Y-%m") for m in tot.index], "real": tot.Sales.astype(int).tolist(),
                   "inicioPrueba": prueba.Mes.min().strftime("%Y-%m"),
                   "pred": {m: tot[m].astype(int).tolist() for m in nombres}},
        # residuos del mejor modelo en 2017, uno por categoría × región × mes
        "residuos": (prueba.Sales - prueba[best]).round(0).astype(int).tolist(),
        "prueba": resumen["prueba"], "pruebaRef": resumen["referencia"], "recursiva": resumen["prueba_recursiva"],
        "hiper": resumen["hiperparametros"], "filas": resumen["filas"],
        "ventas2017": resumen["ventas_2017"], "pron2018": resumen["pronostico_2018"],
    }


def build() -> Path:
    df = load_data().assign(Anio=lambda d: d.Order_Date.dt.year)
    mensual = load_mensual()
    meses = sorted(mensual.Mes.unique())
    resumen = load_artifact("resumen.json")
    pron = load_artifact("pronostico_2018.csv")
    rec = load_artifact("prueba_recursiva_2017.csv")
    meses18 = sorted(pron.Mes.unique())
    meses17 = sorted(rec.Mes.unique())

    subcats = (df.groupby(["Category", "Sub-Category"]).size().reset_index()
                 .sort_values(["Category", "Sub-Category"]))
    states = sorted(STATE_ABBR)
    region_of = df.groupby("State").Region.first()
    modelos = mejores_por_modelo()
    todos = load_artifact("resultados_modelos.csv")
    ref = todos.query("modelo == @REFERENCIA").iloc[0]
    todos = todos.query("modelo != @REFERENCIA")
    imp = load_artifact("importancia.csv")

    data = {
        "cats": CATS, "regs": REGIONS, "years": YEARS,
        "meses": [pd.Timestamp(m).strftime("%Y-%m") for m in meses],
        "meses18": [pd.Timestamp(m).strftime("%Y-%m") for m in meses18],
        # ventas mensuales reales [categoría][región][48 meses]
        "real": _cubo(mensual, "Sales", [("Category", CATS), ("Region", REGIONS), ("Mes", meses)]),
        # proyección 2018 [categoría][región][12 meses]
        "pron": _cubo(pron, "Pronostico", [("Category", CATS), ("Region", REGIONS), ("Mes", meses18)]),
        # 2017 proyectado a 12 meses con datos hasta 2016 [categoría][región][12 meses]
        "prueba": _cubo(rec, "Pronostico", [("Category", CATS), ("Region", REGIONS), ("Mes", meses17)]),
        "subcats": subcats["Sub-Category"].tolist(),
        "subcatCat": [CATS.index(c) for c in subcats.Category],
        # ventas por subcategoría [subcategoría][región][año]
        "sub": _cubo(df, "Sales", [("Sub-Category", subcats["Sub-Category"].tolist()), ("Region", REGIONS),
                                   ("Anio", YEARS)]),
        "states": states, "abbr": [STATE_ABBR[s] for s in states],
        "stateReg": [REGIONS.index(region_of[s]) if s in region_of else -1 for s in states],
        # ventas por estado [estado][categoría][año]
        "st": _cubo(df, "Sales", [("State", states), ("Category", CATS), ("Anio", YEARS)]),
        "modelo": {
            "nombre": resumen["mejor_modelo"], "busqueda": resumen["busqueda"],
            "mae": resumen["prueba"]["MAE"], "maeRef": resumen["referencia"]["MAE"],
            "desvioAnual": resumen["prueba_recursiva"]["total_desvio_total_pct"],
            "desvioAnualRef": resumen["referencia"]["total_desvio_total_pct"],
            "errorMensual": resumen["prueba_recursiva"]["total_MAE_pct"],
            "ranking": [{"modelo": f.modelo, "mae": round(f.MAE, 1)} for f in modelos.itertuples()]
                       + [{"modelo": REFERENCIA, "mae": round(ref.MAE, 1)}],
            "importancia": [{"nombre": f.nombre, "valor": round(f.importancia, 5), "std": round(f.std, 5)}
                            for f in imp.itertuples()],
            **_modelos(resumen, modelos, todos, ref, imp),
        },
        "etl": {"lineas": len(df), "columnas": df.shape[1] - 1, "faltantes": int(df.isna().sum().sum()),
                "duplicados": int(df.duplicated().sum()), "filasMensual": len(mensual),
                "ceros": int((mensual.Sales == 0).sum()),
                "desde": df.Order_Date.min().strftime("%Y-%m-%d"), "hasta": df.Order_Date.max().strftime("%Y-%m-%d")},
        "eda": _eda(df, mensual),
    }
    PANEL_DIR.mkdir(exist_ok=True)
    DATA_OUT.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    return DATA_OUT


if __name__ == "__main__":
    out = build()
    print(f"OK: {out} ({out.stat().st_size / 1024:,.0f} KB)")
