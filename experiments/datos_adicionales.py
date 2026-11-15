"""
experiments/datos_adicionales.py — Descarga y prepara los tres conjuntos de datos adicionales.

Fuente: Monash Time Series Forecasting Repository (Godahewa et al., 2021), archivado en Zenodo con
licencia CC BY 4.0. Los tres conjuntos son paneles mensuales de naturaleza distinta:

    turismo   Tourism Monthly   doi:10.5281/zenodo.4656096   demanda turística, estacionalidad fuerte
    hospital  Hospital          doi:10.5281/zenodo.4656014   conteos de pacientes, estacionalidad débil
    m3        M3 Monthly        doi:10.5281/zenodo.4656298   series heterogéneas de seis dominios

Regla de selección de series (fijada en experiments/protocolo.md antes de evaluar ningún modelo):
  1. series que terminan en la fecha final más frecuente del conjunto (para tener un calendario común);
  2. con al menos 48 meses y todos los valores de los últimos 48 meses mayores que cero;
  3. de las elegibles, ordenadas por nombre, se toman 24 al azar con la semilla 42;
  4. se conservan los últimos 48 meses (el mismo largo que Sample Superstore).

Salida: data/adicionales/<nombre>.csv con las columnas Category (serie), Region ("-"), Mes, Sales.

Ejecutar desde la raíz:  python experiments/datos_adicionales.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SALIDA = ROOT / "data" / "adicionales"
SEMILLA, N_SERIES, N_MESES = 42, 24, 48
FUENTES = {
    "turismo": ("4656096", "tourism_monthly_dataset"),
    "hospital": ("4656014", "hospital_dataset"),
    "m3": ("4656298", "m3_monthly_dataset"),
}


def leer_tsf(texto):
    """Lee el formato .tsf de Monash: devuelve [(nombre, fecha_inicial, valores)]."""
    series, en_datos = [], False
    for linea in texto.splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        if linea.lower().startswith("@data"):
            en_datos = True
            continue
        if not en_datos:
            continue
        *atributos, valores = linea.split(":")
        nombre = atributos[0]
        inicio = pd.Timestamp(atributos[-1].split(" ")[0]) if len(atributos) > 1 else pd.NaT
        v = np.array([np.nan if x == "?" else float(x) for x in valores.split(",")])
        series.append((nombre, inicio, v))
    return series


def preparar(nombre, registro, archivo):
    url = f"https://zenodo.org/records/{registro}/files/{archivo}.zip?download=1"
    pedido = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(pedido, timeout=120) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    series = leer_tsf(z.read(z.namelist()[0]).decode("latin1"))

    # fecha del último dato de cada serie (primer día del mes)
    fin = {n: (i.to_period("M") + len(v) - 1).to_timestamp() for n, i, v in series}
    fin_comun = pd.Series(fin).mode().iloc[0]
    elegibles = sorted(n for n, i, v in series
                       if fin[n] == fin_comun and len(v) >= N_MESES
                       and np.isfinite(v[-N_MESES:]).all() and (v[-N_MESES:] > 0).all())
    rng = np.random.default_rng(SEMILLA)
    elegidas = set(rng.choice(elegibles, size=N_SERIES, replace=False))
    meses = pd.date_range(end=fin_comun, periods=N_MESES, freq="MS")
    panel = pd.concat([pd.DataFrame({"Category": n, "Region": "-", "Mes": meses, "Sales": v[-N_MESES:]})
                       for n, i, v in series if n in elegidas], ignore_index=True)
    SALIDA.mkdir(parents=True, exist_ok=True)
    panel.to_csv(SALIDA / f"{nombre}.csv", index=False)
    print(f"{nombre:9s} {len(series):5d} series · {len(elegibles):5d} elegibles · {N_SERIES} elegidas · "
          f"{meses[0]:%Y-%m} a {meses[-1]:%Y-%m} · venta media {panel.Sales.mean():,.1f}")


if __name__ == "__main__":
    for nombre, (registro, archivo) in FUENTES.items():
        preparar(nombre, registro, archivo)
