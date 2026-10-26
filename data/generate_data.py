"""
data/generate_data.py — ETL del proyecto.

Extracción:     data/raw/Sample - Superstore.csv
Transformación: fechas a tipo fecha, orden cronológico y tabla mensual de ventas
Carga:          data/superstore_transformado.csv  (una fila por línea de pedido, la usa el EDA)
                data/ventas_mensuales.csv         (una fila por categoría × región × mes, la usa el modelo)

Ejecutar:  python data/generate_data.py
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw" / "Sample - Superstore.csv"
OUT = BASE / "superstore_transformado.csv"
OUT_MENSUAL = BASE / "ventas_mensuales.csv"


def tabla_mensual(df: pd.DataFrame) -> pd.DataFrame:
    """Ventas totales por categoría × región × mes.

    Incluye todas las combinaciones (3 categorías × 4 regiones × 48 meses): un mes sin
    pedidos queda con ventas 0 en lugar de desaparecer, para que la serie no tenga huecos.
    """
    meses = pd.date_range(df.Order_Date.min().to_period("M").to_timestamp(),
                          df.Order_Date.max().to_period("M").to_timestamp(), freq="MS")
    idx = pd.MultiIndex.from_product(
        [sorted(df.Category.unique()), sorted(df.Region.unique()), meses],
        names=["Category", "Region", "Mes"])
    return (df.groupby(["Category", "Region", pd.Grouper(key="Order_Date", freq="MS")]).Sales.sum()
              .rename_axis(idx.names).reindex(idx, fill_value=0).round(2).reset_index())


def main() -> None:
    # --- Extracción
    df = pd.read_csv(RAW, encoding="latin1")

    # --- Transformación
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")
    df = df.rename(columns={"Order Date": "Order_Date"}).sort_values("Order_Date")
    mensual = tabla_mensual(df)

    # --- Carga
    df.to_csv(OUT, index=False)
    mensual.to_csv(OUT_MENSUAL, index=False)
    print(f"OK: {len(df):,} filas -> {OUT.name} · {len(mensual):,} filas -> {OUT_MENSUAL.name}")


if __name__ == "__main__":
    main()
