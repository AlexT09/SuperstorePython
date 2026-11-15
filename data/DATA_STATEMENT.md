# Declaración de los datos

## Conjunto principal: Sample Superstore

| | |
|---|---|
| **Qué es** | Líneas de pedido de una tienda minorista de Estados Unidos (muebles, artículos de oficina y tecnología). |
| **Procedencia** | Conjunto de ejemplo que Tableau distribuye con su software. Archivo: `data/raw/Sample - Superstore.csv`. |
| **Naturaleza** | Datos de demostración. No corresponden a una empresa identificada y no se conoce cómo se generaron. |
| **Periodo** | 3 de enero de 2014 a 30 de diciembre de 2017. |
| **Tamaño** | 9.994 líneas de pedido y 21 variables. |
| **Licencia** | Tableau no publica una licencia abierta para este archivo. Se usa aquí con fines académicos. |
| **Datos personales** | Incluye nombres de clientes ficticios. El proyecto no los usa: solo agrega ventas por categoría, región y mes. |

### Variables que usa el proyecto

| Variable | Tipo | Descripción |
|---|---|---|
| `Order Date` | fecha | Fecha del pedido; define el mes. |
| `Category` | categórica (3) | Furniture, Office Supplies, Technology. |
| `Region` | categórica (4) | Central, East, South, West. |
| `Sales` | numérica (USD) | Valor de la línea de pedido. Es lo que se pronostica, sumado por mes. |
| `Segment`, `Sub-Category`, `State`, `Quantity`, `Discount`, `Profit` | varias | Solo en el análisis exploratorio. |

### Transformación

`data/generate_data.py` limpia el archivo crudo (`data/superstore_transformado.csv`) y suma las ventas
por categoría, región y mes (`data/ventas_mensuales.csv`): 12 series de 48 meses, 576 filas, sin
meses vacíos. La variable objetivo del modelo es el índice ventas del mes ÷ promedio de los 12 meses
anteriores; por eso el primer año solo aporta historia.

### Sesgos y límites conocidos

- **No es una muestra de un negocio real.** Las conclusiones valen para este conjunto y sirven para
  comparar métodos; no deben trasladarse a una empresa sin validarlas con sus datos.
- **Un solo país y cuatro años.** No hay ciclos económicos completos, ni cambios de precios, ni
  promociones identificadas.
- **Series cortas.** Con 48 meses por serie hay tres o cuatro observaciones de cada mes del año: la
  estacionalidad de una serie individual se estima con mucho ruido.
- **Tamaños desiguales.** Unas pocas series concentran las ventas, de modo que el error en dólares
  lo dominan las series grandes.
- **Sin variables externas.** No hay festivos, clima, inventario ni competencia.

## Conjuntos adicionales

Se usan solo para evaluar si los resultados se mantienen fuera del conjunto principal. Provienen del
Monash Time Series Forecasting Repository (Godahewa, Bergmeir, Webb, Hyndman y Montero-Manso, 2021),
archivado en Zenodo con licencia **CC BY 4.0**.

| Archivo | Conjunto original | DOI | Series usadas | Periodo usado | Qué mide |
|---|---|---|---|---|---|
| `data/adicionales/turismo.csv` | Tourism Monthly (366 series) | 10.5281/zenodo.4656096 | 24 | oct. 2003 – sep. 2007 | demanda turística |
| `data/adicionales/hospital.csv` | Hospital (767 series) | 10.5281/zenodo.4656014 | 24 | ene. 2003 – dic. 2006 | conteos de pacientes |
| `data/adicionales/m3.csv` | M3 Monthly (1.428 series) | 10.5281/zenodo.4656298 | 24 | mar. 1990 – feb. 1994 | series de seis dominios |

`experiments/datos_adicionales.py` los descarga y aplica la regla de selección fijada en
`experiments/protocolo.md`: series con calendario común y los últimos 48 meses completos y positivos;
24 al azar con la semilla 42. Las columnas `Category` y `Region` se reutilizan como identificador de
la serie para que el mismo código sirva en los cuatro conjuntos.

**Límite:** son muestras de 24 series; los resultados en ellas no describen los conjuntos completos.
