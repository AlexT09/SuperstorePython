# Proyección de ventas Sample Superstore

Análisis y proyección de ventas del dataset **Sample Superstore** (2014–2017), en dos partes:

1. **EDA:** ¿qué factores (categoría, región, segmento, tiempo) explican el nivel de ventas?
2. **Machine learning:** ¿cuánto venderá cada categoría en cada región, mes a mes, durante 2018?

## Contenido

Una app **Dash** con una pestaña por etapa del proyecto:

| Pestaña | Qué muestra |
|---|---|
| Introducción · Problema y objetivos · Marco teórico · Metodología | Planteamiento, modelos, variables y flujo |
| ETL | Del CSV crudo a la tabla mensual que usa el modelo |
| EDA | Distribución de las ventas, factores, estacionalidad y tendencia |
| Modelos y resultados | 7 modelos evaluados en 2017, importancia de variables |
| Comparación | Ranking frente a la referencia, Grid vs. Random Search, residuos, prueba a 12 meses |
| Proyección 2018 | Proyección por categoría, región y mes, con su error típico |
| Predicción | Formulario interactivo: carga `model.pkl` y predice las ventas de un mes en tiempo real |
| Conclusiones | Hallazgos, recomendaciones y limitaciones |
| Presentación | El Panel de ventas (`panel/panel_ventas.html`) con todo el proyecto, para exponer |

## Modelo

- **Qué predice:** las ventas del mes de cada categoría × región (12 series × 48 meses).
- **Variables:** categoría, región, mes del año y el historial de ventas (mismo mes del año anterior,
  mes anterior, promedio de 3 meses), todas conocidas antes de empezar el mes.
- **Objetivo:** un índice, ventas del mes ÷ promedio de los 12 meses anteriores. Se convierte a
  dólares al final, así la misma regla sirve para series grandes y pequeñas.
- **Modelos (scikit-learn / XGBoost):** Ridge, Lasso, KNN, Árbol de decisión, Random Forest,
  XGBoost y SVR, dentro de un `Pipeline` con `OneHotEncoder` y `StandardScaler`.
- **Ajuste:** `GridSearchCV` y `RandomizedSearchCV` con `TimeSeriesSplit` (4 pliegues).
- **Evaluación:** entrenamiento 2015–2016, prueba 2017, contra la referencia "repetir el mismo mes
  del año anterior". El mejor modelo se elige por validación cruzada, no por la prueba.
- **Proyección 2018:** recursivo, mes a mes. Se valida igual: proyectando 2017 completo solo con
  datos hasta 2016.

Resultado actual: Random Forest, MAE por serie de $2.634 frente a $2.912 de la referencia. En la
prueba a 12 meses el total anual quedó a +3,4 % del real (la referencia, a −16,9 %), con un error
típico mensual de ±18 % para el total de la empresa.

## Requisitos

- **Python** 3.10 o superior
- Internet para el Panel de ventas (Plotly y fuentes se cargan de CDN)

## Cómo correr el proyecto

```powershell
python -m venv .venv
.venv\Scripts\activate       # en macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Después abre **http://127.0.0.1:8050** en el navegador.

Los datos procesados y los resultados del modelo ya vienen en el repositorio, así que no hace falta
regenerarlos. Si cambias el CSV original o el modelo, ejecuta los pasos en este orden:

```powershell
python data/generate_data.py         # CSV crudo -> superstore_transformado.csv y ventas_mensuales.csv
python model/train_model.py          # entrena y compara (~35 s) -> model/artifacts/
python dashboard/build_dashboard.py  # resultados -> panel/panel_data.json
```

## Panel de ventas (`panel/`)

`panel/panel_ventas.html` es una página independiente con estilo de vidrio que contiene **el proyecto
completo**, con una sección por etapa en la barra lateral:

- **Proyecto:** Introducción, Problema y objetivos, Marco teórico, Metodología.
- **Datos:** ETL (código, calidad de datos, tabla mensual) y EDA (univariado, bivariado y en el tiempo).
- **Modelo:** Modelos y resultados, Comparación (ranking, Grid vs. Random Search, prueba a 12 meses,
  residuos) y Proyección 2018 (banda de error y tabla por categoría × región).
- **Panel de ventas:** Resumen 2018 (proyección, curva frente a 2017, tarjeta por región) y Detalle
  (historia + proyección, estacionalidad, categoría × región, subcategorías y mapa).
- **Cierre:** Conclusiones, recomendaciones y limitaciones.

Filtra por categoría y región desde el encabezado (en Resumen también con las tarjetas de la
derecha). `app.py` lo sirve en **http://127.0.0.1:8050/panel/panel_ventas.html**; `#eda`,
`#pronostico`, etc. al final de la URL abre esa sección, y `panel_data.json` se genera si falta.

## Estructura

```
├── app.py                    # Punto de entrada de la app Dash
├── common.py                 # Rutas, carga de datos y resultados, estilos y componentes compartidos
├── requirements.txt          # Dependencias con versiones exactas
├── data/
│   ├── raw/                  # Dataset original
│   ├── generate_data.py      # ETL
│   ├── superstore_transformado.csv
│   └── ventas_mensuales.csv  # Categoría × región × mes (entrada del modelo)
├── model/
│   ├── features.py           # Variables y preprocesamiento (compartidos)
│   ├── evaluate.py           # Métricas e importancia de variables
│   ├── train_model.py        # Entrena, compara, proyecta 2018
│   └── artifacts/            # model.pkl, resultados, predicciones y proyección (generados)
├── dashboard/
│   └── build_dashboard.py    # Genera panel/panel_data.json
├── panel/                    # Panel de ventas (servido por app.py en /panel/)
├── tabs/                     # Una pestaña por archivo, cada una expone layout()
└── assets/style.css          # Estilos de la app Dash
```

## Problemas comunes

- **Las pestañas del modelo dicen "Modelo no entrenado":** ejecuta `python model/train_model.py`.
- **El Panel de ventas no muestra gráficos:** necesita internet (Plotly viene de un CDN).
- **`python` no se reconoce en Windows:** prueba con `py app.py` y `py -m pip install -r requirements.txt`.
- **El puerto 8050 está ocupado:** cierra el otro proceso que lo esté usando.

---

Proyecto: Alex Teran y David Estrada
