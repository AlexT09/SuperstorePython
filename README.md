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
| EDA | Distribución de las ventas, factores, estacionalidad, tendencia, atípicos y relación con el mes del año anterior |
| Modelos y resultados | 7 modelos evaluados en 2017, rejillas de Grid Search, sobreajuste (entrenamiento vs. prueba) e importancia de variables |
| Comparación | Ranking de los modelos frente a la referencia y prueba de la proyección a 12 meses |
| Proyección 2018 | Proyección por categoría, región y mes, con su error típico |
| Predicción | Formulario interactivo: carga `model.pkl` y predice las ventas de un mes en tiempo real |
| Conclusiones | Hallazgos, recomendaciones y limitaciones |

## Modelo

- **Qué predice:** las ventas del mes de cada categoría × región (12 series × 48 meses).
- **Variables:** categoría, región, mes del año y el historial de ventas (mismo mes del año anterior,
  mes anterior, promedio de 3 meses), todas conocidas antes de empezar el mes.
- **Objetivo:** un índice, ventas del mes ÷ promedio de los 12 meses anteriores. Se convierte a
  dólares al final, así la misma regla sirve para series grandes y pequeñas.
- **Modelos (scikit-learn / XGBoost):** Ridge, Lasso, KNN, Árbol de decisión, Random Forest,
  XGBoost y SVR, dentro de un `Pipeline` con `OneHotEncoder` y `StandardScaler`.
- **Ajuste:** `GridSearchCV` con `TimeSeriesSplit` (4 pliegues).
- **Evaluación:** entrenamiento 2015–2016, prueba 2017, contra la referencia "repetir el mismo mes
  del año anterior". El mejor modelo se elige por validación cruzada, no por la prueba.
- **Proyección 2018:** recursivo, mes a mes. Se valida igual: proyectando 2017 completo solo con
  datos hasta 2016.

Resultado actual: Random Forest, MAE por serie de $2.634 frente a $2.912 de la referencia. En la
prueba a 12 meses el total anual quedó a +3,4 % del real (la referencia, a −16,9 %), con un error
típico mensual de ±18 % para el total de la empresa.

## Requisitos

- **Python** 3.10 o superior

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
python model/train_model.py          # entrena y compara (~30 s) -> model/artifacts/
```


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
├── tabs/                     # Una pestaña por archivo, cada una expone layout()
└── assets/                   # Estilos de la app Dash (style.css) y foto de fondo
```

## Problemas comunes

- **Las pestañas del modelo dicen "Modelo no entrenado":** ejecuta `python model/train_model.py`.
- **`python` no se reconoce en Windows:** prueba con `py app.py` y `py -m pip install -r requirements.txt`.
- **El puerto 8050 está ocupado:** cierra el otro proceso que lo esté usando.

---

Proyecto: Alex Teran y David Estrada
