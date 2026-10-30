# Proyección de ventas Sample Superstore

Análisis y proyección de ventas del dataset **Sample Superstore** (2014–2017), en dos partes:

1. **EDA:** ¿qué factores (categoría, región, segmento, tiempo) explican el nivel de ventas?
2. **Machine learning:** ¿cuánto venderá cada categoría en cada región, mes a mes, durante 2018?

## Contenido

El proyecto se presenta de dos formas, con los mismos datos y el mismo modelo:

- **App Dash** (`app.py`): panel interactivo. Publicada en **https://superstorepython.onrender.com**
  (también se puede correr en local, ver [Cómo correr el proyecto](#cómo-correr-el-proyecto)).
- **Libro Jupyter Book** (`libro/`): el trabajo en notebooks, con el código comentado y el
  entrenamiento de los modelos a la vista. Publicado en **https://alext09.github.io/SuperstorePython/**
  (ver [Libro (Jupyter Book)](#libro-jupyter-book)).

### App Dash

Una pestaña por etapa del proyecto:

| Pestaña | Qué muestra |
|---|---|
| Introducción · Problema y objetivos · Marco teórico · Metodología | Planteamiento, modelos, variables y flujo |
| ETL | Del CSV crudo a la tabla mensual que usa el modelo |
| EDA | Distribución de las ventas, factores, estacionalidad, tendencia, atípicos, relación con el mes del año anterior, mapas por estado (ventas totales y venta mediana por línea), correlación de Sales con cantidad y descuento, contenedores 2D (descuento vs. ventas) y burbujas por subcategoría |
| Modelos y resultados | 7 modelos evaluados en 2017, ranking frente a la referencia, rejillas de Grid Search, sobreajuste (entrenamiento vs. prueba), importancia de variables y prueba de la proyección a 12 meses; al final, el entrenamiento del modelo: elige uno de los 7 modelos, lo entrena al presionar un botón (Grid Search con validación cruzada temporal y evaluación en 2017) y permite descargarlo como `.pkl` |
| Proyección y predicción | Proyección 2018 por categoría, región y mes, con su error típico, y al final la predicción interactiva: formulario que carga `model.pkl` y predice las ventas de un mes en tiempo real |
| Conclusiones | Hallazgos, recomendaciones y limitaciones |

Además, en todas las pestañas hay un **Asistente IA** (botón 💬 abajo a la derecha).

## Asistente IA

Chatbot de atención al cliente que traduce la proyección de ventas 2018 en recomendaciones para el
negocio (objetivo específico 5): **metas** por región, **inventario**, **alertas** y **presupuesto**
por trimestre, con rangos basados en el error medido al proyectar 2017. También explica qué factores
explican las ventas (EDA) y cómo se eligió el modelo.

- Responde solo con los datos y resultados del proyecto; si algo no está, lo dice en vez de inventarlo.
- Atajos de decisión, sugerencias según la pestaña abierta, calificación 👍/👎 de cada respuesta
  (`logs/chat_feedback.csv`) y descarga de la conversación.
- Usa la librería `openai`, compatible con varios proveedores. Por defecto, **Google Gemini** (nivel
  gratuito).

Para activarlo, copia `.env.example` como `.env` y pon tu clave (se crea gratis en
[Google AI Studio](https://aistudio.google.com) → *Get API key*). El archivo `.env` está en
`.gitignore`: la clave nunca se sube al repositorio. Sin clave, la app funciona igual y el asistente
avisa que no puede responder.

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
copy .env.example .env       # opcional, para el Asistente IA: luego pon tu clave en .env
python app.py
```

Después abre **http://127.0.0.1:8050** en el navegador.

Los datos procesados y los resultados del modelo ya vienen en el repositorio, así que no hace falta
regenerarlos. Si cambias el CSV original o el modelo, ejecuta los pasos en este orden:

```powershell
python data/generate_data.py         # CSV crudo -> superstore_transformado.csv y ventas_mensuales.csv
python model/train_model.py          # entrena y compara (~30 s) -> model/artifacts/
```

## Libro (Jupyter Book)

El mismo proyecto en notebooks, en la carpeta `libro/`. Usa los datos (`data/`) y el modelo
(`model/`) de la raíz del repositorio, así que muestra las mismas cifras que la app. Publicado en
**https://alext09.github.io/SuperstorePython/**.

| Capítulo | Archivo | Pestaña de la app |
|---|---|---|
| Introducción · Problema y objetivos · Marco teórico · Metodología | `intro.md`, `problema.md`, `marco_teorico.md`, `metodologia.md` | Las mismas |
| ETL | `etl.ipynb` | ETL |
| EDA | `eda.ipynb` | EDA |
| Modelos y resultados | `modelos.ipynb` | Modelos y resultados |
| Proyección 2018 | `proyeccion.ipynb` | Proyección y predicción |
| Conclusiones | `conclusiones.ipynb` | Conclusiones |

*Modelos y resultados* empieza entrenando los 7 modelos con el código a la vista: división en el
tiempo, rejillas de hiperparámetros, `GridSearchCV` con `TimeSeriesSplit` y evaluación en 2017.
Al final comprueba que llega a los mismos resultados guardados en `model/artifacts/`. Todo el código
de los notebooks está comentado; en el libro se despliega con el botón *Mostrar código*.

La predicción interactiva, el entrenamiento con botón y el Asistente IA son solo de la app
(https://superstorepython.onrender.com): necesitan un servidor de Python, y GitHub Pages solo sirve
páginas estáticas.

### Cómo construir el libro

```powershell
pip install -r libro/requirements.txt   # las dependencias de la app más Jupyter Book
cd libro
jupyter-book build .                    # ejecuta los notebooks (~1 min) -> libro/_build/html/
```

Ábrelo con doble clic en `libro/_build/html/index.html`. Las gráficas son de Plotly: el libro carga
plotly.js una sola vez (`_config.yml` → `html_js_files`), por eso en Jupyter los notebooks no las
muestran y en el libro construido sí.

### Cómo publicarlo

El libro publicado se sirve desde la rama `gh-pages`, que contiene solo el HTML de
`libro/_build/html/` más un archivo `.nojekyll` (sin él GitHub no sirve las carpetas que empiezan
con `_`). Para actualizarlo, construye el libro y reemplaza el contenido de esa rama:

```powershell
git worktree add ..\gh-pages gh-pages        # la rama en una carpeta aparte
cd ..\gh-pages
git rm -r -q .
Copy-Item -Recurse ..\SuperstorePython\libro\_build\html\* .
New-Item .nojekyll -ItemType File -Force
git add -A; git commit -m "Publica el libro"; git push origin gh-pages
cd ..\SuperstorePython; git worktree remove ..\gh-pages
```

En GitHub, *Settings → Pages* debe tener como origen la rama `gh-pages` (carpeta `/`).

## Estructura

```
├── app.py                    # Punto de entrada de la app Dash
├── common.py                 # Rutas, carga de datos y resultados, estilos y componentes compartidos
├── requirements.txt          # Dependencias con versiones exactas
├── .env.example              # Plantilla de configuración del Asistente IA (copiar como .env)
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
│   ├── asistente.py          # Asistente IA: widget flotante, atajos y conexión con el modelo de IA
│   └── asistente_contexto.py # Lo que sabe el asistente: instrucciones y resúmenes de datos
├── assets/                   # Estilos (style.css, chat.css) y efectos interactivos (ui.js)
└── libro/                    # Jupyter Book: el proyecto en notebooks
    ├── _config.yml           # Configuración del libro
    ├── _toc.yml              # Tabla de contenidos (mismo orden que las pestañas de la app)
    ├── common.py             # Rutas a data/ y model/, carga de resultados, paleta y show() para las figuras
    ├── requirements.txt      # Dependencias de la app + Jupyter Book
    ├── intro.md … metodologia.md       # Capítulos de texto
    └── etl.ipynb … conclusiones.ipynb  # Capítulos con código y gráficas
```

## Problemas comunes

- **Las pestañas del modelo dicen "Modelo no entrenado":** ejecuta `python model/train_model.py`.
- **`python` no se reconoce en Windows:** prueba con `py app.py` y `py -m pip install -r requirements.txt`.
- **El puerto 8050 está ocupado:** cierra el otro proceso que lo esté usando.
- **`No module named 'openai'`:** instala las dependencias con el mismo Python con el que corres la app
  (`python -m pip install -r requirements.txt`).
- **El asistente responde con un error:** revisa la clave en `.env`; si el error es 404, cambia
  `OPENAI_MODEL` por un modelo vigente (la lista está en Google AI Studio).

---

Proyecto: Alex Teran y David Estrada
