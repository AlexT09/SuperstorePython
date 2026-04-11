# Gaming Profiles 2025 — PlayStation Dashboard

**Autor:** Alex Terán | Ciencia de Datos  
**Asignatura:** Visualización de Datos en R

Dashboard interactivo desarrollado con **R Shiny** para explorar el dataset
*Gaming Profiles 2025 — PlayStation*, que cubre información de juegos,
jugadores y precios de la plataforma PlayStation.

![Dashboard Preview](assets/preview.png)

---

## Descripción

La aplicación cuenta con **5 pestañas interactivas**:

| Pestaña | Contenido |
|---|---|
| 🏠 Resumen General | KPIs globales, plataformas, géneros y países |
| 🎮 Catálogo de Juegos | Top géneros, publishers y tabla filtrable |
| 👥 Jugadores | Distribución geográfica de los 356.600 usuarios |
| 💰 Precios | Histograma, boxplot por moneda y precio por género |
| 🔍 Explorador | Filtros dinámicos de precio, género y plataforma |

---

## Dataset

Los datos provienen de [Gaming Profiles 2025](https://www.kaggle.com/datasets/artyomkruglov/gaming-profiles-2025-steam-playstation-xbox)
(Kaggle, CC0 — dominio público).

| Archivo | Registros | Descripción |
|---|---|---|
| `games.csv` | 23.152 | Catálogo de juegos PlayStation |
| `players.csv` | 356.600 | Jugadores registrados con país |
| `prices.csv` | 62.816 | Precios en USD, EUR, GBP, JPY y RUB |

---

## Estructura del repositorio

```
gaming_shiny/
├── app.R              # Código completo de la app (ui + server)
├── data/
│   ├── games.csv
│   ├── players.csv
│   └── prices.csv
├── assets/
│   └── preview.png    # Captura de pantalla de la app
└── README.md
```

---

## Cómo ejecutar la app localmente

### Requisitos previos

- R (>= 4.0): https://cran.r-project.org
- RStudio: https://posit.co/download/rstudio-desktop/

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/gaming_shiny.git
cd gaming_shiny
```

### Paso 2 — Instalar dependencias

Abre RStudio y ejecuta en la consola:

```r
install.packages(c(
  "shiny",
  "shinydashboard",
  "tidyverse",
  "plotly",
  "DT",
  "scales",
  "lubridate"
))
```

### Paso 3 — Ejecutar la app

Desde la consola de RStudio, estando en la carpeta `gaming_shiny/`:

```r
shiny::runApp()
```

O abre `app.R` en RStudio y haz clic en el botón **▶ Run App**.

### Paso 4 — Abrir en el navegador

La app se abre automáticamente. Si no, accede a:

```
http://127.0.0.1:XXXX
```

(El puerto lo indica RStudio en la consola)

---

## Despliegue en shinyapps.io

### Paso 1 — Crear cuenta

Crea una cuenta gratuita en: https://www.shinyapps.io

### Paso 2 — Configurar las credenciales en RStudio

```r
install.packages("rsconnect")
library(rsconnect)

rsconnect::setAccountInfo(
  name   = "TU_NOMBRE_EN_SHINYAPPS",
  token  = "TU_TOKEN",
  secret = "TU_SECRET"
)
```

> Los valores de `name`, `token` y `secret` los encuentras en
> shinyapps.io → tu cuenta → **Tokens** → **Show**.

### Paso 3 — Desplegar

```r
rsconnect::deployApp(appDir = ".")
```

Después de 2–4 minutos la app estará disponible en:

```
https://TU_NOMBRE.shinyapps.io/gaming_shiny/
```

---

## Tecnologías usadas

- **R** + **RStudio**
- **Shiny** — framework de aplicaciones web
- **shinydashboard** — layout tipo dashboard
- **Plotly** — gráficos interactivos
- **tidyverse** — manipulación de datos
- **DT** — tablas interactivas

---

*Publicado como parte del curso de Visualización de Datos en R.*
