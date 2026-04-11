# ============================================================
# app.R — Gaming Profiles 2025 | PlayStation Dashboard
# Autor: Alex Terán | Ciencia de Datos
# ============================================================

library(shiny)
library(shinydashboard)
library(tidyverse)
library(plotly)
library(DT)
library(scales)
library(lubridate)

# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================

games <- read_csv("data/games.csv", show_col_types = FALSE) %>%
  mutate(
    release_date     = as.Date(release_date),
    year             = year(release_date),
    genres_clean     = str_remove_all(coalesce(genres, ""), "[\\[\\]']"),
    publishers_clean = str_remove_all(coalesce(publishers, ""), "[\\[\\]']"),
    developers_clean = str_remove_all(coalesce(developers, ""), "[\\[\\]']")
  )

players <- read_csv("data/players.csv", show_col_types = FALSE) %>%
  filter(!is.na(country), country != "")

prices <- read_csv("data/prices.csv", show_col_types = FALSE) %>%
  filter(!is.na(usd), usd > 0, usd < 150) %>%
  mutate(date_acquired = as.Date(date_acquired))

games_prices <- games %>% inner_join(prices, by = "gameid")

plataformas_opciones <- c("Todas", sort(unique(na.omit(games$platform))))

# ============================================================
# UI
# ============================================================

ui <- dashboardPage(
  skin = "blue",

  # --- HEADER ---
  dashboardHeader(
    title = span(icon("playstation"), " Gaming PS 2025"),
    titleWidth = 260
  ),

  # --- SIDEBAR ---
  dashboardSidebar(
    width = 260,
    sidebarMenu(
      id = "menu",
      menuItem("Resumen General",  tabName = "resumen",  icon = icon("chart-pie")),
      menuItem("Catálogo de Juegos", tabName = "juegos", icon = icon("gamepad")),
      menuItem("Jugadores",        tabName = "jugadores", icon = icon("users")),
      menuItem("Precios",          tabName = "precios",   icon = icon("dollar-sign")),
      menuItem("Explorador",       tabName = "explorador", icon = icon("search"))
    ),
    hr(),
    tags$div(style = "padding: 0 15px;",
      selectInput("filtro_plataforma", "Plataforma:",
                  choices = plataformas_opciones, selected = "Todas"),
      sliderInput("filtro_año", "Año de lanzamiento:",
                  min = 2010, max = 2024, value = c(2015, 2024), sep = "")
    ),
    hr(),
    tags$div(style = "padding: 10px 15px; font-size: 11px; color: #aaa;",
      "Alex Terán | Ciencia de Datos", br(),
      "Gaming Profiles 2025 — PlayStation"
    )
  ),

  # --- BODY ---
  dashboardBody(

    # CSS personalizado
    tags$head(tags$style(HTML("
      .skin-blue .main-header .logo { background-color: #003087; font-weight: bold; }
      .skin-blue .main-header .navbar { background-color: #003087; }
      .skin-blue .main-sidebar { background-color: #001f5b; }
      .skin-blue .sidebar-menu > li.active > a,
      .skin-blue .sidebar-menu > li:hover > a { background-color: #003087; }
      .value-box-icon { font-size: 45px !important; }
      .small-box h3 { font-size: 32px; }
    "))),

    tabItems(

      # ==========================================
      # TAB 1 — RESUMEN GENERAL
      # ==========================================
      tabItem(tabName = "resumen",

        fluidRow(
          valueBoxOutput("vbox_juegos",    width = 3),
          valueBoxOutput("vbox_jugadores", width = 3),
          valueBoxOutput("vbox_precios",   width = 3),
          valueBoxOutput("vbox_paises",    width = 3)
        ),

        fluidRow(
          box(
            title = "Juegos por plataforma", width = 6,
            solidHeader = TRUE, status = "primary",
            plotlyOutput("resumen_plataforma", height = 280)
          ),
          box(
            title = "Top 8 géneros más populares", width = 6,
            solidHeader = TRUE, status = "primary",
            plotlyOutput("resumen_generos", height = 280)
          )
        ),

        fluidRow(
          box(
            title = "Lanzamientos por año", width = 8,
            solidHeader = TRUE, status = "primary",
            plotlyOutput("resumen_años", height = 260)
          ),
          box(
            title = "Top 5 países de jugadores", width = 4,
            solidHeader = TRUE, status = "primary",
            plotlyOutput("resumen_paises_pie", height = 260)
          )
        )
      ),

      # ==========================================
      # TAB 2 — CATÁLOGO DE JUEGOS
      # ==========================================
      tabItem(tabName = "juegos",

        fluidRow(
          valueBoxOutput("juegos_total",    width = 4),
          valueBoxOutput("juegos_plat",     width = 4),
          valueBoxOutput("juegos_generos",  width = 4)
        ),

        fluidRow(
          box(
            title = "Top 10 géneros", width = 6,
            solidHeader = TRUE, status = "primary",
            plotlyOutput("juegos_plot_generos", height = 320)
          ),
          box(
            title = "Top 10 publishers", width = 6,
            solidHeader = TRUE, status = "primary",
            plotlyOutput("juegos_plot_publishers", height = 320)
          )
        ),

        fluidRow(
          box(
            title = "Tabla de juegos (filtrada)", width = 12,
            solidHeader = TRUE, status = "primary",
            DTOutput("juegos_tabla")
          )
        )
      ),

      # ==========================================
      # TAB 3 — JUGADORES
      # ==========================================
      tabItem(tabName = "jugadores",

        fluidRow(
          valueBoxOutput("jug_total",   width = 4),
          valueBoxOutput("jug_paises",  width = 4),
          valueBoxOutput("jug_top",     width = 4)
        ),

        fluidRow(
          box(
            title = "Top 15 países con más jugadores", width = 8,
            solidHeader = TRUE, status = "primary",
            plotlyOutput("jug_barras", height = 380)
          ),
          box(
            title = "Top 6 países (proporción)", width = 4,
            solidHeader = TRUE, status = "primary",
            plotlyOutput("jug_pie", height = 380)
          )
        )
      ),

      # ==========================================
      # TAB 4 — PRECIOS
      # ==========================================
      tabItem(tabName = "precios",

        fluidRow(
          valueBoxOutput("precio_promedio", width = 3),
          valueBoxOutput("precio_mediana",  width = 3),
          valueBoxOutput("precio_min",      width = 3),
          valueBoxOutput("precio_max",      width = 3)
        ),

        fluidRow(
          box(
            title = "Distribución de precios (USD)", width = 7,
            solidHeader = TRUE, status = "primary",
            plotlyOutput("precio_hist", height = 300)
          ),
          box(
            title = "Rangos de precio", width = 5,
            solidHeader = TRUE, status = "primary",
            plotlyOutput("precio_rangos", height = 300)
          )
        ),

        fluidRow(
          box(
            title = "Precios por moneda (Boxplot)", width = 6,
            solidHeader = TRUE, status = "primary",
            plotlyOutput("precio_boxplot", height = 280)
          ),
          box(
            title = "Top géneros por precio promedio", width = 6,
            solidHeader = TRUE, status = "primary",
            plotlyOutput("precio_genero", height = 280)
          )
        )
      ),

      # ==========================================
      # TAB 5 — EXPLORADOR INTERACTIVO
      # ==========================================
      tabItem(tabName = "explorador",

        fluidRow(
          box(
            title = "Filtros", width = 12,
            solidHeader = TRUE, status = "primary",
            collapsible = TRUE,
            fluidRow(
              column(4, sliderInput("exp_precio", "Rango de precio (USD):",
                                   min = 0, max = 100, value = c(0, 60))),
              column(4, selectInput("exp_genero", "Género:",
                                   choices = NULL, selected = NULL)),
              column(4, selectInput("exp_plataforma2", "Plataforma:",
                                   choices = plataformas_opciones,
                                   selected = "Todas"))
            )
          )
        ),

        fluidRow(
          box(
            title = "USD vs EUR por juego", width = 6,
            solidHeader = TRUE, status = "primary",
            plotlyOutput("exp_scatter", height = 340)
          ),
          box(
            title = "Precio promedio por género (filtrado)", width = 6,
            solidHeader = TRUE, status = "primary",
            plotlyOutput("exp_precio_genero", height = 340)
          )
        ),

        fluidRow(
          box(
            title = "Resultados de la búsqueda", width = 12,
            solidHeader = TRUE, status = "primary",
            DTOutput("exp_tabla")
          )
        )
      )
    )
  )
)

# ============================================================
# SERVER
# ============================================================

server <- function(input, output, session) {

  # ---- Datos reactivos filtrados ----
  games_r <- reactive({
    df <- games
    if (input$filtro_plataforma != "Todas")
      df <- df %>% filter(platform == input$filtro_plataforma)
    df %>% filter(!is.na(year),
                  year >= input$filtro_año[1],
                  year <= input$filtro_año[2])
  })

  prices_r <- reactive({
    if (input$filtro_plataforma != "Todas") {
      ids <- games %>% filter(platform == input$filtro_plataforma) %>% pull(gameid)
      prices %>% filter(gameid %in% ids)
    } else {
      prices
    }
  })

  # Poblar selector de géneros en explorador
  observe({
    generos <- games %>%
      filter(!is.na(genres_clean), genres_clean != "") %>%
      mutate(genres_clean = str_remove_all(genres_clean, "[\\[\\]']")) %>%
      separate_rows(genres_clean, sep = ",\\s*") %>%
      filter(genres_clean != "") %>%
      pull(genres_clean) %>% unique() %>% sort()
    updateSelectInput(session, "exp_genero",
                      choices = c("Todos", generos), selected = "Todos")
  })

  # ==========================================
  # TAB 1 — RESUMEN
  # ==========================================

  output$vbox_juegos <- renderValueBox({
    valueBox(comma(nrow(games_r())), "Juegos en catálogo",
             icon = icon("gamepad"), color = "blue")
  })
  output$vbox_jugadores <- renderValueBox({
    valueBox(comma(nrow(players)), "Jugadores registrados",
             icon = icon("users"), color = "light-blue")
  })
  output$vbox_precios <- renderValueBox({
    valueBox(paste0("$", round(median(prices_r()$usd, na.rm=TRUE), 2)),
             "Precio mediano (USD)",
             icon = icon("dollar-sign"), color = "navy")
  })
  output$vbox_paises <- renderValueBox({
    valueBox(n_distinct(players$country), "Países representados",
             icon = icon("globe"), color = "blue")
  })

  output$resumen_plataforma <- renderPlotly({
    games_r() %>%
      count(platform, sort = TRUE) %>%
      plot_ly(x = ~n, y = ~reorder(platform, n), type = "bar",
              orientation = "h",
              marker = list(color = "#003087"),
              hovertemplate = "<b>%{y}</b><br>%{x} juegos<extra></extra>") %>%
      layout(xaxis = list(title = ""), yaxis = list(title = ""),
             paper_bgcolor = "white", plot_bgcolor = "white",
             margin = list(l = 5, r = 20, t = 5, b = 30))
  })

  output$resumen_generos <- renderPlotly({
    games_r() %>%
      filter(genres_clean != "") %>%
      separate_rows(genres_clean, sep = ",\\s*") %>%
      filter(genres_clean != "") %>%
      count(genres_clean, sort = TRUE) %>%
      slice_head(n = 8) %>%
      plot_ly(x = ~n, y = ~reorder(genres_clean, n), type = "bar",
              orientation = "h",
              marker = list(color = "#0070CC"),
              hovertemplate = "<b>%{y}</b><br>%{x} juegos<extra></extra>") %>%
      layout(xaxis = list(title = ""), yaxis = list(title = ""),
             paper_bgcolor = "white", plot_bgcolor = "white",
             margin = list(l = 5, r = 20, t = 5, b = 30))
  })

  output$resumen_años <- renderPlotly({
    games_r() %>%
      count(year, platform) %>%
      filter(!is.na(year)) %>%
      plot_ly(x = ~year, y = ~n, color = ~platform,
              colors = c("#001f5b","#003087","#0070CC","#93C6E0"),
              type = "bar",
              hovertemplate = "Año: %{x}<br>Juegos: %{y}<extra></extra>") %>%
      layout(barmode = "stack",
             xaxis = list(title = "Año"),
             yaxis = list(title = "Juegos lanzados"),
             paper_bgcolor = "white", plot_bgcolor = "white",
             legend = list(orientation = "h", y = -0.25))
  })

  output$resumen_paises_pie <- renderPlotly({
    top5 <- players %>% count(country, sort = TRUE) %>% slice_head(n = 5)
    otros <- tibble(country = "Otros", n = nrow(players) - sum(top5$n))
    bind_rows(top5, otros) %>%
      plot_ly(labels = ~country, values = ~n, type = "pie",
              marker = list(colors = c("#001f5b","#003087","#0070CC",
                                       "#4D9AC2","#93C6E0","#d0e8f5")),
              textinfo = "label+percent",
              hovertemplate = "<b>%{label}</b><br>%{value:,}<extra></extra>") %>%
      layout(showlegend = FALSE, paper_bgcolor = "white",
             margin = list(t = 10, b = 10))
  })

  # ==========================================
  # TAB 2 — JUEGOS
  # ==========================================

  output$juegos_total <- renderValueBox({
    valueBox(comma(nrow(games_r())), "Juegos",
             icon = icon("gamepad"), color = "blue")
  })
  output$juegos_plat <- renderValueBox({
    valueBox(n_distinct(games_r()$platform), "Plataformas",
             icon = icon("desktop"), color = "light-blue")
  })
  output$juegos_generos <- renderValueBox({
    n <- games_r() %>%
      filter(genres_clean != "") %>%
      separate_rows(genres_clean, sep = ",\\s*") %>%
      filter(genres_clean != "") %>%
      pull(genres_clean) %>% n_distinct()
    valueBox(n, "Géneros únicos", icon = icon("tags"), color = "navy")
  })

  output$juegos_plot_generos <- renderPlotly({
    games_r() %>%
      filter(genres_clean != "") %>%
      separate_rows(genres_clean, sep = ",\\s*") %>%
      filter(genres_clean != "") %>%
      count(genres_clean, sort = TRUE) %>%
      slice_head(n = 10) %>%
      plot_ly(x = ~n, y = ~reorder(genres_clean, n), type = "bar",
              orientation = "h",
              marker = list(color = ~n,
                            colorscale = list(c(0,"#93C6E0"), c(1,"#003087")),
                            showscale = FALSE),
              hovertemplate = "<b>%{y}</b><br>%{x} juegos<extra></extra>") %>%
      layout(xaxis = list(title = "Juegos"),
             yaxis = list(title = ""),
             paper_bgcolor = "white", plot_bgcolor = "white")
  })

  output$juegos_plot_publishers <- renderPlotly({
    games_r() %>%
      filter(publishers_clean != "") %>%
      separate_rows(publishers_clean, sep = ",\\s*") %>%
      filter(publishers_clean != "") %>%
      count(publishers_clean, sort = TRUE) %>%
      slice_head(n = 10) %>%
      plot_ly(x = ~n, y = ~reorder(publishers_clean, n), type = "bar",
              orientation = "h",
              marker = list(color = "#0070CC"),
              hovertemplate = "<b>%{y}</b><br>%{x} juegos<extra></extra>") %>%
      layout(xaxis = list(title = "Juegos"),
             yaxis = list(title = ""),
             paper_bgcolor = "white", plot_bgcolor = "white")
  })

  output$juegos_tabla <- renderDT({
    games_r() %>%
      select(gameid, title, platform, year,
             genres_clean, publishers_clean) %>%
      rename(ID = gameid, Título = title, Plataforma = platform,
             Año = year, Géneros = genres_clean,
             Publisher = publishers_clean) %>%
      datatable(options = list(pageLength = 8, scrollX = TRUE,
                               language = list(url = '//cdn.datatables.net/plug-ins/1.10.11/i18n/Spanish.json')),
                rownames = FALSE)
  })

  # ==========================================
  # TAB 3 — JUGADORES
  # ==========================================

  output$jug_total <- renderValueBox({
    valueBox(comma(nrow(players)), "Jugadores registrados",
             icon = icon("users"), color = "blue")
  })
  output$jug_paises <- renderValueBox({
    valueBox(n_distinct(players$country), "Países",
             icon = icon("globe"), color = "light-blue")
  })
  output$jug_top <- renderValueBox({
    top <- players %>% count(country, sort = TRUE) %>% slice_head(n = 1)
    valueBox(top$country,
             paste0("País líder (", comma(top$n), " jugadores)"),
             icon = icon("flag"), color = "navy")
  })

  output$jug_barras <- renderPlotly({
    players %>%
      count(country, sort = TRUE) %>%
      slice_head(n = 15) %>%
      mutate(pct = round(n / nrow(players) * 100, 1)) %>%
      plot_ly(x = ~n, y = ~reorder(country, n), type = "bar",
              orientation = "h",
              marker = list(color = "#003087"),
              text  = ~paste0(comma(n), " (", pct, "%)"),
              hovertemplate = "<b>%{y}</b><br>%{text}<extra></extra>") %>%
      layout(xaxis = list(title = "Jugadores"),
             yaxis = list(title = ""),
             paper_bgcolor = "white", plot_bgcolor = "white")
  })

  output$jug_pie <- renderPlotly({
    top6 <- players %>% count(country, sort = TRUE) %>% slice_head(n = 6)
    otros <- tibble(country = "Otros", n = nrow(players) - sum(top6$n))
    bind_rows(top6, otros) %>%
      plot_ly(labels = ~country, values = ~n, type = "pie",
              marker = list(colors = c("#001f5b","#003087","#0055b3",
                                       "#0070CC","#4D9AC2","#93C6E0","#d0e8f5")),
              textinfo = "label+percent",
              hovertemplate = "<b>%{label}</b><br>%{value:,} jugadores<extra></extra>") %>%
      layout(showlegend = FALSE, paper_bgcolor = "white")
  })

  # ==========================================
  # TAB 4 — PRECIOS
  # ==========================================

  output$precio_promedio <- renderValueBox({
    valueBox(paste0("$", round(mean(prices_r()$usd, na.rm=TRUE), 2)),
             "Precio promedio", icon = icon("chart-bar"), color = "blue")
  })
  output$precio_mediana <- renderValueBox({
    valueBox(paste0("$", round(median(prices_r()$usd, na.rm=TRUE), 2)),
             "Precio mediana", icon = icon("dollar-sign"), color = "light-blue")
  })
  output$precio_min <- renderValueBox({
    valueBox(paste0("$", min(prices_r()$usd, na.rm=TRUE)),
             "Precio mínimo", icon = icon("arrow-down"), color = "navy")
  })
  output$precio_max <- renderValueBox({
    valueBox(paste0("$", max(prices_r()$usd, na.rm=TRUE)),
             "Precio máximo", icon = icon("arrow-up"), color = "blue")
  })

  output$precio_hist <- renderPlotly({
    prices_r() %>% filter(usd <= 70) %>%
      plot_ly(x = ~usd, type = "histogram", nbinsx = 25,
              marker = list(color = "#003087",
                            line = list(color = "white", width = 0.8)),
              hovertemplate = "Precio: $%{x}<br>Cantidad: %{y}<extra></extra>") %>%
      layout(xaxis = list(title = "Precio (USD)"),
             yaxis = list(title = "Frecuencia"),
             paper_bgcolor = "white", plot_bgcolor = "white")
  })

  output$precio_rangos <- renderPlotly({
    prices_r() %>%
      mutate(rango = case_when(
        usd <= 5  ~ "≤ $5",
        usd <= 20 ~ "$5–$20",
        usd <= 40 ~ "$20–$40",
        usd <= 70 ~ "$40–$70",
        TRUE      ~ "> $70"
      )) %>%
      count(rango) %>%
      mutate(pct  = round(n / sum(n) * 100, 1),
             rango = factor(rango, levels = c("≤ $5","$5–$20","$20–$40",
                                              "$40–$70","> $70"))) %>%
      plot_ly(x = ~rango, y = ~pct, type = "bar",
              marker = list(color = c("#001f5b","#003087","#0070CC",
                                      "#4D9AC2","#93C6E0")),
              text  = ~paste0(pct, "%"),
              textposition = "outside",
              hovertemplate = "%{x}<br>%{y:.1f}%<extra></extra>") %>%
      layout(yaxis = list(title = "%", ticksuffix = "%"),
             xaxis = list(title = "Rango de precio"),
             paper_bgcolor = "white", plot_bgcolor = "white",
             showlegend = FALSE)
  })

  output$precio_boxplot <- renderPlotly({
    prices_r() %>%
      select(usd, eur, gbp) %>%
      filter(if_all(everything(), ~!is.na(.) & . > 0 & . < 80)) %>%
      pivot_longer(everything(), names_to = "Moneda", values_to = "Precio") %>%
      mutate(Moneda = toupper(Moneda)) %>%
      plot_ly(x = ~Moneda, y = ~Precio, type = "box", color = ~Moneda,
              colors = c("#003087","#0070CC","#93C6E0"),
              hovertemplate = "%{x}<br>%{y:.2f}<extra></extra>") %>%
      layout(showlegend = FALSE,
             xaxis = list(title = "Moneda"),
             yaxis = list(title = "Precio"),
             paper_bgcolor = "white", plot_bgcolor = "white")
  })

  output$precio_genero <- renderPlotly({
    games_prices %>%
      filter(genres_clean != "") %>%
      separate_rows(genres_clean, sep = ",\\s*") %>%
      filter(genres_clean != "") %>%
      group_by(genres_clean) %>%
      summarise(precio_prom = mean(usd, na.rm = TRUE),
                n = n(), .groups = "drop") %>%
      filter(n >= 15) %>%
      slice_max(precio_prom, n = 8) %>%
      plot_ly(x = ~precio_prom,
              y = ~reorder(genres_clean, precio_prom),
              type = "bar", orientation = "h",
              marker = list(color = "#003087"),
              hovertemplate = "<b>%{y}</b><br>$%{x:.2f}<extra></extra>") %>%
      layout(xaxis = list(title = "Precio promedio (USD)"),
             yaxis = list(title = ""),
             paper_bgcolor = "white", plot_bgcolor = "white")
  })

  # ==========================================
  # TAB 5 — EXPLORADOR
  # ==========================================

  datos_explorador <- reactive({
    df <- games_prices

    if (input$exp_plataforma2 != "Todas")
      df <- df %>% filter(platform == input$exp_plataforma2)

    if (!is.null(input$exp_genero) && input$exp_genero != "Todos") {
      df <- df %>%
        filter(str_detect(genres_clean,
                          fixed(input$exp_genero, ignore_case = TRUE)))
    }

    df %>% filter(usd >= input$exp_precio[1],
                  usd <= input$exp_precio[2])
  })

  output$exp_scatter <- renderPlotly({
    datos_explorador() %>%
      filter(!is.na(eur), eur > 0, eur < 100) %>%
      plot_ly(x = ~usd, y = ~eur, type = "scatter", mode = "markers",
              color = ~platform,
              colors = c("#001f5b","#003087","#0070CC","#93C6E0"),
              marker = list(size = 5, opacity = 0.5),
              text  = ~paste0("<b>", title, "</b><br>",
                              platform, "<br>USD: $", usd, " | EUR: €", eur),
              hovertemplate = "%{text}<extra></extra>") %>%
      layout(xaxis = list(title = "Precio USD"),
             yaxis = list(title = "Precio EUR"),
             paper_bgcolor = "white", plot_bgcolor = "white",
             legend = list(orientation = "h", y = -0.2))
  })

  output$exp_precio_genero <- renderPlotly({
    datos_explorador() %>%
      filter(genres_clean != "") %>%
      separate_rows(genres_clean, sep = ",\\s*") %>%
      filter(genres_clean != "") %>%
      group_by(genres_clean) %>%
      summarise(precio_prom = mean(usd, na.rm = TRUE),
                n = n(), .groups = "drop") %>%
      filter(n >= 5) %>%
      slice_max(precio_prom, n = 10) %>%
      plot_ly(x = ~precio_prom,
              y = ~reorder(genres_clean, precio_prom),
              type = "bar", orientation = "h",
              marker = list(color = "#0070CC"),
              hovertemplate = "<b>%{y}</b><br>$%{x:.2f}<extra></extra>") %>%
      layout(xaxis = list(title = "Precio prom. (USD)"),
             yaxis = list(title = ""),
             paper_bgcolor = "white", plot_bgcolor = "white")
  })

  output$exp_tabla <- renderDT({
    datos_explorador() %>%
      select(title, platform, year, genres_clean,
             publishers_clean, usd, eur, gbp) %>%
      rename(Título = title, Plataforma = platform, Año = year,
             Géneros = genres_clean, Publisher = publishers_clean,
             USD = usd, EUR = eur, GBP = gbp) %>%
      datatable(options = list(pageLength = 6, scrollX = TRUE,
                               language = list(url = '//cdn.datatables.net/plug-ins/1.10.11/i18n/Spanish.json')),
                rownames = FALSE)
  })
}

# ============================================================
# LANZAR APP
# ============================================================
shinyApp(ui = ui, server = server)
