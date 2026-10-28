"""tabs/asistente.py — Chatbot flotante de atención al cliente y apoyo a decisiones (abajo a la derecha).

No es una pestaña: app.py agrega widget() al layout principal, así el chat está en todas las pestañas y la
conversación no se pierde al cambiar de pestaña.

Funciones:
- Atajos siempre visibles con los usos de la proyección que plantea el proyecto (metas, inventario, alertas y
  presupuesto), más resumen ejecutivo, factores que explican las ventas (EDA) y confiabilidad.
- Sugerencias según la pestaña abierta; el asistente también sabe qué pestaña está viendo el usuario.
- Calificación 👍/👎 de cada respuesta (se guarda en logs/chat_feedback.csv para mejorar el asistente).
- Descargar la conversación (.md) y empezar una nueva.
- Mensajes de error claros (límite gratuito, servicio saturado, clave inválida).

La IA se usa con la librería `openai`, compatible con Gemini, Groq, OpenAI u Ollama. Proveedor, clave y modelo
se leen del archivo .env (OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL). Lo que sabe el asistente está en
tabs/asistente_contexto.py y los estilos en assets/chat.css.
"""
import csv
import os
from datetime import datetime
from functools import lru_cache

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, clientside_callback, ctx, dcc, html, no_update
from openai import (APIConnectionError, APIStatusError, AuthenticationError, OpenAI, PermissionDeniedError,
                    RateLimitError)

from common import ROOT, modelo_listo
from tabs.asistente_contexto import AUTORES, base_conocimiento

MAX_MENSAJES = 10  # solo se envían los últimos mensajes de la conversación, para controlar el consumo
LOG_FEEDBACK = ROOT / "logs" / "chat_feedback.csv"

BIENVENIDA = ("¡Hola! 👋 Soy el **Asistente Superstore**. Traduzco la proyección de ventas 2018 en decisiones: "
              "**metas, inventario, alertas y presupuesto**, y te digo qué tan confiable es cada cifra.\n\n"
              "Usa los atajos de abajo o escríbeme tu pregunta.")

# Atajos: los cuatro usos de la proyección que plantea el proyecto (metas, inventario, alerta temprana y
# presupuesto), más el resumen para gerencia, la Pregunta 1 del EDA y la confiabilidad.
# (etiqueta del botón, pregunta que se envía)
ACCIONES = [
    ("📋 Resumen ejecutivo", "Dame un resumen ejecutivo para gerencia: cómo vamos y las 3 decisiones más "
                            "importantes para 2018."),
    ("🎯 Metas", "Propón metas de ventas 2018 por región, con un rango según el error del modelo."),
    ("📦 Inventario", "¿Cuándo y en qué categorías conviene reforzar inventario en 2018?"),
    ("⚠️ Alertas", "¿Qué categorías y regiones necesitan atención en 2018: caídas o menor crecimiento "
                   "proyectado, o cifras poco confiables?"),
    ("💵 Presupuesto", "Arma el presupuesto de ingresos 2018 por trimestre, con un rango mínimo y máximo."),
    ("🔎 ¿Qué explica las ventas?", "¿Qué factores explican el nivel de ventas: categoría, región, segmento y "
                                   "época del año?"),
    ("🔍 Confianza", "¿Qué tan confiable es la proyección y en qué cifras debo tener más cuidado?"),
]

# Pestañas del tablero: id -> (nombre, sugerencias para esa pestaña)
PESTANAS = {
    "introduccion": ("Introducción", ["¿Qué decisiones puedo tomar con este tablero?",
                                      "¿Cuánto se proyecta vender en 2018?"]),
    "problema": ("Problema y objetivos", ["¿Cómo ayuda la proyección a planificar metas e inventario?",
                                          "¿Cuánto se proyecta vender en 2018?"]),
    "marco": ("Marco teórico", ["¿Qué variables pesan más en el modelo?",
                                "¿Qué modelos se compararon?"]),
    "metodologia": ("Metodología", ["¿Cómo se comprobó que el modelo funciona?",
                                    "¿Qué limitaciones tiene el análisis?"]),
    "etl": ("ETL", ["¿Cómo crecieron las ventas de 2014 a 2017?",
                    "¿Qué categoría vendió más en 2017?"]),
    "eda": ("EDA", ["¿En qué meses se concentran las ventas?",
                    "¿Qué segmento de cliente compra más y cómo ha cambiado?"]),
    "modelos": ("Modelos y resultados", ["¿Por qué se eligió Random Forest?",
                                         "¿Qué modelo quedó en segundo lugar?"]),
    "proyeccion": ("Proyección 2018", ["¿Qué región crece más en 2018?",
                                       "¿Cuánto se proyecta para Technology en West en diciembre?"]),
    "prediccion": ("Predicción", ["¿Cómo interpreto el índice de la predicción?",
                                  "¿Cuánto vendió Furniture en East en noviembre de 2017?"]),
    "conclusiones": ("Conclusiones", ["¿Cuáles son las 3 conclusiones principales?",
                                      "¿Qué limitaciones tiene el análisis?"]),
}


# ---------------------------------------------------------------- IA
@lru_cache(maxsize=1)
def cliente():
    # Lee OPENAI_API_KEY y OPENAI_BASE_URL del entorno. Los reintentos absorben cortes pasajeros (p. ej. error 503).
    return OpenAI(timeout=60, max_retries=2)


def modelo_ia() -> str:
    return os.getenv("OPENAI_MODEL", "gemini-3.5-flash")


def _ahora() -> str:
    return datetime.now().strftime("%H:%M")


def _mensaje_error(e: Exception) -> str:
    """Traduce el error técnico a un mensaje de atención al cliente."""
    if isinstance(e, RateLimitError):
        return "Se alcanzó el límite de consultas gratuitas por minuto. Espera un momento y vuelve a intentarlo. 🙏"
    if isinstance(e, (AuthenticationError, PermissionDeniedError)):
        return f"Hay un problema con la clave de la IA (archivo .env). Avísale al equipo del proyecto: {AUTORES}."
    if isinstance(e, APIConnectionError):
        return "No pude conectarme con el servicio de IA (sin internet o tardó demasiado). Intenta de nuevo."
    if isinstance(e, APIStatusError) and e.status_code >= 500:
        return "El servicio de IA está saturado en este momento. Intenta de nuevo en unos segundos."
    return f"Lo siento, no pude responder en este momento. Intenta de nuevo.\n\n_Detalle: {str(e)[:160]}_"


def consultar_ia(historial: list, pestana: str) -> dict:
    """Envía la conversación al modelo y devuelve el mensaje de respuesta (o uno de error)."""
    nombre = PESTANAS.get(pestana, ("el tablero",))[0]
    sistema = base_conocimiento() + f"\n\nCONTEXTO DE LA SESIÓN: el usuario está viendo la pestaña «{nombre}»."
    conversacion = [{"role": m["role"], "content": m["content"]} for m in historial if not m.get("error")]
    try:
        r = cliente().chat.completions.create(
            model=modelo_ia(),
            messages=[{"role": "system", "content": sistema}] + conversacion[-MAX_MENSAJES:],
        )
        texto = (r.choices[0].message.content or "").strip() or "No tengo una respuesta para eso. ¿Lo reformulas?"
        return {"role": "assistant", "content": texto, "t": _ahora()}
    except Exception as e:  # noqa: BLE001 — cualquier fallo se muestra como mensaje, la app sigue funcionando
        return {"role": "assistant", "content": _mensaje_error(e), "t": _ahora(), "error": True}


def _guardar_feedback(historial: list, i: int, voto: str):
    """Agrega una fila a logs/chat_feedback.csv con la pregunta, la respuesta y el voto."""
    pregunta = next((m["content"] for m in reversed(historial[:i]) if m["role"] == "user"), "")
    try:
        LOG_FEEDBACK.parent.mkdir(exist_ok=True)
        nuevo = not LOG_FEEDBACK.exists()
        with LOG_FEEDBACK.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if nuevo:
                w.writerow(["fecha", "voto", "pregunta", "respuesta", "modelo"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), voto, pregunta,
                        historial[i]["content"], modelo_ia()])
    except OSError:
        pass  # el registro es opcional: si falla, el chat sigue funcionando


# ---------------------------------------------------------------- interfaz
def widget():
    """Botón flotante + ventana de chat. Se agrega una sola vez en app.layout."""
    if not modelo_listo():
        return html.Div()
    return html.Div([
        dcc.Store(id="chat-historial", data=[], storage_type="session"),  # sobrevive a recargar la página
        dcc.Download(id="chat-descarga"),
        html.Div([
            html.Div([
                html.Div("🤖", className="chat-avatar"),
                html.Div([html.Div("Asistente Superstore", className="chat-title"),
                          html.Div("En línea · apoyo a decisiones con IA", className="chat-status")]),
                html.Div([
                    html.Button("⤓", id="chat-exportar", className="chat-icon-btn", n_clicks=0,
                                title="Descargar conversación", **{"aria-label": "Descargar conversación"}),
                    html.Button("↺", id="chat-reiniciar", className="chat-icon-btn", n_clicks=0,
                                title="Nueva conversación", **{"aria-label": "Nueva conversación"}),
                    html.Button("×", id="chat-cerrar", className="chat-icon-btn chat-close", n_clicks=0,
                                title="Cerrar", **{"aria-label": "Cerrar chat"}),
                ], className="chat-head-actions"),
            ], className="chat-head"),
            html.Div(id="chat-mensajes", className="chat-body", **{"aria-live": "polite"}),
            html.Div([html.Button(etiqueta, id={"type": "chat-sug", "g": "accion", "i": i}, className="chat-action",
                                  title=pregunta) for i, (etiqueta, pregunta) in enumerate(ACCIONES)],
                     className="chat-actions"),
            html.Div([
                dbc.Input(id="chat-input", placeholder="Escribe tu pregunta…", autocomplete="off",
                          className="chat-input", maxLength=500),
                html.Button("➤", id="chat-enviar", className="chat-send", n_clicks=0, title="Enviar",
                            **{"aria-label": "Enviar"}),
            ], className="chat-foot"),
            html.Div("Respuestas generadas por IA con los datos del proyecto; pueden contener errores.",
                     className="chat-note"),
        ], id="chat-panel", className="chat-panel", role="dialog", **{"aria-label": "Asistente Superstore"}),
        html.Div("💡 ¿Te ayudo a decidir? Pregúntame por metas, inventario o alertas.",
                 id="chat-teaser", className="chat-teaser", n_clicks=0),
        html.Button([html.Span("💬", id="chat-icono"), html.Span(className="chat-launcher-dot")],
                    id="chat-abrir", className="chat-launcher", n_clicks=0, title="Asistente Superstore",
                    **{"aria-label": "Abrir asistente"}),
    ])


def _fila_usuario(m):
    return html.Div([html.Div(dcc.Markdown(m["content"]), className="chat-msg user"),
                     html.Div(m.get("t", ""), className="chat-meta")], className="chat-fila user")


def _fila_bot(m, i=None):
    meta = [html.Span(m.get("t", ""))]
    if i is not None and not m.get("error"):
        for voto, icono, titulo in (("up", "👍", "Respuesta útil"), ("down", "👎", "Respuesta no útil")):
            meta.append(html.Button(icono, id={"type": "chat-fb", "i": i, "v": voto}, title=titulo,
                                    className="chat-fb" + (" activo" if m.get("fb") == voto else "")))
    clase = "chat-msg bot" + (" error" if m.get("error") else "")
    return html.Div([html.Div(dcc.Markdown(m["content"]), className=clase),
                     html.Div(meta, className="chat-meta")], className="chat-fila bot")


@callback(Output("chat-mensajes", "children"), Input("chat-historial", "data"), Input("pestana-activa", "data"))
def mostrar(historial, pestana):
    """Dibuja la conversación. El contenedor usa column-reverse, así que la lista va del más nuevo al más viejo
    y el chat queda siempre pegado abajo, en el último mensaje."""
    historial = historial or []
    items = [_fila_bot({"role": "assistant", "content": BIENVENIDA})]
    if not historial:
        sugerencias = PESTANAS.get(pestana, ("", []))[1]
        items.append(html.Div(
            [html.Div("Sugerencias para esta pestaña", className="chat-chips-title")]
            + [html.Button(s, id={"type": "chat-sug", "g": pestana, "i": i}, className="chat-chip")
               for i, s in enumerate(sugerencias)], className="chat-chips"))
    for i, m in enumerate(historial):
        items.append(_fila_usuario(m) if m["role"] == "user" else _fila_bot(m, i))
    if historial and historial[-1]["role"] == "user":
        items.append(html.Div(html.Div([html.Span(), html.Span(), html.Span()], className="chat-msg bot chat-typing"),
                              className="chat-fila bot", title="Escribiendo…"))
    return items[::-1]


@callback(Output("chat-historial", "data"), Output("chat-input", "value"),
          Input("chat-enviar", "n_clicks"), Input("chat-input", "n_submit"),
          Input({"type": "chat-sug", "g": ALL, "i": ALL}, "n_clicks"),
          State("chat-input", "value"), State("chat-historial", "data"), prevent_initial_call=True)
def enviar(_clicks, _enter, _sugerencias, texto, historial):
    """Agrega la pregunta del usuario. La respuesta la genera responder(), que se dispara a continuación."""
    historial = historial or []
    if historial and historial[-1]["role"] == "user":  # aún esperando la respuesta anterior
        return no_update, no_update
    origen = ctx.triggered_id
    if isinstance(origen, dict):  # clic en un atajo o en una sugerencia
        if not ctx.triggered[0]["value"]:  # el botón apenas se dibujó, no hubo clic
            return no_update, no_update
        if origen["g"] == "accion":
            texto, limpiar = ACCIONES[origen["i"]][1], no_update  # conserva lo que el usuario estaba escribiendo
        else:
            texto, limpiar = PESTANAS[origen["g"]][1][origen["i"]], no_update
    else:
        limpiar = ""
    if not texto or not texto.strip():
        return no_update, no_update
    return historial + [{"role": "user", "content": texto.strip(), "t": _ahora()}], limpiar


@callback(Output("chat-historial", "data", allow_duplicate=True), Input("chat-historial", "data"),
          State("pestana-activa", "data"), prevent_initial_call=True)
def responder(historial, pestana):
    """Si el último mensaje es del usuario, consulta al modelo y agrega la respuesta."""
    if not historial or historial[-1]["role"] != "user":
        return no_update
    return historial + [consultar_ia(historial, pestana)]


@callback(Output("chat-historial", "data", allow_duplicate=True),
          Input({"type": "chat-fb", "i": ALL, "v": ALL}, "n_clicks"), State("chat-historial", "data"),
          prevent_initial_call=True)
def calificar(_clicks, historial):
    """Marca la respuesta como útil / no útil (un segundo clic quita la marca) y la registra."""
    origen = ctx.triggered_id
    if not isinstance(origen, dict) or not ctx.triggered[0]["value"] or not historial:
        return no_update
    i, voto = origen["i"], origen["v"]
    if i >= len(historial) or historial[i]["role"] != "assistant":
        return no_update
    historial = [dict(m) for m in historial]
    if historial[i].get("fb") == voto:
        historial[i].pop("fb")
    else:
        historial[i]["fb"] = voto
        _guardar_feedback(historial, i, voto)
    return historial


@callback(Output("chat-historial", "data", allow_duplicate=True), Input("chat-reiniciar", "n_clicks"),
          prevent_initial_call=True)
def reiniciar(_clicks):
    return []


@callback(Output("chat-descarga", "data"), Input("chat-exportar", "n_clicks"), State("chat-historial", "data"),
          prevent_initial_call=True)
def exportar(_clicks, historial):
    """Descarga la conversación como Markdown, útil para dejar registro de lo que se decidió."""
    if not historial:
        return no_update
    ahora = datetime.now()
    partes = [f"# Conversación con el Asistente Superstore\n\nExportada el {ahora:%Y-%m-%d %H:%M}.\n"]
    for m in historial:
        quien = "Tú" if m["role"] == "user" else "Asistente"
        partes.append(f"**{quien}** ({m.get('t', '')}):\n\n{m['content']}\n")
    partes.append("---\n_Respuestas generadas por IA a partir de los resultados del proyecto. Las cifras de 2018 "
                  "son proyecciones, no ventas reales._\n")
    return dcc.send_string("\n".join(partes), f"conversacion_asistente_{ahora:%Y%m%d_%H%M}.md")


# Abrir y cerrar la ventana se hace en el navegador (sin ir al servidor) para que sea instantáneo.
# El aviso flotante (teaser) se oculta en cuanto el usuario interactúa con el chat.
clientside_callback(
    """
    function(_abrir, _cerrar, _teaser, clase) {
        const quien = (dash_clientside.callback_context.triggered[0] || {}).prop_id || "";
        const abierto = (clase || "").includes("abierto");
        const abrir = quien.startsWith("chat-cerrar") ? false : (quien.startsWith("chat-teaser") ? true : !abierto);
        if (abrir) { setTimeout(() => { const i = document.getElementById("chat-input"); if (i) i.focus(); }, 200); }
        return [abrir ? "chat-panel abierto" : "chat-panel", abrir ? "×" : "💬", "chat-teaser oculto"];
    }
    """,
    Output("chat-panel", "className"), Output("chat-icono", "children"), Output("chat-teaser", "className"),
    Input("chat-abrir", "n_clicks"), Input("chat-cerrar", "n_clicks"), Input("chat-teaser", "n_clicks"),
    State("chat-panel", "className"), prevent_initial_call=True,
)
