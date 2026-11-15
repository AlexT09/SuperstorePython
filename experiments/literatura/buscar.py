"""
experiments/literatura/buscar.py — Búsqueda documentada para la matriz de novedad.

Consulta tres bases abiertas (OpenAlex, Crossref y arXiv) con las mismas cadenas y guarda, por
cadena y base, el número de resultados y los títulos más relevantes (busqueda.json), con la fecha.
Ejecutar:  python experiments/literatura/buscar.py
"""
import datetime
import json
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

CADENAS = [
    '"group seasonal indices" forecasting',
    '"seasonal indices" shrinkage forecasting',
    '"seasonal index" "partial pooling" sales forecasting',
    'hierarchical shrinkage combination weights "seasonal naive" retail sales',
    '"forecast combination" series-specific weights shrinkage seasonal profile',
    '"Superstore" sales forecasting seasonal',
    '"índices estacionales" pronóstico ventas minoristas',
    'cross-learning seasonality pooled seasonal profile short time series retail',
]
N = 8


def leer(url):
    """Descarga una URL; si Python no puede verificar el certificado (pasa en Windows), usa curl."""
    try:
        pedido = urllib.request.Request(url, headers={"User-Agent": "busqueda-bibliografica/1.0"})
        with urllib.request.urlopen(pedido, timeout=60) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return subprocess.run(["curl", "-sL", "-m", "60", "-A", "busqueda-bibliografica/1.0", url],
                              capture_output=True, check=True).stdout.decode("utf-8", "replace")


def openalex(q):
    d = json.loads(leer("https://api.openalex.org/works?per-page=%d&search=%s" % (N, urllib.parse.quote(q))))
    return d["meta"]["count"], [(w.get("title"), w.get("publication_year"), w.get("doi")) for w in d["results"]]


def crossref(q):
    d = json.loads(leer("https://api.crossref.org/works?rows=%d&query=%s" % (N, urllib.parse.quote(q))))["message"]
    return d["total-results"], [((w.get("title") or [""])[0], (w.get("issued", {}).get("date-parts") or [[None]])[0][0],
                                 w.get("DOI")) for w in d["items"]]


def arxiv(q):
    x = leer("https://export.arxiv.org/api/query?max_results=%d&search_query=all:%s" % (N, urllib.parse.quote(q)))
    total = int(re.search(r"<opensearch:totalResults[^>]*>(\d+)<", x).group(1))
    items = []
    for e in x.split("<entry>")[1:]:                      # cada <entry> es un artículo
        dato = lambda etiqueta: re.search(r"<%s>(.*?)</%s>" % (etiqueta, etiqueta), e, re.S).group(1)  # noqa: E731
        items.append((re.sub(r"\s+", " ", dato("title")), dato("published")[:4], dato("id")))
    return total, items


salida = {"fecha": datetime.date.today().isoformat(), "consultas": []}
for q in CADENAS:
    for base, f in [("OpenAlex", openalex), ("Crossref", crossref), ("arXiv", arxiv)]:
        try:
            total, items = f(q)
        except Exception as e:                                          # se deja constancia del fallo
            total, items = None, [("ERROR: %s" % e, None, None)]
        salida["consultas"].append({"base": base, "cadena": q, "resultados": total, "primeros": items})
        print(base, "|", q, "|", total)
Path(__file__).with_name("busqueda.json").write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf-8")
