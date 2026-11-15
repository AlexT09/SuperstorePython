"""
experiments/analizar.py — Tablas, pruebas de hipótesis y veredicto a partir de las predicciones guardadas.

Lee experiments/resultados/pred_<conjunto>.csv y tabla_maestra.csv (salidas de ejecutar.py) y aplica
exactamente lo fijado en experiments/protocolo.md. No entrena nada: se puede repetir en segundos.

Salidas en experiments/resultados/:
    lideres.csv            todas las combinaciones de todos los conjuntos (media ± sd, IC, tiempos)
    mejores.csv            la mejor combinación de cada modelo base por conjunto + original + referencias
    pruebas.csv            original frente a las 10 referencias (DM, GW, Wilcoxon, Holm, IC, efecto)
    mcs.csv                p-valor del Model Confidence Set de cada combinación
    entre_conjuntos.csv    rangos por conjunto; Friedman–Iman–Davenport y Wilcoxon–Holm en numeros.json
    ablacion.csv · sensibilidad_lam.csv · optimizadores.csv · anytime.csv · conformal.csv
    residuos.csv · por_serie.csv · por_mes.csv · por_tercil.csv · veredicto.json · numeros.json

Ejecutar desde la raíz:  python experiments/analizar.py
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

import config as C
from model.features import FEATURES, TARGET, agregar_variables
from src import estadistica as E

META = ["Category", "Region", "Mes", "Sales", "prom_12m"]
REFS_FIJAS = [f"{C.INGENUO}|-", f"{C.THETA}|-", f"{C.HGB}|bayes"]
ORIG = f"{C.ORIGINAL}|bayes"


def mensual(p, col):
    """MAE de cada mes evaluado (promedio sobre las series): la unidad de todas las pruebas."""
    return p.assign(e=(p.Sales - p[col]).abs()).groupby("Mes").e.mean().to_numpy()


def escala_mase(conjunto, primer_origen):
    panel = pd.read_csv(C.CONJUNTOS[conjunto], parse_dates=["Mes"])
    d = agregar_variables(panel).dropna(subset=FEATURES + [TARGET])
    d = d[d.Mes < primer_origen]
    return float((d.Sales - d.lag_12).abs().mean())


def metricas(p, col, escala, rng):
    e = p.Sales.to_numpy() - p[col].to_numpy()
    m = mensual(p, col)
    lo, hi = E.ic_bootstrap(m, rng, C.N_BOOT, C.BLOQUE)
    mitad = np.sort(p.Mes.unique())[C.N_ORIGENES // 2]
    return {"MAE": np.abs(e).mean(), "MAE_sd": m.std(ddof=1), "MAE_ic_lo": lo, "MAE_ic_hi": hi,
            "RMSE": np.sqrt((e ** 2).mean()), "MASE": np.abs(e).mean() / escala,
            "WAPE": np.abs(e).sum() / p.Sales.sum() * 100, "desvio_total_pct": (p[col].sum() / p.Sales.sum() - 1) * 100,
            "MAE_primera_mitad": np.abs(e[(p.Mes < mitad).to_numpy()]).mean(),
            "MAE_segunda_mitad": np.abs(e[(p.Mes >= mitad).to_numpy()]).mean()}


def conformal(p, col, niveles=(0.8, 0.9)):
    """Intervalos conformes sobre el índice, calibrados solo con meses evaluados anteriores."""
    meses = np.sort(p.Mes.unique())
    r = ((p.Sales - p[col]).abs() / p.prom_12m).to_numpy()          # error absoluto del índice
    filas = []
    for nivel in niveles:
        cubre, ancho = [], []
        for m in meses[C.N_ORIGENES // 2:]:
            cal, act = r[(p.Mes < m).to_numpy()], (p.Mes == m).to_numpy()
            q = np.quantile(cal, min(1.0, nivel * (1 + 1 / cal.size)))
            radio = q * p.prom_12m.to_numpy()[act]
            cubre += list(np.abs(p.Sales.to_numpy()[act] - p[col].to_numpy()[act]) <= radio)
            ancho += list(2 * radio)
        filas.append({"nivel": nivel, "cobertura": float(np.mean(cubre)), "ancho_medio": float(np.mean(ancho)),
                      "n": len(cubre)})
    return filas


def residuos(p, col):
    """Supuestos de los residuos del bucle externo: White, Ljung–Box, BDS y Shapiro–Wilk."""
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_white
    from statsmodels.tsa.stattools import bds
    e = (p.Sales - p[col]).to_numpy()
    X = np.column_stack([np.ones(e.size), p[col].to_numpy(), p.Mes.dt.month.to_numpy()])
    serie = p.assign(e=e).groupby("Mes").e.sum().to_numpy()         # residuo del total mensual
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            p_bds = float(np.atleast_1d(bds(serie, max_dim=2)[1])[0])
        except Exception:
            p_bds = np.nan
        return {"p_white": float(het_white(e, X)[1]), "p_ljung_box": float(acorr_ljungbox(serie, lags=[6]).lb_pvalue.iloc[0]),
                "p_bds": p_bds, "p_shapiro": float(stats.shapiro(serie).pvalue),
                "sesgo_medio": float(e.mean()), "asimetria": float(stats.skew(e))}


def main():
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(C.SEMILLA)
    maestra = pd.read_csv(C.RES / "tabla_maestra.csv", parse_dates=["origen"])
    costo = maestra.groupby(["conjunto", "modelo", "optimizador"]).agg(
        evaluaciones=("evaluaciones", "mean"), busqueda_s=("busqueda_s", "mean"), ajuste_ms=("ajuste_ms", "median"),
        inferencia_ms=("inferencia_ms", "median"), puntaje_interno=("puntaje_interno", "mean")).reset_index()
    N = {"protocolo": {"tag": "protocolo-v1", "origenes": C.N_ORIGENES, "presupuesto": C.PRESUPUESTO, "semilla": C.SEMILLA}}
    lideres, mejores, pruebas, mcs_filas, abl, sens, conf, resid, rangos_bayes = [], [], [], [], [], [], [], [], {}

    for conjunto in C.CONJUNTOS:
        p = pd.read_csv(C.RES / f"pred_{conjunto}.csv", parse_dates=["Mes"])
        escala = escala_mase(conjunto, p.Mes.min())
        cols = [c for c in p.columns if c not in META and not c.startswith(("abl|", "lam|"))]
        filas = []
        for c in cols:
            modelo, opt = c.split("|")
            filas.append({"conjunto": conjunto, "modelo": modelo, "optimizador": opt, "columna": c, **metricas(p, c, escala, rng)})
        tabla = pd.DataFrame(filas).merge(costo, on=["conjunto", "modelo", "optimizador"], how="left")
        lideres.append(tabla)

        # Mejor combinación de cada modelo base (por MAE del bucle externo) + referencias fijas + original
        base = tabla[tabla.modelo.isin(C.BASE)]
        mejor_base = base.loc[base.groupby("modelo").MAE.idxmin()]
        refs = pd.concat([mejor_base, tabla[tabla.columna.isin(REFS_FIJAS)]])
        orig = tabla[tabla.columna == ORIG]
        mejores.append(pd.concat([orig, refs]).sort_values("MAE"))
        mejor_ref = refs.loc[refs.MAE.idxmin()]

        # Pruebas por pares: original (bayes) frente a las 10 referencias
        m_orig = mensual(p, ORIG)
        comp = pd.DataFrame([{"conjunto": conjunto, "referencia": r.modelo, "optimizador": r.optimizador,
                              "MAE_original": float(orig.MAE.iloc[0]), "MAE_referencia": r.MAE,
                              **E.comparar(m_orig, mensual(p, r.columna), rng)} for r in refs.itertuples()])
        for c in ["p_DM", "p_GW", "p_wilcoxon"]:
            comp[c + "_holm"] = E.holm(comp[c])
        comp["lb_p"] = [float(acorr_ljungbox(m_orig - mensual(p, r.columna), lags=[3]).lb_pvalue.iloc[0])
                        for r in refs.itertuples()]
        pruebas.append(comp)

        # MCS sobre todas las combinaciones
        L = np.column_stack([mensual(p, c) for c in cols])
        pm = E.mcs(L, cols, rng)
        mcs_filas += [{"conjunto": conjunto, "columna": c, "modelo": c.split("|")[0], "optimizador": c.split("|")[1],
                       "MAE": float(L[:, i].mean()), "p_mcs": pm[c], "en_conjunto": pm[c] >= C.ALFA_MCS}
                      for i, c in enumerate(cols)]

        # Ablación (frente al modelo completo) y sensibilidad a lam
        variantes = [c for c in p.columns if c.startswith("abl|")]
        ab = pd.DataFrame([{"conjunto": conjunto, "variante": c[4:], **metricas(p, c, escala, rng),
                            **{k: v for k, v in E.comparar(m_orig, mensual(p, c), rng).items()
                               if k in ("dif", "ic_lo", "ic_hi", "p_DM", "p_GW", "d_cohen")}} for c in variantes])
        ab["p_DM_holm"], ab["p_GW_holm"] = E.holm(ab.p_DM), E.holm(ab.p_GW)
        abl.append(pd.concat([pd.DataFrame([{"conjunto": conjunto, "variante": "Modelo completo", **metricas(p, ORIG, escala, rng)}]), ab]))
        sens += [{"conjunto": conjunto, "lam": float(c[4:]), **{k: v for k, v in metricas(p, c, escala, rng).items()
                                                                 if k in ("MAE", "RMSE", "MASE", "desvio_total_pct")},
                  "MAE_mejor_referencia": mejor_ref.MAE} for c in p.columns if c.startswith("lam|")]

        # Incertidumbre y residuos: original y mejor referencia
        for nombre, c in [("original", ORIG), ("mejor referencia", mejor_ref.columna)]:
            conf += [{"conjunto": conjunto, "modelo": nombre, "columna": c, **f} for f in conformal(p, c)]
        for r in pd.concat([orig, refs]).itertuples():
            resid.append({"conjunto": conjunto, "modelo": r.modelo, "optimizador": r.optimizador, **residuos(p, r.columna)})

        # Para la prueba entre conjuntos: todos con búsqueda bayesiana (o sin ajuste)
        rangos_bayes[conjunto] = {c.split("|")[0]: float(mensual(p, c).mean()) / escala
                                  for c in cols if c.split("|")[1] in ("bayes", "-")}

        N[conjunto] = {"series": int(p.groupby(["Category", "Region"]).ngroups), "pronosticos": int(len(p)),
                       "meses": [str(p.Mes.min().date()), str(p.Mes.max().date())], "escala_mase": escala,
                       "MAE_original": float(orig.MAE.iloc[0]), "mejor_referencia": f"{mejor_ref.modelo} ({mejor_ref.optimizador})",
                       "MAE_mejor_referencia": float(mejor_ref.MAE), "columna_mejor_referencia": mejor_ref.columna,
                       "posicion_original": int((pd.concat([orig, refs]).MAE.rank(method="min"))[orig.index[0]]),
                       "mcs_tamano": int(sum(pm[c] >= C.ALFA_MCS for c in cols)), "mcs_total": len(cols),
                       "original_en_mcs": bool(pm[ORIG] >= C.ALFA_MCS), "p_mcs_original": float(pm[ORIG]),
                       "mejor_ref_en_mcs": bool(pm[mejor_ref.columna] >= C.ALFA_MCS)}

        if conjunto == C.PRINCIPAL:
            t = p[META].assign(pred=p[ORIG], ref=p[mejor_ref.columna])
            t["e"], t["ae"], t["ae_ref"] = t.Sales - t.pred, (t.Sales - t.pred).abs(), (t.Sales - t.ref).abs()
            g = t.groupby(["Category", "Region"]).agg(venta_media=("Sales", "mean"), MAE=("ae", "mean"),
                                                      MAE_ref=("ae_ref", "mean"), sesgo=("e", "mean")).reset_index()
            g.to_csv(C.RES / "por_serie.csv", index=False)
            t.groupby(t.Mes.dt.month).agg(MAE=("ae", "mean"), MAE_ref=("ae_ref", "mean"), sesgo=("e", "mean")) \
                .rename_axis("mes").reset_index().to_csv(C.RES / "por_mes.csv", index=False)
            t["tercil"] = pd.qcut(t.Sales, 3, labels=["bajo", "medio", "alto"])
            t.groupby("tercil", observed=True).agg(venta_media=("Sales", "mean"), MAE=("ae", "mean"), MAE_ref=("ae_ref", "mean"),
                                                   sesgo=("e", "mean")).reset_index().to_csv(C.RES / "por_tercil.csv", index=False)
            N["errores"] = {"series_mejor_que_ref": int((g.MAE < g.MAE_ref).sum()), "series": len(g)}

            # Comparación de los métodos de optimización (las 28 combinaciones + el original)
            opt = tabla[tabla.modelo.isin(C.BASE + [C.ORIGINAL])]
            opt.pivot(index="modelo", columns="optimizador", values="MAE").to_csv(C.RES / "optimizadores_mae.csv")
            resumen = opt.groupby("optimizador").agg(MAE_medio=("MAE", "mean"), evaluaciones=("evaluaciones", "mean"),
                                                     busqueda_s=("busqueda_s", "mean"))
            resumen["veces_mejor"] = opt.loc[opt.groupby("modelo").MAE.idxmin()].optimizador.value_counts()
            resumen.fillna(0).reset_index().to_csv(C.RES / "optimizadores.csv", index=False)
            # Curva de desempeño en cualquier momento: mejor MAE interno hasta la evaluación t (media sobre orígenes)
            ms = maestra[maestra.conjunto == conjunto]
            curvas = []
            for (modelo, o), g2 in ms.groupby(["modelo", "optimizador"]):
                h = [np.minimum.accumulate(json.loads(x)) for x in g2.historial]
                n = min(map(len, h))
                curvas += [{"modelo": modelo, "optimizador": o, "evaluacion": i + 1, "mejor_MAE_interno": v}
                           for i, v in enumerate(np.mean([x[:n] for x in h], axis=0))]
            pd.DataFrame(curvas).to_csv(C.RES / "anytime.csv", index=False)

    lideres, mejores, pruebas = pd.concat(lideres), pd.concat(mejores), pd.concat(pruebas)
    lideres.to_csv(C.RES / "lideres.csv", index=False)
    mejores.to_csv(C.RES / "mejores.csv", index=False)
    pruebas.to_csv(C.RES / "pruebas.csv", index=False)
    pd.DataFrame(mcs_filas).to_csv(C.RES / "mcs.csv", index=False)
    pd.concat(abl).to_csv(C.RES / "ablacion.csv", index=False)
    pd.DataFrame(sens).to_csv(C.RES / "sensibilidad_lam.csv", index=False)
    pd.DataFrame(conf).to_csv(C.RES / "conformal.csv", index=False)
    pd.DataFrame(resid).to_csv(C.RES / "residuos.csv", index=False)

    # Entre conjuntos: Friedman–Iman–Davenport sobre el MASE y Wilcoxon–Holm del original contra cada modelo
    ec = pd.DataFrame(rangos_bayes).T                       # conjuntos x modelos
    fid = E.friedman_iman_davenport(ec.to_numpy())
    ec.to_csv(C.RES / "entre_conjuntos.csv")
    otros = [m for m in ec.columns if m != C.ORIGINAL]
    pw = [float(stats.wilcoxon(ec[C.ORIGINAL], ec[m]).pvalue) for m in otros]
    N["entre_conjuntos"] = {"rango_medio": dict(zip(ec.columns, map(float, fid["rangos"]))), "F": fid["F"], "p": fid["p"],
                            "wilcoxon": dict(zip(otros, pw)), "wilcoxon_holm": dict(zip(otros, map(float, E.holm(pw))))}

    # Veredicto (regla de decisión del protocolo, sección 8)
    ps = pruebas[pruebas.conjunto == C.PRINCIPAL]
    mejor = ps.loc[ps.MAE_referencia.idxmin()]
    n = N[C.PRINCIPAL]
    rechazos = {"MCS": not n["mejor_ref_en_mcs"] and n["original_en_mcs"], "Giacomini-White": bool(mejor.p_GW_holm < C.ALFA),
                "Diebold-Mariano": bool(mejor.p_DM_holm < C.ALFA), "Wilcoxon": bool(mejor.p_wilcoxon_holm < C.ALFA)}
    adicionales = [c for c in C.CONJUNTOS if c != C.PRINCIPAL]
    gana = {c: N[c]["MAE_original"] < N[c]["MAE_mejor_referencia"] for c in adicionales}
    peor = {}
    for c in adicionales:
        pc = pruebas[pruebas.conjunto == c]
        fila = pc.loc[pc.MAE_referencia.idxmin()]
        peor[c] = bool(fila.dif > 0 and fila.p_DM_holm < C.ALFA)
    c1 = bool((ps.dif < 0).all())
    c2 = sum(rechazos.values()) >= 3
    c3 = bool(abs(mejor.d_cohen) >= 0.2 or abs(mejor.delta_cliff) >= 0.147)
    c4 = sum(gana.values()) >= 2 and not any(peor.values())
    veredicto = {"mejor_referencia": f"{mejor.referencia} ({mejor.optimizador})",
                 "c1_menor_MAE_que_todas": c1, "c2_tres_pruebas": c2, "rechazos": rechazos,
                 "c3_efecto": c3, "d_cohen": float(mejor.d_cohen), "delta_cliff": float(mejor.delta_cliff),
                 "c4_consistencia": bool(c4), "gana_en": gana, "significativamente_peor_en": peor,
                 "superior": bool(c1 and c2 and c3 and c4)}
    (C.RES / "veredicto.json").write_text(json.dumps(veredicto, indent=2, ensure_ascii=False), encoding="utf-8")
    (C.RES / "numeros.json").write_text(json.dumps(N, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(veredicto, indent=1, ensure_ascii=False))
    print(mejores[["conjunto", "modelo", "optimizador", "MAE", "MAE_sd", "MASE", "desvio_total_pct"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
