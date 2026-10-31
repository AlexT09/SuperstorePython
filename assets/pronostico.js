/* Pronóstico animado (sección tabs/pronostico.py). Dibuja en SVG:
   - barras trimestrales tipo holograma (2014–2017 reales, 2018 pronóstico con su rango de error),
   - una línea punteada con las ventas de los últimos 12 meses y un círculo por año (el total del año),
   - un punto que se enciende en 2018 con el total proyectado.
   Al presionar "Predecir 2018": se dibuja la historia, se escanea 2018, suben sus barras y se enciende el punto. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const reducir = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const TONOS = ["#e9f1f7", "#5fd0f0", "#8fd3b0", "#f0b851"]; // T1 blanco, T2 cian, T3 verde agua, T4 ámbar
  const usd = (v) => "$" + Math.round(v).toLocaleString("en-US");
  const corto = (v) => (v >= 1e6 ? "$" + (v / 1e6).toFixed(1) + "M" : "$" + Math.round(v / 1e3) + "k");
  const estado = { datos: null, clics: 0, predicho: false, temporizadores: [] };

  function el(tag, attrs, padre) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (padre) padre.appendChild(n);
    return n;
  }
  function despues(ms, fn) { estado.temporizadores.push(setTimeout(fn, reducir ? 0 : ms)); }
  function animar(nodo, cuadros, opciones) {
    if (reducir || !nodo.animate) return;
    nodo.animate(cuadros, Object.assign({ fill: "backwards", easing: "cubic-bezier(.16,1,.3,1)" }, opciones));
  }
  function contar(nodo, desde, hasta, ms, formato) {
    if (!nodo) return;
    if (reducir) { nodo.textContent = formato(hasta); return; }
    const t0 = performance.now();
    (function paso(t) {
      const p = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - p, 3);
      nodo.textContent = formato(desde + (hasta - desde) * e);
      if (p < 1) requestAnimationFrame(paso);
    })(t0);
  }

  // modo: "espera" (historia + 2018 en blanco), "prediccion" (secuencia completa) o "final" (sin animación)
  function dibujar(modo) {
    const caja = document.getElementById("pron-holo");
    const d = estado.datos;
    if (!caja || !d) return;
    estado.temporizadores.forEach(clearTimeout);
    estado.temporizadores = [];
    caja.innerHTML = "";

    const W = caja.clientWidth, H = caja.clientHeight;
    if (W < 50 || H < 50) return;
    const m = { l: 58, r: Math.max(150, W * 0.11), t: 70, b: 34 };
    const anchoUtil = W - m.l - m.r, altoUtil = H - m.t - m.b;
    const B = d.barras, n = B.length, paso = anchoUtil / n;
    const x = (i) => m.l + paso * (i + 0.5);
    const base = H - m.b;

    // Escalas: las barras usan el 58 % inferior; la línea, la franja superior
    const maxB = Math.max(...B.map((b) => b.v * (b.pron ? 1 + d.err / 100 : 1))) * 1.05;
    const altoB = altoUtil * 0.58;
    const yB = (v) => base - (v / maxB) * altoB;
    const mov = B.filter((b) => b.m != null).map((b) => b.m);
    const minL = Math.min(...mov), maxL = Math.max(...mov), rango = maxL - minL || 1;
    const yL = (v) => m.t + altoUtil * 0.5 - ((v - minL) / rango) * altoUtil * 0.5;

    const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "pron-svg", role: "img",
      "aria-label": `Ventas trimestrales 2014 a 2017 y pronóstico 2018: ${usd(d.v18)}` }, caja);
    const defs = el("defs", {}, svg);
    el("feGaussianBlur", { stdDeviation: 4, result: "b" }, el("filter", { id: "pron-brillo", x: "-50%", y: "-50%", width: "200%", height: "200%" }, defs));
    const halo = el("radialGradient", { id: "pron-halo" }, defs);
    el("stop", { offset: "0%", "stop-color": "#fff3d6", "stop-opacity": 1 }, halo);
    el("stop", { offset: "28%", "stop-color": "#ff8a3d", "stop-opacity": 0.85 }, halo);
    el("stop", { offset: "65%", "stop-color": "#ff4d2e", "stop-opacity": 0.25 }, halo);
    el("stop", { offset: "100%", "stop-color": "#ff4d2e", "stop-opacity": 0 }, halo);
    const haz = el("linearGradient", { id: "pron-haz", x1: 0, y1: 0, x2: 0, y2: 1 }, defs);
    el("stop", { offset: "0%", "stop-color": "#ffd9a0", "stop-opacity": 0.9 }, haz);
    el("stop", { offset: "100%", "stop-color": "#5fd0f0", "stop-opacity": 0 }, haz);
    TONOS.concat(["#7fe8ff"]).forEach((c, i) => {
      const g = el("linearGradient", { id: `pron-g${i}`, x1: 0, y1: 0, x2: 0, y2: 1 }, defs);
      el("stop", { offset: "0%", "stop-color": c, "stop-opacity": 0.95 }, g);
      el("stop", { offset: "70%", "stop-color": c, "stop-opacity": 0.35 }, g);
      el("stop", { offset: "100%", "stop-color": c, "stop-opacity": 0.04 }, g);
    });

    // Cuadrícula de las barras con sus montos
    const rejilla = el("g", { class: "pron-rejilla" }, svg);
    [0, 0.5, 1].forEach((f) => {
      const y = base - altoB * f;
      el("line", { x1: m.l, x2: W - m.r + 40, y1: y, y2: y }, rejilla);
      el("text", { x: m.l - 10, y: y + 4, "text-anchor": "end" }, rejilla).textContent = corto(maxB * f);
    });
    // Zona de 2018
    const i18 = B.findIndex((b) => b.pron);
    const xz = m.l + paso * i18;
    const zona = el("rect", { x: xz, y: m.t - 40, width: paso * (n - i18), height: base - m.t + 40, rx: 14, class: "pron-zona" }, svg);
    el("text", { x: xz + 12, y: m.t - 22, class: "pron-zona-etq" }, svg).textContent = "Pronóstico";

    // Etiquetas de años en el eje
    B.forEach((b, i) => {
      if (b.q === 1) el("text", { x: x(i) + paso * 1.5, y: base + 22, class: "pron-eje" + (b.pron ? " pron" : "") }, svg).textContent = b.anio;
    });

    const tip = document.createElement("div");
    tip.className = "pron-tip";
    caja.appendChild(tip);
    function mostrarTip(e, html) {
      tip.innerHTML = html;
      tip.classList.add("visible");
      const r = caja.getBoundingClientRect();
      tip.style.left = Math.min(e.clientX - r.left + 14, W - 190) + "px";
      tip.style.top = Math.max(e.clientY - r.top - 50, 4) + "px";
    }
    const ocultarTip = () => tip.classList.remove("visible");

    // Barras
    const anchoBarra = Math.max(4, Math.min(22, paso * 0.42));
    const gBarras = el("g", {}, svg);
    const barras18 = [];
    B.forEach((b, i) => {
      const g = el("g", { class: "pron-barra" + (b.pron ? " pron" : "") }, gBarras);
      if (b.pron && modo === "espera") {
        el("rect", { x: x(i) - anchoBarra / 2, y: base - altoB * 0.35, width: anchoBarra, height: altoB * 0.35, rx: 3, class: "pron-fantasma" }, g);
        if (b.q === 2) el("text", { x: x(i) + paso / 2, y: base - altoB * 0.35 - 12, class: "pron-interroga", "text-anchor": "middle" }, g).textContent = "?";
        return;
      }
      const y = yB(b.v);
      const rect = el("rect", { x: x(i) - anchoBarra / 2, y, width: anchoBarra, height: base - y, rx: 3,
        fill: `url(#pron-g${b.pron ? 4 : (b.q - 1)})` }, g);
      el("rect", { x: x(i) - anchoBarra / 2, y, width: anchoBarra, height: 2.5, rx: 1, class: "pron-tapa",
        fill: b.pron ? "#bff4ff" : TONOS[b.q - 1] }, g);
      if (b.pron) {
        const lo = yB(b.v * (1 - d.err / 100)), hi = yB(b.v * (1 + d.err / 100));
        const bigote = el("g", { class: "pron-bigote" }, g);
        el("line", { x1: x(i), x2: x(i), y1: hi, y2: lo }, bigote);
        el("line", { x1: x(i) - 5, x2: x(i) + 5, y1: hi, y2: hi }, bigote);
        el("line", { x1: x(i) - 5, x2: x(i) + 5, y1: lo, y2: lo }, bigote);
        barras18.push({ g, bigote });
      }
      const zonaHover = el("rect", { x: x(i) - paso / 2, y: m.t, width: paso, height: base - m.t, fill: "transparent" }, g);
      zonaHover.addEventListener("mousemove", (e) => mostrarTip(e, `<strong>${b.t}</strong>${usd(b.v)}` +
        (b.pron ? `<small>pronóstico · rango ${usd(b.v * (1 - d.err / 100))} – ${usd(b.v * (1 + d.err / 100))}</small>` : "<small>ventas reales</small>")));
      zonaHover.addEventListener("mouseleave", ocultarTip);
      if (modo === "prediccion" && !b.pron) animar(g, [{ transform: "scaleY(0)", opacity: 0 }, { transform: "scaleY(1)", opacity: 1 }], { duration: 700, delay: i * 45 });
      if (modo === "espera") animar(g, [{ transform: "scaleY(0)", opacity: 0 }, { transform: "scaleY(1)", opacity: 1 }], { duration: 600, delay: i * 25 });
    });

    // Línea punteada (12 meses móviles): tramo real y tramo pronosticado
    const pts = B.map((b, i) => (b.m == null ? null : { i, b, x: x(i), y: yL(b.m) })).filter(Boolean);
    const real = pts.filter((p) => !p.b.pron);
    const iUltReal = real[real.length - 1];
    const pron = [iUltReal].concat(pts.filter((p) => p.b.pron));
    const camino = (ps) => ps.map((p, k) => (k ? "L" : "M") + p.x.toFixed(1) + " " + p.y.toFixed(1)).join(" ");

    const clipReal = el("rect", { x: 0, y: 0, width: W, height: H }, el("clipPath", { id: "pron-clip-real" }, defs));
    const clipPron = el("rect", { x: 0, y: 0, width: W, height: H }, el("clipPath", { id: "pron-clip-pron" }, defs));
    el("path", { d: camino(real), class: "pron-linea", "clip-path": "url(#pron-clip-real)" }, svg);
    const lineaPron = modo === "espera" ? null : el("path", { d: camino(pron), class: "pron-linea pron", "clip-path": "url(#pron-clip-pron)" }, svg);

    // Círculos de cada año (T4 = total del año)
    const marcas = el("g", {}, svg);
    pts.filter((p) => p.b.q === 4 && !p.b.pron).forEach((p) => {
      const g = el("g", { class: "pron-anio", transform: `translate(${p.x},${p.y})` }, marcas);
      el("circle", { r: 10, class: "pron-anillo" }, g);
      el("circle", { r: 3.6, class: "pron-nucleo" }, g);
      el("text", { y: -18, "text-anchor": "middle", class: "pron-anio-etq" }, g).textContent = p.b.anio;
      el("circle", { r: 18, fill: "transparent" }, g).addEventListener("mousemove", (e) =>
        mostrarTip(e, `<strong>Total ${p.b.anio}</strong>${usd(p.b.m)}<small>ventas reales del año</small>`));
      g.lastChild.addEventListener("mouseleave", ocultarTip);
      const retraso = modo === "prediccion" ? 1400 * (p.x - m.l) / (iUltReal.x - m.l) : 0;
      if (modo === "prediccion") animar(g, [{ opacity: 0, transform: `translate(${p.x}px,${p.y}px) scale(0)` },
        { opacity: 1, transform: `translate(${p.x}px,${p.y}px) scale(1)` }], { duration: 500, delay: retraso });
    });

    // Punto final de 2018
    const fin = pts[pts.length - 1];
    // En espera el círculo no revela la altura del pronóstico: queda a la altura del último año real
    const yFin = modo === "espera" ? iUltReal.y - 24 : fin.y;
    const gFin = el("g", { class: "pron-fin" + (modo === "espera" ? " espera" : ""), transform: `translate(${fin.x},${yFin})` }, svg);
    if (modo === "espera") {
      el("circle", { r: 14, class: "pron-anillo espera" }, gFin);
      el("text", { y: -26, "text-anchor": "middle", class: "pron-fin-anio" }, gFin).textContent = "2018";
      el("text", { y: 5, "text-anchor": "middle", class: "pron-interroga chico" }, gFin).textContent = "?";
    } else {
      el("line", { x1: 0, x2: 0, y1: 16, y2: base - fin.y, class: "pron-haz" }, gFin);
      el("circle", { r: 58, fill: "url(#pron-halo)", class: "pron-halo" }, gFin);
      el("circle", { r: 15, class: "pron-anillo fin" }, gFin);
      el("circle", { r: 6.5, class: "pron-nucleo fin", filter: "url(#pron-brillo)" }, gFin);
      el("circle", { r: 4.5, class: "pron-nucleo fin" }, gFin);
      const etq = el("g", { class: "pron-fin-etq" }, gFin);
      el("text", { y: -30, "text-anchor": "middle", class: "pron-fin-anio" }, etq).textContent = "2018";
      // Monto y crecimiento a la derecha del punto, para no salirse por arriba
      const valor = el("text", { x: 26, y: 2, class: "pron-fin-valor" }, etq);
      const crec = el("text", { x: 26, y: 19, class: "pron-fin-crec" + (d.crec < 0 ? " baja" : "") }, etq);
      valor.textContent = usd(d.v18);
      crec.textContent = `${d.crec >= 0 ? "▲" : "▼"} ${Math.abs(d.crec).toFixed(1)} % vs 2017`;
      el("circle", { r: 22, fill: "transparent" }, gFin).addEventListener("mousemove", (e) =>
        mostrarTip(e, `<strong>Pronóstico 2018</strong>${usd(d.v18)}<small>error típico mensual ±${d.err.toFixed(0)} %</small>`));
      gFin.lastChild.addEventListener("mouseleave", ocultarTip);

      if (modo === "prediccion") {
        const escanear = el("rect", { x: xz, y: m.t - 40, width: 26, height: base - m.t + 40, class: "pron-escaner" }, svg);
        animar(escanear, [{ transform: "translateX(0)", opacity: 0 }, { opacity: 1, offset: 0.2 }, { opacity: 1, offset: 0.8 },
          { transform: `translateX(${paso * (n - i18) - 26}px)`, opacity: 0 }], { duration: 800, delay: 1450, fill: "both", easing: "ease-in-out" });
        animar(zona, [{ opacity: 0.2 }, { opacity: 1 }], { duration: 600, delay: 1400 });
        barras18.forEach(({ g, bigote }, k) => {
          animar(g, [{ transform: "scaleY(0)", opacity: 0 }, { transform: "scaleY(1)", opacity: 1 }], { duration: 650, delay: 2200 + k * 170 });
          animar(bigote, [{ opacity: 0 }, { opacity: 1 }], { duration: 400, delay: 2800 + k * 170 });
        });
        animar(clipReal, [{ width: "0px" }, { width: `${iUltReal.x + 4}px` }], { duration: 1400, easing: "linear" });
        animar(clipPron, [{ x: `${iUltReal.x}px`, width: "0px" }, { x: `${iUltReal.x}px`, width: `${fin.x - iUltReal.x + 20}px` }],
          { duration: 800, delay: 2200, easing: "ease-out" });
        const T = 3000;
        animar(gFin.querySelector(".pron-halo"), [{ transform: "scale(0)", opacity: 0 }, { transform: "scale(1.5)", opacity: 1, offset: 0.55 },
          { transform: "scale(1)", opacity: 1 }], { duration: 1100, delay: T });
        animar(gFin.querySelector(".pron-haz"), [{ opacity: 0, transform: "scaleY(0)" }, { opacity: 1, transform: "scaleY(1)" }], { duration: 700, delay: T + 150 });
        gFin.querySelectorAll(".pron-anillo, .pron-nucleo").forEach((c) =>
          animar(c, [{ transform: "scale(0)", opacity: 0 }, { transform: "scale(1)", opacity: 1 }], { duration: 600, delay: T }));
        animar(etq, [{ opacity: 0, transform: "translateY(12px)" }, { opacity: 1, transform: "translateY(0)" }], { duration: 700, delay: T + 250 });
        despues(T + 250, () => contar(valor, d.v17, d.v18, 1300, usd));
      }
    }

    // Estado y KPI
    const estadoTxt = document.createElement("div");
    estadoTxt.className = "pron-estado";
    caja.appendChild(estadoTxt);
    const k18 = document.getElementById("pron-k18"), k18d = document.getElementById("pron-k18-det"), kc = document.getElementById("pron-kcrec");
    const btn = document.getElementById("pron-btn");
    if (modo === "espera") {
      estadoTxt.textContent = `${d.filtro} · listo para predecir`;
      if (k18) k18.textContent = "—";
      if (k18d) k18d.textContent = "presiona Predecir";
      if (kc) { kc.textContent = "—"; kc.className = "pron-k-val"; }
      if (btn) btn.classList.add("pulso");
      return;
    }
    if (btn) btn.classList.remove("pulso");
    const fijarKpis = (animado) => {
      if (k18d) k18d.textContent = `rango mensual ±${d.err.toFixed(0)} %`;
      if (kc) kc.className = "pron-k-val " + (d.crec >= 0 ? "sube" : "baja");
      if (animado) {
        contar(k18, d.v17, d.v18, 1300, usd);
        contar(kc, 0, d.crec, 1300, (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)} %`);
      } else {
        if (k18) k18.textContent = usd(d.v18);
        if (kc) kc.textContent = `${d.crec >= 0 ? "+" : ""}${d.crec.toFixed(1)} %`;
      }
    };
    if (modo === "prediccion") {
      estadoTxt.textContent = `${d.filtro} · leyendo la historia 2014–2017…`;
      despues(1450, () => { estadoTxt.textContent = "Calculando 2018 mes a mes con Random Forest…"; estadoTxt.classList.add("activo"); });
      despues(3000, () => { estadoTxt.textContent = `${d.filtro} · pronóstico listo`; estadoTxt.classList.remove("activo"); fijarKpis(true); });
      if (k18) k18.textContent = "…";
    } else {
      estadoTxt.textContent = `${d.filtro} · pronóstico listo`;
      fijarKpis(false);
    }
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    pronostico: {
      dibujar: function (datos, clics) {
        const porBoton = (clics || 0) > estado.clics;
        estado.clics = clics || 0;
        estado.datos = datos;
        if (porBoton) estado.predicho = true;
        else if (!clics) estado.predicho = false; // se volvió a abrir la sección
        // Al cambiar un filtro después de predecir, la predicción se vuelve a animar
        requestAnimationFrame(() => dibujar(estado.predicho ? "prediccion" : "espera"));
        return "";
      },
    },
  });

  let espera;
  window.addEventListener("resize", () => {
    clearTimeout(espera);
    espera = setTimeout(() => estado.datos && dibujar(estado.predicho ? "final" : "espera"), 200);
  });
})();
