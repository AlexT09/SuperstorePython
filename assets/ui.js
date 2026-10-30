/* Efectos interactivos del panel (Dash carga este archivo solo por estar en assets/).
   - Menú lateral plegable (se recuerda entre visitas).
   - Brillo de las tarjetas que sigue al cursor.
   - Cifras de los KPI que cuentan desde 0 al aparecer.
   - Barra de progreso de lectura, botón "volver arriba" y aparición de bloques al hacer scroll.
   Dash pinta el contenido después de cargar la página, por eso se observa el DOM con MutationObserver. */
(function () {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function guardar(clave, valor) { try { localStorage.setItem(clave, valor); } catch (e) { /* sin almacenamiento */ } }
  function leer(clave) { try { return localStorage.getItem(clave); } catch (e) { return null; } }

  // ---------- menú plegable ----------
  if (leer("sidebar-mini") === "1") document.body.classList.add("sidebar-mini");
  document.addEventListener("click", function (e) {
    if (!e.target.closest(".btn-menu")) return;
    const mini = document.body.classList.toggle("sidebar-mini");
    guardar("sidebar-mini", mini ? "1" : "0");
    // los gráficos de Plotly se reajustan al nuevo ancho cuando termina la transición
    setTimeout(function () { window.dispatchEvent(new Event("resize")); }, 380);
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "[" && !/input|textarea/i.test(e.target.tagName)) {
      const b = document.querySelector(".btn-menu"); if (b) b.click();
    }
  });

  // ---------- brillo bajo el cursor ----------
  document.addEventListener("pointermove", function (e) {
    const card = e.target.closest(".soft-card, .kpi, .hero, .nav-card");
    if (!card) return;
    const r = card.getBoundingClientRect();
    card.style.setProperty("--mx", (e.clientX - r.left) + "px");
    card.style.setProperty("--my", (e.clientY - r.top) + "px");
  }, { passive: true });

  // ---------- conteo animado de cifras ----------
  // Separa prefijo, número y sufijo: "$2,297,201" · "+12.4 %" · "±19 %" · "0.65"
  const RE = /^([^\d\-+]*?)([+\-]?)([\d,]*\.?\d+)(.*)$/;
  function animarCifra(el) {
    if (el.dataset.contado) return;
    el.dataset.contado = "1";
    // se anima el nodo de texto que maneja React (no se reemplaza), así los callbacks lo siguen actualizando
    const nodo = el.firstChild;
    if (!nodo || nodo.nodeType !== 3 || el.childNodes.length !== 1) return;
    const texto = nodo.nodeValue;
    const m = texto.trim().match(RE);
    if (!m || reduceMotion) return;
    const [, pre, signo, num, suf] = m;
    const objetivo = parseFloat(num.replace(/,/g, ""));
    if (!isFinite(objetivo) || objetivo === 0) return;
    const dec = (num.split(".")[1] || "").length;
    const miles = num.includes(",");
    const dur = 1100, t0 = performance.now();
    function fmt(v) {
      let s = v.toFixed(dec);
      if (miles) { const p = s.split("."); p[0] = p[0].replace(/\B(?=(\d{3})+(?!\d))/g, ","); s = p.join("."); }
      return pre + signo + s + suf;
    }
    let puesto = null;
    (function paso(t) {
      if (puesto !== null && nodo.nodeValue !== puesto) return;  // Dash cambió el valor: se deja el nuevo
      const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 4);
      puesto = k < 1 ? fmt(objetivo * e) : texto;
      nodo.nodeValue = puesto;
      if (k < 1) requestAnimationFrame(paso);
    })(t0);
  }

  // ---------- aparición al hacer scroll ----------
  const io = "IntersectionObserver" in window ? new IntersectionObserver(function (entradas) {
    entradas.forEach(function (en) {
      if (!en.isIntersecting) return;
      en.target.classList.add("visible");
      en.target.querySelectorAll(".kpi-value").forEach(animarCifra);
      io.unobserve(en.target);
    });
  }, { rootMargin: "0px 0px -8% 0px" }) : null;

  function preparar(raiz) {
    raiz.querySelectorAll(".kpi-value:not([data-contado])").forEach(function (el) {
      const r = el.getBoundingClientRect();
      if (r.top < window.innerHeight) animarCifra(el);
    });
    if (!io) return;
    raiz.querySelectorAll(".contenido .section-title:not(.obs), .contenido .kpi:not(.obs)").forEach(function (el) {
      el.classList.add("obs");
      const r = el.getBoundingClientRect();
      if (r.top > window.innerHeight) { el.classList.add("revelar"); io.observe(el); }
    });
  }
  new MutationObserver(function () { preparar(document); })
    .observe(document.body, { childList: true, subtree: true });

  // ---------- progreso de lectura y volver arriba ----------
  let arriba = null, barra = null;
  function crearControles() {
    if (barra) return;
    barra = document.createElement("div"); barra.className = "scroll-progress"; document.body.appendChild(barra);
    arriba = document.createElement("button");
    arriba.className = "btn-arriba"; arriba.type = "button"; arriba.title = "Volver arriba";
    arriba.setAttribute("aria-label", "Volver arriba");
    arriba.innerHTML = '<i class="bi bi-arrow-up"></i>';
    arriba.addEventListener("click", function () { window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" }); });
    document.body.appendChild(arriba);
  }
  function alDesplazar() {
    crearControles();
    const h = document.documentElement.scrollHeight - window.innerHeight;
    const p = h > 0 ? window.scrollY / h : 0;
    barra.style.setProperty("--scroll", p.toFixed(4));
    arriba.classList.toggle("visible", window.scrollY > 600);
  }
  window.addEventListener("scroll", alDesplazar, { passive: true });
  document.addEventListener("DOMContentLoaded", alDesplazar);

  // al cambiar de sección se vuelve arriba
  const push = history.pushState;
  history.pushState = function () { push.apply(this, arguments); window.scrollTo({ top: 0 }); };
})();
