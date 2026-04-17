"""
Tablero Streamlit — API de Análisis de Riesgo Financiero
Conecta directamente con el backend FastAPI en localhost:8000
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Analytics — USTA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API = "http://localhost:8000"
ACTIVOS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
COLORES = {
    "AAPL":  "#58a6ff",
    "MSFT":  "#3fb950",
    "GOOGL": "#d29922",
    "AMZN":  "#bc8cff",
    "TSLA":  "#f85149",
}

# ─────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0d1117; }
  .stMetric { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 16px; }
  .stMetric label { color: #7d8590 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; }
  div[data-testid="metric-container"] { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 16px 20px; }
  .badge-buy  { background: rgba(63,185,80,0.15);  color: #3fb950; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 700; }
  .badge-sell { background: rgba(248,81,73,0.15);  color: #f85149; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 700; }
  .badge-neu  { background: rgba(125,133,144,0.15);color: #7d8590; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_api(path: str):
    """GET request a la API con caché de 5 minutos."""
    try:
        r = requests.get(f"{API}{path}", timeout=30)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=300)
def post_api(path: str, payload: dict):
    """POST request a la API con caché de 5 minutos."""
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def plotly_theme():
    """Configuración de tema oscuro para gráficos Plotly."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="monospace", size=11, color="#7d8590"),
        margin=dict(l=10, r=10, t=40, b=10),
    )

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Risk Analytics")
    st.markdown("**Teoría del Riesgo — USTA**")
    st.divider()

    # Status API
    data, err = get_api("/")
    if data:
        st.success(f"✓ API conectada · v{data.get('version','?')}")
    else:
        st.error(f"✗ API desconectada\n{err}")
        st.info("Ejecuta: `python main.py`")

    st.divider()
    pagina = st.radio(
        "Sección",
        ["Dashboard", "Precios e Indicadores", "VaR & CVaR",
         "Markowitz & CAPM", "Señales", "Macroeconómico"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown(f"<small style='color:#7d8590'>API: `{API}`</small>", unsafe_allow_html=True)
    if st.button("🔄 Limpiar caché"):
        st.cache_data.clear()
        st.rerun()

# ════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════
if pagina == "Dashboard":
    st.title("Dashboard — Portafolio de inversión")

    if not data:
        st.error("No se puede conectar con la API. Asegúrate de que `python main.py` esté corriendo.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Activos", len(data.get("activos_disponibles", [])), help="Activos en el portafolio")
    col2.metric("Estado API", data.get("status", "—").upper())
    col3.metric("Versión", f"v{data.get('version','?')}")
    col4.metric("Fuente datos", "Yahoo Finance + FRED")

    st.divider()

    activos_data, err = get_api("/activos")
    if activos_data:
        st.subheader("Activos del portafolio")
        df = pd.DataFrame(activos_data["activos"])
        st.dataframe(
            df[["ticker","nombre","sector","moneda"]],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Endpoints disponibles")
    endpoints = [
        ("GET",  "/",                   "Health check del servidor"),
        ("GET",  "/activos",            "Lista activos del portafolio"),
        ("GET",  "/precios/{ticker}",   "Precios históricos Yahoo Finance"),
        ("GET",  "/rendimientos/{ticker}", "Rendimientos simples y logarítmicos"),
        ("GET",  "/indicadores/{ticker}", "SMA, EMA, RSI, MACD, Bollinger"),
        ("POST", "/var",                "VaR y CVaR — 3 métodos"),
        ("GET",  "/capm",               "Beta y rendimiento esperado CAPM"),
        ("POST", "/frontera-eficiente", "Frontera eficiente de Markowitz"),
        ("GET",  "/alertas",            "Señales automáticas de trading"),
        ("GET",  "/macro",              "Indicadores macroeconómicos FRED"),
    ]
    df_ep = pd.DataFrame(endpoints, columns=["Método","Ruta","Descripción"])
    st.dataframe(df_ep, use_container_width=True, hide_index=True)

# ════════════════════════════════════════
# PRECIOS E INDICADORES
# ════════════════════════════════════════
elif pagina == "Precios e Indicadores":
    st.title("Precios e Indicadores Técnicos")

    col1, col2, col3 = st.columns([2,2,1])
    ticker     = col1.selectbox("Activo", ACTIVOS, index=0)
    fecha_ini  = col2.selectbox("Horizonte", ["2023-01-01","2022-01-01","2020-01-01"],
                                 index=1, format_func=lambda x: {"2023-01-01":"Último año","2022-01-01":"2 años","2020-01-01":"5 años"}[x])
    calcular   = col3.button("Calcular", type="primary", use_container_width=True)

    if calcular or True:
        with st.spinner(f"Descargando datos de {ticker}..."):
            ind_data, err = get_api(f"/indicadores/{ticker}?fecha_inicio={fecha_ini}")

        if err:
            st.error(f"Error: {err}")
            st.stop()

        datos = pd.DataFrame(ind_data["datos"]).dropna(subset=["cierre"])
        señales = ind_data.get("señales", [])
        resumen = ind_data.get("resumen", {})

        # KPIs
        precio_actual = datos["cierre"].iloc[-1]
        precio_inicio = datos["cierre"].iloc[0]
        cambio_pct    = (precio_actual - precio_inicio) / precio_inicio * 100
        rsi_actual    = resumen.get("rsi_actual", 0) or 0
        vols = np.log(datos["cierre"] / datos["cierre"].shift(1)).dropna()
        vol_anual = vols.std() * np.sqrt(252) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio actual", f"${precio_actual:.2f}", f"{cambio_pct:+.2f}%")
        c2.metric("RSI actual", f"{rsi_actual:.1f}",
                  "Sobrecomprado" if rsi_actual > 70 else "Sobrevendido" if rsi_actual < 30 else "Neutral")
        c3.metric("Volatilidad anual", f"{vol_anual:.1f}%")
        c4.metric("Observaciones", f"{len(datos)} días")

        # Señales activas
        if señales:
            st.divider()
            st.subheader("Señales activas")
            cols = st.columns(len(señales))
            for i, s in enumerate(señales):
                with cols[i]:
                    color = "🟢" if s["tipo"] == "COMPRA" else "🔴"
                    st.info(f"{color} **{s['tipo']}** — {s['indicador']}\n\n{s['descripcion']}")

        # ── Gráfico precio + SMA ──
        st.divider()
        st.subheader(f"Precio histórico — {ticker}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=datos["fecha"], y=datos["cierre"],
            name=ticker, line=dict(color=COLORES[ticker], width=2)))
        if "sma_20" in datos.columns:
            fig.add_trace(go.Scatter(x=datos["fecha"], y=datos["sma_20"],
                name="SMA 20", line=dict(color="#3fb950", width=1, dash="dot")))
        if "sma_50" in datos.columns:
            fig.add_trace(go.Scatter(x=datos["fecha"], y=datos["sma_50"],
                name="SMA 50", line=dict(color="#d29922", width=1, dash="dot")))
        fig.update_layout(**plotly_theme(), height=350,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        # ── RSI + MACD ──
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("RSI (14 días)")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=datos["fecha"], y=datos["rsi_14"],
                name="RSI", line=dict(color="#bc8cff", width=1.5), fill="tozeroy",
                fillcolor="rgba(188,140,255,0.05)"))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="rgba(248,81,73,0.5)",
                              annotation_text="Sobrecomprado (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="rgba(63,185,80,0.5)",
                              annotation_text="Sobrevendido (30)")
            fig_rsi.update_layout(**plotly_theme(), height=250, yaxis=dict(range=[0,100]))
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col_b:
            st.subheader("MACD")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=datos["fecha"], y=datos["macd"],
                name="MACD", line=dict(color="#58a6ff", width=1.5)))
            fig_macd.add_trace(go.Scatter(x=datos["fecha"], y=datos["macd_señal"],
                name="Signal", line=dict(color="#f85149", width=1)))
            hist = datos["macd_hist"].fillna(0)
            fig_macd.add_trace(go.Bar(x=datos["fecha"], y=hist, name="Histograma",
                marker_color=["rgba(63,185,80,0.6)" if v>=0 else "rgba(248,81,73,0.6)" for v in hist]))
            fig_macd.update_layout(**plotly_theme(), height=250,
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_macd, use_container_width=True)

        # ── Bollinger ──
        st.subheader("Bandas de Bollinger")
        fig_boll = go.Figure()
        fig_boll.add_trace(go.Scatter(x=datos["fecha"], y=datos["boll_superior"],
            name="Banda superior", line=dict(color="rgba(248,81,73,0.6)", width=1)))
        fig_boll.add_trace(go.Scatter(x=datos["fecha"], y=datos["boll_inferior"],
            name="Banda inferior", line=dict(color="rgba(63,185,80,0.6)", width=1),
            fill="tonexty", fillcolor="rgba(88,166,255,0.03)"))
        fig_boll.add_trace(go.Scatter(x=datos["fecha"], y=datos["boll_media"],
            name="SMA 20", line=dict(color="rgba(88,166,255,0.6)", width=1, dash="dot")))
        fig_boll.add_trace(go.Scatter(x=datos["fecha"], y=datos["cierre"],
            name=ticker, line=dict(color="#fff", width=1.5)))
        fig_boll.update_layout(**plotly_theme(), height=300,
                               legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_boll, use_container_width=True)

# ════════════════════════════════════════
# VaR & CVaR
# ════════════════════════════════════════
elif pagina == "VaR & CVaR":
    st.title("Valor en Riesgo (VaR) y CVaR")

    col1, col2 = st.columns([2,1])
    confianza = col1.select_slider("Nivel de confianza", [0.90, 0.95, 0.99],
                                    value=0.95, format_func=lambda x: f"{x*100:.0f}%")
    calcular = col2.button("Calcular VaR", type="primary", use_container_width=True)

    payload = {
        "tickers":          ["AAPL","MSFT","GOOGL","AMZN","TSLA"],
        "pesos":            [0.30, 0.25, 0.20, 0.15, 0.10],
        "nivel_confianza":  confianza,
    }

    with st.spinner("Calculando VaR con 3 métodos..."):
        var_data, err = post_api("/var", payload)

    if err:
        st.error(f"Error: {err}")
        st.stop()

    vh = var_data["var_historico"]
    vp = var_data["var_parametrico"]
    vm = var_data["var_monte_carlo"]
    st_p = var_data["estadisticas_portafolio"]
    kp = var_data["backtesting_kupiec"]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VaR Histórico", vh["var_porcentaje"], f"${abs(vh['var_monetario']):,.0f} USD")
    c2.metric("CVaR Histórico", vh["cvar_porcentaje"], f"${abs(vh['cvar_monetario']):,.0f} USD")
    c3.metric("Sharpe Ratio", f"{st_p['sharpe_ratio']:.3f}")
    c4.metric("Volatilidad anual", f"{st_p['volatilidad_anual']*100:.1f}%")

    st.info(f"📌 {vh['interpretacion']}")

    # ── Tabla comparativa ──
    st.divider()
    st.subheader("Comparación de métodos")
    df_var = pd.DataFrame([
        {"Método":"Histórico",   "VaR (%)":vh["var_porcentaje"], "VaR (USD)":f"${abs(vh['var_monetario']):,.0f}", "CVaR (%)":vh["cvar_porcentaje"], "CVaR (USD)":f"${abs(vh['cvar_monetario']):,.0f}", "Supuesto":"Sin supuesto distribucional"},
        {"Método":"Paramétrico", "VaR (%)":vp["var_porcentaje"], "VaR (USD)":f"${abs(vp['var_monetario']):,.0f}", "CVaR (%)":vp["cvar_porcentaje"], "CVaR (USD)":f"${abs(vp['cvar_monetario']):,.0f}", "Supuesto":"Distribución normal"},
        {"Método":"Monte Carlo", "VaR (%)":vm["var_porcentaje"], "VaR (USD)":f"${abs(vm['var_monetario']):,.0f}", "CVaR (%)":vm["cvar_porcentaje"], "CVaR (USD)":f"${abs(vm['cvar_monetario']):,.0f}", "Supuesto":"Normal (10k simulaciones)"},
    ])
    st.dataframe(df_var, use_container_width=True, hide_index=True)

    # ── Gráficas ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("VaR por método")
        fig_var = go.Figure(go.Bar(
            x=["Histórico","Paramétrico","Monte Carlo"],
            y=[abs(vh["var_decimal"])*100, abs(vp["var_decimal"])*100, abs(vm["var_decimal"])*100],
            marker_color=["#f85149","#d29922","#58a6ff"],
            text=[vh["var_porcentaje"], vp["var_porcentaje"], vm["var_porcentaje"]],
            textposition="outside",
        ))
        fig_var.update_layout(**plotly_theme(), height=320,
                              yaxis_title="VaR (%)", showlegend=False)
        st.plotly_chart(fig_var, use_container_width=True)

    with col_b:
        st.subheader("VaR vs CVaR por método")
        metodos = ["Histórico","Paramétrico","Monte Carlo"]
        var_vals  = [abs(vh["var_decimal"])*100, abs(vp["var_decimal"])*100, abs(vm["var_decimal"])*100]
        cvar_vals = [abs(vh["cvar_decimal"])*100, abs(vp["cvar_decimal"])*100, abs(vm["cvar_decimal"])*100]
        fig_cvar = go.Figure()
        fig_cvar.add_trace(go.Bar(name="VaR",  x=metodos, y=var_vals,  marker_color="rgba(248,81,73,0.7)"))
        fig_cvar.add_trace(go.Bar(name="CVaR", x=metodos, y=cvar_vals, marker_color="rgba(248,81,73,1.0)"))
        fig_cvar.update_layout(**plotly_theme(), height=320,
                               barmode="group", yaxis_title="(%)")
        st.plotly_chart(fig_cvar, use_container_width=True)

    # ── Backtesting Kupiec ──
    st.divider()
    st.subheader("Backtesting de Kupiec")
    c1, c2, c3 = st.columns(3)
    c1.metric("Excedencias observadas", kp["excedencias_observadas"], f"de {kp['total_observaciones']} días")
    c2.metric("Tasa real",     f"{kp['tasa_excedencias_real']*100:.2f}%")
    c3.metric("Tasa esperada", f"{kp['tasa_excedencias_esperada']*100:.2f}%")
    if kp["modelo_adecuado"]:
        st.success(f"✓ {kp['interpretacion']}")
    else:
        st.warning(f"⚠ {kp['interpretacion']}")

# ════════════════════════════════════════
# MARKOWITZ & CAPM
# ════════════════════════════════════════
elif pagina == "Markowitz & CAPM":
    st.title("Frontera Eficiente de Markowitz & CAPM")

    calcular = st.button("Calcular Frontera Eficiente + CAPM", type="primary")
    st.caption("Puede tardar ~30 segundos — optimiza 500 portafolios y calcula regresión contra el S&P 500")

    with st.spinner("Optimizando portafolios y calculando Beta..."):
        frontera_data, err1 = post_api("/frontera-eficiente", {
            "tickers": ACTIVOS, "pesos": [0.3,0.25,0.2,0.15,0.1]
        })
        capm_data, err2 = get_api("/capm")

    if err1 or err2:
        st.error(f"Error: {err1 or err2}")
        st.stop()

    # Portafolios óptimos
    st.subheader("Portafolios óptimos")
    c1, c2, c3 = st.columns(3)
    ports = [
        (c1, "Mínima varianza",  frontera_data["portafolio_min_varianza"],  "#3fb950"),
        (c2, "Máximo Sharpe",    frontera_data["portafolio_max_sharpe"],    "#58a6ff"),
        (c3, "Igual ponderado",  frontera_data["portafolio_igual_ponderado"],"#7d8590"),
    ]
    for col, titulo, p, color in ports:
        with col:
            st.metric(titulo, p["retorno_pct"],
                      f"Vol: {p['volatilidad_pct']} · Sharpe: {p['sharpe_ratio']:.3f}")
            pesos_df = pd.DataFrame(list(p["pesos"].items()), columns=["Ticker","Peso"])
            pesos_df["Peso (%)"] = (pesos_df["Peso"] * 100).round(1).astype(str) + "%"
            st.dataframe(pesos_df[["Ticker","Peso (%)"]], hide_index=True, use_container_width=True)

    # ── Frontera eficiente ──
    st.divider()
    st.subheader("Frontera eficiente — nube de Markowitz")
    sim     = frontera_data["simulacion"]
    front   = frontera_data["frontera_eficiente"]
    min_var = frontera_data["portafolio_min_varianza"]
    max_sh  = frontera_data["portafolio_max_sharpe"]

    fig_front = go.Figure()
    fig_front.add_trace(go.Scatter(
        x=[v*100 for v in sim["volatilidades"]],
        y=[r*100 for r in sim["retornos"]],
        mode="markers",
        name="Portafolios simulados",
        marker=dict(
            size=5,
            color=sim["sharpes"],
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Sharpe", thickness=12),
            opacity=0.6,
        ),
    ))
    fig_front.add_trace(go.Scatter(
        x=[p["volatilidad"]*100 for p in front],
        y=[p["retorno"]*100 for p in front],
        mode="lines", name="Frontera eficiente",
        line=dict(color="#3fb950", width=2.5),
    ))
    fig_front.add_trace(go.Scatter(
        x=[min_var["volatilidad_anual"]*100], y=[min_var["retorno_anual"]*100],
        mode="markers", name="Mínima varianza",
        marker=dict(size=14, color="#3fb950", symbol="star"),
    ))
    fig_front.add_trace(go.Scatter(
        x=[max_sh["volatilidad_anual"]*100], y=[max_sh["retorno_anual"]*100],
        mode="markers", name="Máximo Sharpe",
        marker=dict(size=14, color="#58a6ff", symbol="triangle-up"),
    ))
    fig_front.update_layout(
        **plotly_theme(), height=440,
        xaxis_title="Volatilidad anual (%)",
        yaxis_title="Retorno anual (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_front, use_container_width=True)

    # ── CAPM table ──
    st.divider()
    st.subheader("CAPM — Beta y Alpha por activo")
    activos_capm = capm_data.get("activos", {})
    rows = []
    for ticker, d in activos_capm.items():
        rows.append({
            "Ticker":                   ticker,
            "Beta":                     round(d.get("beta", 0), 4),
            "Alpha anual (%)":          round((d.get("alpha_anual", 0) or 0) * 100, 2),
            "Rend. esperado CAPM (%)":  round((d.get("rendimiento_esperado_capm", 0) or 0) * 100, 2),
            "Volatilidad anual (%)":    round((d.get("volatilidad_anual", 0) or 0) * 100, 1),
            "R²":                       round(d.get("r_cuadrado", 0) or 0, 4),
            "Interpretación Beta":      d.get("interpretacion_beta", "—"),
        })
    df_capm = pd.DataFrame(rows)
    st.dataframe(df_capm, use_container_width=True, hide_index=True)

    # ── Gráfico Beta ──
    st.subheader("Beta por activo vs mercado (S&P 500)")
    tickers_list = [r["Ticker"] for r in rows]
    betas_list   = [r["Beta"] for r in rows]
    fig_beta = go.Figure(go.Bar(
        x=tickers_list, y=betas_list,
        marker_color=[COLORES[t] for t in tickers_list],
        text=[f"{b:.3f}" for b in betas_list],
        textposition="outside",
    ))
    fig_beta.add_hline(y=1, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                       annotation_text="Beta = 1 (mercado)")
    fig_beta.update_layout(**plotly_theme(), height=320,
                           yaxis_title="Beta", showlegend=False)
    st.plotly_chart(fig_beta, use_container_width=True)

# ════════════════════════════════════════
# SEÑALES
# ════════════════════════════════════════
elif pagina == "Señales":
    st.title("Señales automáticas de trading")

    if st.button("Actualizar señales en tiempo real", type="primary"):
        st.cache_data.clear()

    with st.spinner("Analizando indicadores técnicos de todos los activos..."):
        alertas_data, err = get_api("/alertas")

    if err:
        st.error(f"Error: {err}")
        st.stop()

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total alertas",   alertas_data["total_alertas"])
    c2.metric("Señales compra",  alertas_data["alertas_compra"],  delta="alcistas")
    c3.metric("Señales venta",   alertas_data["alertas_venta"],   delta="bajistas")
    c4.metric("Análisis",        alertas_data["fecha_analisis"])

    # ── Resumen por ticker ──
    st.divider()
    st.subheader("Señal neta por activo")
    resumen = alertas_data.get("resumen", {})
    cols = st.columns(len(resumen))
    for i, (ticker, d) in enumerate(resumen.items()):
        if d.get("error"):
            continue
        with cols[i]:
            señal = d.get("señal_neta", "NEUTRAL")
            color = "🟢" if señal == "COMPRA" else "🔴" if señal == "VENTA" else "⚪"
            st.metric(
                f"{color} {ticker}",
                f"${d.get('precio_actual', 0):.2f}",
                f"RSI: {d.get('rsi_actual', 0):.1f}",
            )
            if señal == "COMPRA":
                st.success(f"**{señal}**")
            elif señal == "VENTA":
                st.error(f"**{señal}**")
            else:
                st.info(f"**{señal}**")

    # ── Tabla detalle ──
    st.divider()
    st.subheader("Detalle de alertas activas")
    alertas = alertas_data.get("alertas", [])
    if alertas:
        df_alertas = pd.DataFrame(alertas)
        df_alertas["tipo_color"] = df_alertas["tipo"].map(
            {"COMPRA": "🟢 COMPRA", "VENTA": "🔴 VENTA"}
        )
        st.dataframe(
            df_alertas[["ticker","tipo_color","indicador","fuerza","descripcion","valor","fecha"]]
            .rename(columns={"tipo_color":"tipo"}),
            use_container_width=True,
            hide_index=True,
        )

        # Gráfico distribución
        st.subheader("Distribución de señales por indicador")
        conteo = df_alertas.groupby(["indicador","tipo"]).size().reset_index(name="count")
        fig_dist = px.bar(
            conteo, x="indicador", y="count", color="tipo",
            color_discrete_map={"COMPRA":"#3fb950","VENTA":"#f85149"},
            template="plotly_dark", height=320,
        )
        fig_dist.update_layout(**plotly_theme())
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("No hay alertas activas en este momento.")

# ════════════════════════════════════════
# MACROECONÓMICO
# ════════════════════════════════════════
elif pagina == "Macroeconómico":
    st.title("Indicadores Macroeconómicos — FRED")

    with st.spinner("Consultando FRED API..."):
        macro_data, err = get_api("/macro")

    if err:
        st.error(f"Error: {err}")
        st.stop()

    datos = macro_data.get("datos", {})

    # Cards macro
    series_config = {
        "DGS3MO":   {"emoji":"🏦", "label":"T-Bills 3M (Rf)"},
        "DGS10":    {"emoji":"📊", "label":"Tesoro 10 años"},
        "CPIAUCSL": {"emoji":"📈", "label":"Inflación CPI"},
        "UNRATE":   {"emoji":"👥", "label":"Desempleo"},
        "FEDFUNDS": {"emoji":"⚙️", "label":"Tasa Fed"},
        "VIXCLS":   {"emoji":"⚡", "label":"VIX"},
    }

    keys = [k for k in series_config if k in datos and not k.startswith("_")]
    cols = st.columns(len(keys))
    for i, key in enumerate(keys):
        d   = datos[key]
        cfg = series_config[key]
        val = d.get("valor")
        delta_text = d.get("fecha", "")
        with cols[i]:
            st.metric(
                f"{cfg['emoji']} {cfg['label']}",
                f"{val:.2f}%" if val is not None and key != "VIXCLS" else (f"{val:.1f} pts" if val else "—"),
                delta_text,
            )

    if macro_data.get("nota"):
        st.info(f"ℹ️ {macro_data['nota']}")
        st.markdown(f"[Obtener clave FRED gratis →](https://fred.stlouisfed.org/docs/api/api_key.html)")

    ctx = macro_data.get("contexto_macro")
    if ctx:
        st.divider()
        st.subheader("Contexto macroeconómico")
        st.write(ctx.get("descripcion",""))
        st.subheader("Impacto en el portafolio")
        for item in ctx.get("impacto_portafolio",[]):
            st.markdown(f"→ {item}")

    # Gráfico barras macro
    if keys:
        st.divider()
        st.subheader("Comparación de tasas")
        tasas_keys = [k for k in ["DGS3MO","DGS10","FEDFUNDS","CPIAUCSL","UNRATE"] if k in datos]
        fig_macro = go.Figure(go.Bar(
            x=[series_config[k]["label"] for k in tasas_keys],
            y=[datos[k].get("valor", 0) for k in tasas_keys],
            marker_color=["#58a6ff","#3fb950","#d29922","#f85149","#bc8cff"],
            text=[f"{datos[k].get('valor',0):.2f}%" for k in tasas_keys],
            textposition="outside",
        ))
        fig_macro.update_layout(**plotly_theme(), height=340,
                                yaxis_title="%", showlegend=False)
        st.plotly_chart(fig_macro, use_container_width=True)
