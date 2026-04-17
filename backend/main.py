from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn

from models import PortafolioRequest, HealthCheck
from services.datos import (
    descargar_precios,
    obtener_info_activo,
    ACTIVOS_INFO,
)

app = FastAPI(
    title="API de Análisis de Riesgo Financiero",
    description="""
Sistema de análisis de riesgo para portafolios de inversión.

**Módulos disponibles:**
- Precios históricos reales (Yahoo Finance) ✅
- Rendimientos y estadísticas
- Indicadores técnicos: SMA, EMA, RSI, MACD, Bollinger
- Valor en Riesgo (VaR) y CVaR
- CAPM y Beta
- Frontera eficiente de Markowitz
- Señales de trading automatizadas
- Datos macroeconómicos (FRED)
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ACTIVOS = list(ACTIVOS_INFO.keys())


# ─────────────────────────────────────────────
# ENDPOINT 1: HEALTH CHECK — GET /
# ─────────────────────────────────────────────

@app.get("/", response_model=HealthCheck, tags=["Sistema"])
def health_check():
    """Verifica que el servidor esté corriendo. Responde instantáneo."""
    return HealthCheck(
        status="ok",
        mensaje="API de Riesgo Financiero funcionando correctamente",
        version="1.0.0",
        activos_disponibles=ACTIVOS,
    )


# ─────────────────────────────────────────────
# ENDPOINT 2: LISTAR ACTIVOS — GET /activos
# ─────────────────────────────────────────────

@app.get("/activos", tags=["Activos"])
def listar_activos():
    """
    Lista los activos del portafolio.
    Retorna información básica SIN llamar a Yahoo Finance
    para que sea instantáneo.
    """
    activos = []
    for ticker, info in ACTIVOS_INFO.items():
        activos.append({
            "ticker": ticker,
            "nombre": info["nombre"],
            "sector": info["sector"],
            "moneda": info["moneda"],
            "nota":   "Usa /precios/{ticker} para precios en tiempo real",
        })
    return {
        "total":   len(activos),
        "activos": activos,
        "fuente":  "Yahoo Finance (disponible en /precios/{ticker})",
    }


# ─────────────────────────────────────────────
# ENDPOINT 2b: PRECIO ACTUAL — GET /activos/{ticker}/precio
# ─────────────────────────────────────────────

@app.get("/activos/{ticker}/precio", tags=["Activos"])
def precio_actual(ticker: str):
    """
    Retorna el precio actual de UN activo desde Yahoo Finance.
    Este endpoint SÍ llama a Yahoo Finance (puede tardar unos segundos).
    """
    ticker = ticker.upper()
    if ticker not in ACTIVOS:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' no encontrado. Disponibles: {ACTIVOS}",
        )
    info = obtener_info_activo(ticker)
    return info


# ─────────────────────────────────────────────
# ENDPOINT 3: PRECIOS HISTÓRICOS — GET /precios/{ticker}
# ─────────────────────────────────────────────

@app.get("/precios/{ticker}", tags=["Precios"])
def obtener_precios(
    ticker: str,
    fecha_inicio: str = Query(default="2022-01-01", description="Formato YYYY-MM-DD"),
    fecha_fin: Optional[str] = Query(default=None,  description="Formato YYYY-MM-DD"),
):
    """
    Retorna precios históricos reales desde Yahoo Finance.
    Puede tardar 5-10 segundos porque descarga datos de internet.

    - **ticker**: AAPL, MSFT, GOOGL, AMZN o TSLA
    - **fecha_inicio**: desde cuándo traer datos
    - **fecha_fin**: hasta cuándo (por defecto: hoy)
    """
    ticker = ticker.upper()
    if ticker not in ACTIVOS:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' no encontrado. Disponibles: {ACTIVOS}",
        )
    try:
        df = descargar_precios(ticker, fecha_inicio, fecha_fin)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "ticker":       ticker,
        "fecha_inicio": fecha_inicio,
        "fecha_fin":    fecha_fin or "hoy",
        "total_dias":   len(df),
        "fuente":       "Yahoo Finance",
        "datos":        df.to_dict(orient="records"),
    }


# ─────────────────────────────────────────────
# ENDPOINT 4: RENDIMIENTOS — GET /rendimientos/{ticker}
# ─────────────────────────────────────────────

@app.get("/rendimientos/{ticker}", tags=["Análisis"])
def obtener_rendimientos(
    ticker: str,
    fecha_inicio: str = Query(default="2022-01-01"),
):
    """Calcula rendimientos logarítmicos. Se implementará en la Fase 5."""
    ticker = ticker.upper()
    if ticker not in ACTIVOS:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado")
    return {
        "ticker": ticker,
        "nota":   "Cálculo real en Fase 5",
        "estadisticas_ejemplo": {
            "media_diaria":  0.0012,
            "desv_estandar": 0.0189,
            "minimo":       -0.0892,
            "maximo":        0.0756,
        },
    }


# ─────────────────────────────────────────────
# ENDPOINT 5: INDICADORES — GET /indicadores/{ticker}
# ─────────────────────────────────────────────

@app.get("/indicadores/{ticker}", tags=["Análisis"])
def obtener_indicadores(ticker: str):
    """Indicadores técnicos SMA, EMA, RSI, MACD. Se implementará en la Fase 4."""
    ticker = ticker.upper()
    if ticker not in ACTIVOS:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado")
    return {
        "ticker":                  ticker,
        "indicadores_disponibles": ["SMA_20", "SMA_50", "EMA_20", "RSI_14", "MACD", "Bollinger"],
        "nota":                    "Valores reales en Fase 4",
    }


# ─────────────────────────────────────────────
# ENDPOINT 6: VaR — POST /var
# ─────────────────────────────────────────────

@app.post("/var", tags=["Riesgo"])
def calcular_var(portafolio: PortafolioRequest):
    """VaR y CVaR del portafolio. Se implementará en la Fase 5."""
    return {
        "tickers":         portafolio.tickers,
        "pesos":           portafolio.pesos,
        "nivel_confianza": portafolio.nivel_confianza,
        "nota":            "Cálculo real en Fase 5",
        "resultado_ejemplo": {
            "var_historico":   -0.0234,
            "cvar":            -0.0312,
            "interpretacion": (
                f"Con {portafolio.nivel_confianza*100:.0f}% de confianza, "
                "la pérdida máxima diaria no excedería el 2.34%"
            ),
        },
    }


# ─────────────────────────────────────────────
# ENDPOINT 7: CAPM — GET /capm
# ─────────────────────────────────────────────

@app.get("/capm", tags=["Riesgo"])
def calcular_capm(tickers: List[str] = Query(default=ACTIVOS)):
    """Beta y CAPM por activo. Se implementará en la Fase 6."""
    return {
        "benchmark": "S&P 500 (^GSPC)",
        "nota":      "Cálculo real en Fase 6",
        "resultado_ejemplo": {
            "AAPL":  {"beta": 1.23, "rendimiento_esperado_anual": "14.2%"},
            "MSFT":  {"beta": 1.15, "rendimiento_esperado_anual": "13.5%"},
            "GOOGL": {"beta": 1.08, "rendimiento_esperado_anual": "12.8%"},
        },
    }


# ─────────────────────────────────────────────
# ENDPOINT 8: FRONTERA EFICIENTE — POST /frontera-eficiente
# ─────────────────────────────────────────────

@app.post("/frontera-eficiente", tags=["Portafolio"])
def calcular_frontera(portafolio: PortafolioRequest):
    """Frontera eficiente de Markowitz. Se implementará en la Fase 6."""
    return {
        "tickers": portafolio.tickers,
        "nota":    "Frontera eficiente en Fase 6",
    }


# ─────────────────────────────────────────────
# ENDPOINT 9: ALERTAS — GET /alertas
# ─────────────────────────────────────────────

@app.get("/alertas", tags=["Señales"])
def obtener_alertas():
    """Señales de trading automatizadas. Se implementará en la Fase 7."""
    return {
        "total_alertas": 0,
        "alertas":       [],
        "nota":          "Señales reales en Fase 7",
    }


# ─────────────────────────────────────────────
# ENDPOINT 10: DATOS MACRO — GET /macro
# ─────────────────────────────────────────────

@app.get("/macro", tags=["Macro"])
def obtener_macro():
    """Datos macroeconómicos desde FRED. Se implementará en la Fase 7."""
    return {
        "fuente": "FRED - Federal Reserve Bank of St. Louis",
        "nota":   "Conexión real a FRED en Fase 7",
        "datos_ejemplo": {
            "tasa_libre_riesgo": 0.0525,
            "inflacion_anual":   0.033,
            "tasa_desempleo":    0.039,
        },
    }


# ─────────────────────────────────────────────
# ARRANCAR EL SERVIDOR
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
