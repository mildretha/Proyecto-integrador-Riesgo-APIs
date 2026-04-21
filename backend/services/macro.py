import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
from services.datos import descargar_precios, ACTIVOS_INFO
from services.indicadores import (
    calcular_rsi,
    calcular_ema,
    calcular_macd,
    calcular_bollinger,
    calcular_estocastico,
)


def _limpiar(v):
    """Convierte tipos numpy a tipos nativos de Python."""
    if v is None:
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    if isinstance(v, float):
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    return v


# ─────────────────────────────────────────────
# SEÑALES Y ALERTAS DE TRADING
# ─────────────────────────────────────────────

def generar_alertas_portafolio(
    tickers: List[str] = None,
    fecha_inicio: str = "2023-01-01",
) -> dict:
    """
    Genera señales automáticas de compra/venta para todos los activos
    basadas en múltiples indicadores técnicos.

    Lógica de señales:
    - RSI < 30         → COMPRA fuerte (sobrevendido)
    - RSI > 70         → VENTA fuerte  (sobrecomprado)
    - MACD > Señal     → COMPRA (tendencia alcista)
    - MACD < Señal     → VENTA  (tendencia bajista)
    - Precio < BB inf  → COMPRA (precio bajo banda)
    - Precio > BB sup  → VENTA  (precio sobre banda)
    - EMA20 > EMA50    → COMPRA (golden cross corto plazo)
    - EMA20 < EMA50    → VENTA  (death cross corto plazo)
    - Estocástico < 20 → COMPRA (sobrevendido)
    - Estocástico > 80 → VENTA  (sobrecomprado)
    """
    if tickers is None:
        tickers = list(ACTIVOS_INFO.keys())

    todas_alertas = []
    resumen_por_ticker = {}

    for ticker in tickers:
        try:
            df = descargar_precios(ticker, fecha_inicio)
            precios = df["cierre"]
            alertas_ticker = []

            # ── Calcular indicadores del último día ───────────────────────────
            rsi    = calcular_rsi(precios)
            ema20  = calcular_ema(precios, 20)
            ema50  = calcular_ema(precios, 50)
            macd_df = calcular_macd(precios)
            boll_df = calcular_bollinger(precios)
            esto_df = calcular_estocastico(df)

            # Último valor de cada indicador
            rsi_actual    = _limpiar(rsi.iloc[-1])
            ema20_actual  = _limpiar(ema20.iloc[-1])
            ema50_actual  = _limpiar(ema50.iloc[-1])
            macd_actual   = _limpiar(macd_df["macd"].iloc[-1])
            señal_actual  = _limpiar(macd_df["macd_señal"].iloc[-1])
            boll_sup      = _limpiar(boll_df["boll_superior"].iloc[-1])
            boll_inf      = _limpiar(boll_df["boll_inferior"].iloc[-1])
            precio_actual = _limpiar(precios.iloc[-1])
            esto_k        = _limpiar(esto_df["esto_k"].iloc[-1])

            # ── Señal RSI ─────────────────────────────────────────────────────
            if rsi_actual is not None:
                if rsi_actual < 30:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "RSI", "FUERTE",
                        f"RSI={rsi_actual:.1f} — sobrevendido (< 30)",
                        rsi_actual
                    ))
                elif rsi_actual < 40:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "RSI", "DÉBIL",
                        f"RSI={rsi_actual:.1f} — acercándose a zona de compra",
                        rsi_actual
                    ))
                elif rsi_actual > 70:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "RSI", "FUERTE",
                        f"RSI={rsi_actual:.1f} — sobrecomprado (> 70)",
                        rsi_actual
                    ))
                elif rsi_actual > 60:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "RSI", "DÉBIL",
                        f"RSI={rsi_actual:.1f} — acercándose a zona de venta",
                        rsi_actual
                    ))

            # ── Señal MACD ────────────────────────────────────────────────────
            if macd_actual is not None and señal_actual is not None:
                diferencia = round(macd_actual - señal_actual, 4)
                if macd_actual > señal_actual:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "MACD", "MODERADA",
                        f"MACD ({macd_actual:.4f}) sobre línea señal ({señal_actual:.4f})",
                        diferencia
                    ))
                else:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "MACD", "MODERADA",
                        f"MACD ({macd_actual:.4f}) bajo línea señal ({señal_actual:.4f})",
                        diferencia
                    ))

            # ── Señal Bollinger ───────────────────────────────────────────────
            if precio_actual and boll_sup and boll_inf:
                if precio_actual > boll_sup:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "Bollinger", "FUERTE",
                        f"Precio ({precio_actual:.2f}) sobre banda superior ({boll_sup:.2f})",
                        precio_actual
                    ))
                elif precio_actual < boll_inf:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "Bollinger", "FUERTE",
                        f"Precio ({precio_actual:.2f}) bajo banda inferior ({boll_inf:.2f})",
                        precio_actual
                    ))

            # ── Señal EMA (Golden/Death Cross) ────────────────────────────────
            if ema20_actual and ema50_actual:
                if ema20_actual > ema50_actual:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "EMA_Cross", "MODERADA",
                        f"EMA20 ({ema20_actual:.2f}) sobre EMA50 ({ema50_actual:.2f}) — tendencia alcista",
                        ema20_actual - ema50_actual
                    ))
                else:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "EMA_Cross", "MODERADA",
                        f"EMA20 ({ema20_actual:.2f}) bajo EMA50 ({ema50_actual:.2f}) — tendencia bajista",
                        ema20_actual - ema50_actual
                    ))

            # ── Señal Estocástico ─────────────────────────────────────────────
            if esto_k is not None:
                if esto_k < 20:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "Estocástico", "FUERTE",
                        f"%K={esto_k:.1f} — zona de sobrevendido (< 20)",
                        esto_k
                    ))
                elif esto_k > 80:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "Estocástico", "FUERTE",
                        f"%K={esto_k:.1f} — zona de sobrecomprado (> 80)",
                        esto_k
                    ))

            # ── Resumen por ticker ────────────────────────────────────────────
            compras = sum(1 for a in alertas_ticker if a["tipo"] == "COMPRA")
            ventas  = sum(1 for a in alertas_ticker if a["tipo"] == "VENTA")
            señal_neta = "NEUTRAL"
            if compras > ventas + 1:
                señal_neta = "COMPRA"
            elif ventas > compras + 1:
                señal_neta = "VENTA"

            resumen_por_ticker[ticker] = {
                "precio_actual":  precio_actual,
                "rsi_actual":     rsi_actual,
                "señal_neta":     señal_neta,
                "alertas_compra": compras,
                "alertas_venta":  ventas,
                "fecha_analisis": datetime.today().strftime("%Y-%m-%d"),
            }

            todas_alertas.extend(alertas_ticker)

        except Exception as e:
            resumen_por_ticker[ticker] = {"error": str(e)}

    # Ordenar alertas: FUERTE primero
    prioridad = {"FUERTE": 0, "MODERADA": 1, "DÉBIL": 2}
    todas_alertas.sort(key=lambda x: prioridad.get(x.get("fuerza", "DÉBIL"), 2))

    return {
        "fecha_analisis":  datetime.today().strftime("%Y-%m-%d %H:%M"),
        "tickers_analizados": tickers,
        "total_alertas":   len(todas_alertas),
        "alertas_compra":  sum(1 for a in todas_alertas if a["tipo"] == "COMPRA"),
        "alertas_venta":   sum(1 for a in todas_alertas if a["tipo"] == "VENTA"),
        "resumen":         resumen_por_ticker,
        "alertas":         todas_alertas,
    }


def _crear_alerta(
    ticker: str,
    tipo: str,
    indicador: str,
    fuerza: str,
    descripcion: str,
    valor,
) -> dict:
    """Crea un diccionario de alerta estandarizado."""
    return {
        "ticker":      ticker,
        "tipo":        tipo,           # COMPRA o VENTA
        "indicador":   indicador,      # RSI, MACD, Bollinger, etc.
        "fuerza":      fuerza,         # FUERTE, MODERADA, DÉBIL
        "descripcion": descripcion,
        "valor":       _limpiar(valor),
        "fecha":       datetime.today().strftime("%Y-%m-%d"),
    }


# ─────────────────────────────────────────────
# DATOS MACROECONÓMICOS — FRED API
# ─────────────────────────────────────────────

# Series FRED más importantes para análisis de riesgo
SERIES_FRED = {
    "DGS3MO":  "Tasa libre de riesgo (T-Bills 3 meses)",
    "DGS10":   "Tasa del Tesoro a 10 años",
    "CPIAUCSL": "Índice de Precios al Consumidor (CPI)",
    "UNRATE":  "Tasa de desempleo",
    "GDP":     "Producto Interno Bruto",
    "FEDFUNDS": "Tasa de fondos federales",
    "VIXCLS":  "Índice VIX (volatilidad del mercado)",
    "SP500":   "S&P 500",
}



def obtener_datos_fred(
    api_key: Optional[str] = None,
    series: List[str] = None,
) -> dict:
    """
    Obtiene indicadores macroeconómicos desde FRED.
    CPI se calcula como variación anual (no índice acumulado).
    """
    if series is None:
        series = ["DGS3MO", "DGS10", "CPIAUCSL", "UNRATE", "FEDFUNDS", "VIXCLS"]

    if not api_key:
        return _datos_fred_ejemplo()

    resultados = {}
    errores = []

    for serie_id in series:
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            # CPI sin observation_start para garantizar 13+ observaciones mensuales
            if serie_id == "CPIAUCSL":
                params = {
                    "series_id":  serie_id,
                    "api_key":    api_key,
                    "file_type":  "json",
                    "sort_order": "desc",
                    "limit":      16,
                }
            else:
                params = {
                    "series_id":        serie_id,
                    "api_key":          api_key,
                    "file_type":        "json",
                    "sort_order":       "desc",
                    "limit":            5,
                    "observation_start": (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d"),
                }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            observaciones = [o for o in data.get("observations", []) if o["value"] != "."]
            if not observaciones:
                continue

            if serie_id == "CPIAUCSL":
                # Variación anual: (mes actual / mes hace 12 meses - 1) * 100
                if len(observaciones) >= 13:
                    valor_reciente = round((float(observaciones[0]["value"]) / float(observaciones[12]["value"]) - 1) * 100, 2)
                else:
                    valor_reciente = None
            else:
                valor_reciente = float(observaciones[0]["value"])

            fecha_reciente = observaciones[0]["date"]
            interpretacion = _interpretar_indicador(serie_id, valor_reciente)

            resultados[serie_id] = {
                "nombre":         SERIES_FRED.get(serie_id, serie_id),
                "valor":          valor_reciente,
                "fecha":          fecha_reciente,
                "unidad":         "pts" if serie_id == "VIXCLS" else "%",
                "interpretacion": interpretacion,
                "aplica_a":       "Global" if serie_id == "VIXCLS" else "EE.UU.",
            }
        except Exception as e:
            errores.append(f"{serie_id}: {str(e)}")

    if errores:
        resultados["_errores"] = errores

    contexto = _generar_contexto_macro(resultados)

    return {
        "fuente":         "FRED - Federal Reserve Bank of St. Louis",
        "fecha_consulta": datetime.today().strftime("%Y-%m-%d %H:%M"),
        "nota_alcance":   (
            "Los indicadores DGS3MO, DGS10, CPI, UNRATE y FEDFUNDS corresponden a EE.UU. "
            "El VIX es global. Todos afectan el portafolio porque el T-Bills 3M se usa "
            "como tasa libre de riesgo (Rf) en el CAPM."
        ),
        "datos":          resultados,
        "contexto_macro": contexto,
    }


def _interpretar_indicador(serie_id: str, valor) -> str:
    """Interpretación en texto para cada indicador según su valor actual."""
    if valor is None:
        return "Sin datos disponibles."
    if serie_id == "DGS3MO":
        if valor > 5:   return f"Tasa alta ({valor:.2f}%). Encarece el costo de capital. Usado como Rf en CAPM."
        elif valor > 3: return f"Tasa moderada ({valor:.2f}%). Referencia de inversión libre de riesgo para el CAPM."
        else:           return f"Tasa baja ({valor:.2f}%). Favorable para acciones — el costo de oportunidad es bajo."
    if serie_id == "DGS10":
        if valor > 4.5: return f"Rendimiento alto ({valor:.2f}%). Los bonos compiten con las acciones como activo."
        elif valor > 3: return f"Rendimiento moderado ({valor:.2f}%). Entorno neutral para acciones de crecimiento."
        else:           return f"Rendimiento bajo ({valor:.2f}%). Favorable para valoraciones de acciones de crecimiento."
    if serie_id == "CPIAUCSL":
        if valor > 5:   return f"Inflación alta ({valor:.2f}% anual). La Fed probablemente mantendrá tasas elevadas."
        elif valor > 2.5: return f"Inflación moderada ({valor:.2f}% anual). Por encima del objetivo del 2% de la Fed."
        else:           return f"Inflación controlada ({valor:.2f}% anual). Cerca del objetivo del 2% de la Fed."
    if serie_id == "UNRATE":
        if valor > 6:   return f"Desempleo alto ({valor:.2f}%). Señal de debilidad económica."
        elif valor > 4.5: return f"Desempleo moderado ({valor:.2f}%). Mercado laboral en recuperación."
        else:           return f"Desempleo bajo ({valor:.2f}%). Economía sólida — favorable para resultados corporativos."
    if serie_id == "FEDFUNDS":
        if valor > 5:   return f"Tasa Fed alta ({valor:.2f}%). Política restrictiva — presiona valuaciones de acciones."
        elif valor > 3: return f"Tasa Fed moderada ({valor:.2f}%). Política monetaria neutral."
        else:           return f"Tasa Fed baja ({valor:.2f}%). Política expansiva — impulsa mercados de acciones."
    if serie_id == "VIXCLS":
        if valor > 30:  return f"VIX alto ({valor:.1f}). Pánico en el mercado — el VaR puede subestimar el riesgo real."
        elif valor > 20: return f"VIX moderado ({valor:.1f}). Incertidumbre en el mercado — precaución recomendada."
        else:           return f"VIX bajo ({valor:.1f}). Mercado tranquilo — modo risk-on, favorable para acciones."
    return "—"


def _generar_contexto_macro(datos: dict) -> dict:
    """Genera contexto dinámico con los valores reales de FRED."""
    fed    = datos.get("FEDFUNDS", {}).get("valor")
    cpi    = datos.get("CPIAUCSL", {}).get("valor")
    vix    = datos.get("VIXCLS",   {}).get("valor")
    unrate = datos.get("UNRATE",   {}).get("valor")
    rf     = datos.get("DGS3MO",   {}).get("valor")

    partes = []
    if fed and cpi:
        partes.append(
            f"La Fed mantiene su tasa en {fed:.2f}% con inflación en {cpi:.2f}% anual. "
            f"La tasa real (Fed - CPI) es {fed - cpi:.2f}%."
        )
    if vix:
        estado = "tranquilo" if vix < 20 else "con incertidumbre" if vix < 30 else "en pánico"
        partes.append(f"El mercado está {estado} — VIX en {vix:.1f} puntos.")
    if unrate:
        partes.append(f"El desempleo en EE.UU. se ubica en {unrate:.1f}%.")

    descripcion = " ".join(partes) if partes else "Datos macroeconómicos actualizados."

    impacto = []
    if rf:
        impacto.append(f"Rf en CAPM = {rf:.2f}% (T-Bills 3M) — afecta el rendimiento esperado de todos los activos")
    if vix and vix > 25:
        impacto.append("VIX elevado → el VaR histórico puede subestimar el riesgo real del portafolio")
    elif vix:
        impacto.append("VIX bajo → el VaR histórico es representativo del riesgo actual")
    if fed and fed > 4:
        impacto.append("Tasas altas → acciones de tecnología (alto P/E) son más vulnerables a correcciones")
    if cpi and cpi > 3:
        impacto.append(f"Inflación ({cpi:.1f}%) por encima del 2% → la Fed podría mantener tasas restrictivas")

    return {
        "descripcion":        descripcion,
        "impacto_portafolio": impacto if impacto else ["Entorno macroeconómico estable."],
    }


def _datos_fred_ejemplo() -> dict:
    """Datos de ejemplo cuando no hay API key."""
    datos = {
        "DGS3MO":   {"nombre":"Tasa libre de riesgo (T-Bills 3M)", "valor":4.35, "fecha":"2025-04-01", "unidad":"%", "aplica_a":"EE.UU.", "interpretacion":"Tasa moderada (4.35%). Referencia de inversión libre de riesgo para el CAPM."},
        "DGS10":    {"nombre":"Tasa del Tesoro a 10 años",          "valor":4.26, "fecha":"2025-04-01", "unidad":"%", "aplica_a":"EE.UU.", "interpretacion":"Rendimiento alto (4.26%). Los bonos compiten con las acciones como activo."},
        "CPIAUCSL": {"nombre":"Inflación anual (CPI)",              "valor":2.80, "fecha":"2025-03-01", "unidad":"%", "aplica_a":"EE.UU.", "interpretacion":"Inflación moderada (2.80% anual). Por encima del objetivo del 2% de la Fed."},
        "UNRATE":   {"nombre":"Tasa de desempleo",                  "valor":4.20, "fecha":"2025-03-01", "unidad":"%", "aplica_a":"EE.UU.", "interpretacion":"Desempleo moderado (4.20%). Mercado laboral en recuperación."},
        "FEDFUNDS": {"nombre":"Tasa de fondos federales (Fed)",     "valor":4.33, "fecha":"2025-03-01", "unidad":"%", "aplica_a":"EE.UU.", "interpretacion":"Tasa Fed moderada (4.33%). Política monetaria neutral."},
        "VIXCLS":   {"nombre":"VIX — Índice de volatilidad",        "valor":18.87,"fecha":"2025-04-20", "unidad":"pts","aplica_a":"Global","interpretacion":"VIX bajo (18.9). Mercado tranquilo — modo risk-on, favorable para acciones."},
    }
    contexto = _generar_contexto_macro(datos)
    return {
        "fuente":         "FRED - Federal Reserve Bank of St. Louis",
        "fecha_consulta": datetime.today().strftime("%Y-%m-%d %H:%M"),
        "nota":           "Datos de referencia. Configura FRED_API_KEY en .env para datos en tiempo real.",
        "nota_alcance":   (
            "Los indicadores DGS3MO, DGS10, CPI, UNRATE y FEDFUNDS corresponden a EE.UU. "
            "El VIX es global. Todos afectan el portafolio porque el T-Bills 3M se usa "
            "como tasa libre de riesgo (Rf) en el CAPM."
        ),
        "datos":          datos,
        "contexto_macro": contexto,
    }
