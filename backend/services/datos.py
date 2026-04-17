import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Optional, List

# Información fija de los 5 activos del portafolio
ACTIVOS_INFO = {
    "AAPL":  {"nombre": "Apple Inc.",       "sector": "Tecnología", "moneda": "USD"},
    "MSFT":  {"nombre": "Microsoft Corp.",  "sector": "Tecnología", "moneda": "USD"},
    "GOOGL": {"nombre": "Alphabet Inc.",    "sector": "Tecnología", "moneda": "USD"},
    "AMZN":  {"nombre": "Amazon.com Inc.",  "sector": "Consumo",    "moneda": "USD"},
    "TSLA":  {"nombre": "Tesla Inc.",       "sector": "Automotriz", "moneda": "USD"},
}


def obtener_fecha_fin(fecha_fin: Optional[str]) -> str:
    """Si no se da fecha_fin, retorna el día de hoy."""
    if fecha_fin is None:
        return datetime.today().strftime("%Y-%m-%d")
    return fecha_fin


def descargar_precios(
    ticker: str,
    fecha_inicio: str = "2022-01-01",
    fecha_fin: Optional[str] = None,
) -> pd.DataFrame:
    """
    Descarga precios históricos de Yahoo Finance para un ticker.
    Retorna un DataFrame con: fecha, apertura, maximo, minimo, cierre, volumen.
    """
    fecha_fin = obtener_fecha_fin(fecha_fin)

    try:
        # Usamos yf.Ticker().history() — método más estable
        ticker_obj = yf.Ticker(ticker)
        datos = ticker_obj.history(
            start=fecha_inicio,
            end=fecha_fin,
            auto_adjust=True,   # ajusta por splits y dividendos
        )
    except Exception as e:
        raise ValueError(f"Error al conectar con Yahoo Finance: {e}")

    if datos is None or datos.empty:
        raise ValueError(
            f"No se encontraron datos para '{ticker}' "
            f"entre {fecha_inicio} y {fecha_fin}. "
            "Verifica que el ticker sea válido."
        )

    # Renombrar columnas al español
    datos = datos.rename(columns={
        "Open":   "apertura",
        "High":   "maximo",
        "Low":    "minimo",
        "Close":  "cierre",
        "Volume": "volumen",
    })

    # Convertir el índice (fecha) en columna y limpiar timezone
    datos = datos.reset_index()
    datos = datos.rename(columns={"Date": "fecha", "Datetime": "fecha"})
    datos["fecha"] = pd.to_datetime(datos["fecha"]).dt.tz_localize(None)
    datos["fecha"] = datos["fecha"].dt.strftime("%Y-%m-%d")

    # Quedarnos solo con las columnas necesarias
    columnas = ["fecha", "apertura", "maximo", "minimo", "cierre", "volumen"]
    columnas_presentes = [c for c in columnas if c in datos.columns]
    datos = datos[columnas_presentes].copy()

    # Redondear precios a 2 decimales
    for col in ["apertura", "maximo", "minimo", "cierre"]:
        if col in datos.columns:
            datos[col] = datos[col].round(2)

    return datos


def descargar_multiples_precios(
    tickers: List[str],
    fecha_inicio: str = "2022-01-01",
    fecha_fin: Optional[str] = None,
) -> dict:
    """
    Descarga precios de varios tickers a la vez.
    Retorna: {"AAPL": DataFrame, "MSFT": DataFrame, ...}
    """
    fecha_fin = obtener_fecha_fin(fecha_fin)
    resultado = {}
    for ticker in tickers:
        try:
            resultado[ticker] = descargar_precios(ticker, fecha_inicio, fecha_fin)
        except ValueError as e:
            resultado[ticker] = None
            print(f"Advertencia: {e}")
    return resultado


def obtener_precio_actual(ticker: str) -> dict:
    """
    Obtiene precio más reciente y variación del día.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        # fast_info es más rápido que .info completo
        fi = ticker_obj.fast_info
        precio_actual   = round(float(fi.last_price), 2)
        precio_apertura = round(float(fi.open), 2)
        variacion_dia   = round(precio_actual - precio_apertura, 2)
        variacion_pct   = round((variacion_dia / precio_apertura) * 100, 2) if precio_apertura else 0.0
        return {
            "precio_actual":   precio_actual,
            "precio_apertura": precio_apertura,
            "variacion_dia":   variacion_dia,
            "variacion_pct":   variacion_pct,
        }
    except Exception:
        return {
            "precio_actual":   None,
            "precio_apertura": None,
            "variacion_dia":   None,
            "variacion_pct":   None,
        }


def obtener_info_activo(ticker: str) -> dict:
    """Información general + precio actual de un activo."""
    info_base   = ACTIVOS_INFO.get(ticker, {"nombre": ticker, "sector": "N/A", "moneda": "USD"})
    precio_info = obtener_precio_actual(ticker)
    return {
        "ticker":  ticker,
        "nombre":  info_base["nombre"],
        "sector":  info_base["sector"],
        "moneda":  info_base["moneda"],
        **precio_info,
    }
