"""
Configuración del proyecto: series del BCRP verificadas.
"""

FECHA_INICIO = "2003-1"
FECHA_FIN = "2026-8"

SERIE_OBJETIVO = {
    "codigo": "PN38705PM",
    "nombre": "ipc_indice",
    "descripcion": "IPC Lima Metropolitana, indice Dic.2021=100",
}

SERIES_PREDICTORAS = [
    {
        "codigo": "PN01234PM",
        "nombre": "tipo_cambio_venta",
        "descripcion": "Tipo de cambio dolar, promedio del periodo",
    },
    {
        "codigo": "PD12912AM",
        "nombre": "expectativas_inflacion_12m",
        "descripcion": "Expectativa de inflacion a 12 meses (encuesta)",
    },
]

API_BASE = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
FORMATO = "json"
IDIOMA = "esp"

DIR_RAW = "data/raw"
DIR_PROCESSED = "data/processed"