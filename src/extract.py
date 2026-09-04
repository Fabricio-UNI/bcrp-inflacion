import requests
import json

url = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PN38705PM-PN01234PM-PD12912AM/json/2003-1/2026-8/esp"

r = requests.get(url, timeout=30)
datos = r.json()
periodos = datos.get("periods", [])

with open("data/raw/bcrp_mensual.json", "w", encoding="utf-8") as f:
    json.dump(datos, f, ensure_ascii=False, indent=2)

print(f"Guardado: {len(periodos)} periodos")