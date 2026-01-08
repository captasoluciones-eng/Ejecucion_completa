import pandas as pd
import gdown
from google.cloud import bigquery
import os

# -----------------------------
# Configuración
# -----------------------------
# ❌ QUITAR ESTA LÍNEA (ya la configura GitHub Actions):
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\PyScripts\lookerstudio-consolidacion-c10dd284ce9d.json"

# ID de archivo de Google Drive
file_id = "1Ftr23C6irRdSzcDy4tSBjK32o2vUlfE2"
url_csv = f"https://drive.google.com/uc?id={file_id}"

archivo_csv = "EvaluacionFinancieraBitacora.csv"

proyecto_bq = "lookerstudio-consolidacion"
dataset_bq = "DatosLooker_USC_V2"
tabla_bq = "EvaluacionFinancieraBitacora"

# -----------------------------
# 1. Descargar CSV
# -----------------------------
print("Descargando CSV...")
gdown.download(url_csv, archivo_csv, quiet=False)

# -----------------------------
# 2. Leer CSV (forzando separador)
# -----------------------------
print("Leyendo CSV...")
try:
    df = pd.read_csv(archivo_csv, sep=",", on_bad_lines='skip', low_memory=False, encoding='utf-8-sig')
    print(f"CSV leído correctamente. Filas: {len(df)}  Columnas: {len(df.columns)}")
except Exception as e:
    print("Error al leer el CSV:", e)
    exit()

# -----------------------------
# 3. Limpiar nombres de columnas para BigQuery
# -----------------------------
df.columns = (
    df.columns
    .str.strip()                           # quitar espacios
    .str.replace(" ", "_")                 # espacios por guiones bajos
    .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)  # quitar caracteres no válidos
)

# -----------------------------
# 4. Subir a BigQuery
# -----------------------------
print("Subiendo datos a BigQuery...")
client = bigquery.Client(project=proyecto_bq)

tabla_destino = f"{proyecto_bq}.{dataset_bq}.{tabla_bq}"

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",
    autodetect=True
)

try:
    job = client.load_table_from_dataframe(df, tabla_destino, job_config=job_config)
    job.result()
    print(f"✅ Datos subidos correctamente a {tabla_destino}")
except Exception as e:
    print("❌ Error al subir a BigQuery:", e)