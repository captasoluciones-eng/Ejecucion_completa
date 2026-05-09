import pandas as pd
from google.cloud import bigquery
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import os

# -----------------------------
# Configuración de credenciales (MULTI-ENTORNO)
# -----------------------------
def configurar_credenciales():
    github_creds = os.path.expanduser("~/gcp_credentials.json")
    if os.path.exists(github_creds):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = github_creds
        print("✅ Usando credenciales de GitHub Actions")
        return
    local_creds = r"C:\PyScripts\lookerstudio-consolidacion-c10dd284ce9d.json"
    if os.path.exists(local_creds):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_creds
        print("✅ Usando credenciales locales")
        return
    try:
        from google.colab import auth
        auth.authenticate_user()
        print("✅ Usando autenticación de Google Colab")
        return
    except ImportError:
        pass
    print("❌ No se encontraron credenciales de GCP")
    raise EnvironmentError("No se pudo configurar la autenticación con Google Cloud")

configurar_credenciales()

# -----------------------------
# Configuración de parámetros
# -----------------------------
file_id = "1Ftr23C6irRdSzcDy4tSBjK32o2vUlfE2"
archivo_csv = "EvaluacionFinancieraBitacora.csv"
proyecto_bq = "lookerstudio-consolidacion"
dataset_bq = "DatosLooker_USC_V2"
tabla_bq = "EvaluacionFinancieraBitacora"

# -----------------------------
# 1. Descargar CSV con API de Drive
# -----------------------------
def descargar_drive(file_id, destino):
    print("📥 Descargando CSV con API de Drive...")
    try:
        creds = service_account.Credentials.from_service_account_file(
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        service = build("drive", "v3", credentials=creds)
        request = service.files().get_media(fileId=file_id)
        with open(destino, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"  ⬇️ Descargando... {int(status.progress() * 100)}%")
        print(f"✅ Archivo descargado: {destino}")
    except Exception as e:
        print(f"❌ Error al descargar el archivo: {e}")
        exit(1)

descargar_drive(file_id, archivo_csv)

# -----------------------------
# 2. Leer CSV
# -----------------------------
print("📖 Leyendo CSV...")
try:
    df = pd.read_csv(archivo_csv, sep=",", on_bad_lines='skip', low_memory=False, encoding='utf-8-sig')
    print(f"✅ CSV leído correctamente. Filas: {len(df)} | Columnas: {len(df.columns)}")
except Exception as e:
    print(f"❌ Error al leer el CSV: {e}")
    exit(1)

# -----------------------------
# 3. Limpiar nombres de columnas
# -----------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
)
print("🧹 Nombres de columnas limpiados")

# -----------------------------
# 4. Subir a BigQuery
# -----------------------------
print("🚀 Subiendo datos a BigQuery...")
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
    print(f"❌ Error al subir a BigQuery: {e}")
    exit(1)

print("🎉 Proceso completado exitosamente")
