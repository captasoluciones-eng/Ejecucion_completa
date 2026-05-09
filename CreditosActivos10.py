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
file_id = "1Tu1__f_w-s7V2UCyP5RFVNRMFtnJ0zfE"
archivo_csv = "CreditosActivos10_descargado.csv"
proyecto_bq = "lookerstudio-consolidacion"
dataset_bq = "DatosLooker_USC_V2"
tabla_bq = "Full"

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
# 2. Limpiar nombres de columnas
# -----------------------------
def limpiar_chunk(chunk):
    chunk.columns = (
        chunk.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
    )
    for col in chunk.select_dtypes(include='object').columns:
        chunk[col] = chunk[col].str.strip()
    return chunk

# -----------------------------
# 3. Leer y subir CSV por chunks a BigQuery
# -----------------------------
print("📖 Leyendo y subiendo CSV por chunks...")
client = bigquery.Client(project=proyecto_bq)
tabla_destino = f"{proyecto_bq}.{dataset_bq}.{tabla_bq}"
CHUNK_SIZE = 100_000
primer_chunk = True
total_filas = 0
schema_tabla = None

try:
    chunks = pd.read_csv(
        archivo_csv,
        sep=",",
        on_bad_lines='skip',
        low_memory=False,
        encoding='utf-8-sig',
        chunksize=CHUNK_SIZE
    )
    for i, chunk in enumerate(chunks):
        chunk = limpiar_chunk(chunk)
        if primer_chunk:
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_TRUNCATE",
                autodetect=True
            )
            job = client.load_table_from_dataframe(chunk, tabla_destino, job_config=job_config)
            job.result()
            schema_tabla = client.get_table(tabla_destino).schema
            primer_chunk = False
        else:
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",
                schema=schema_tabla
            )
            job = client.load_table_from_dataframe(chunk, tabla_destino, job_config=job_config)
            job.result()

        total_filas += len(chunk)
        print(f"✅ Chunk {i+1} subido | Filas acumuladas: {total_filas:,}")

    print(f"🎉 Total filas subidas: {total_filas:,}")

except Exception as e:
    print(f"❌ Error en chunk {i+1}: {e}")
    exit(1)
