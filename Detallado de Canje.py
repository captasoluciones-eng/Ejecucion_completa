import pandas as pd
import gdown
from google.cloud import bigquery
import os

# -----------------------------
# Configuración de credenciales (MULTI-ENTORNO)
# -----------------------------
def configurar_credenciales():
    """Detecta el entorno y configura las credenciales apropiadas"""
    
    # 1. GitHub Actions (busca credenciales en el home del usuario)
    github_creds = os.path.expanduser("~/gcp_credentials.json")
    if os.path.exists(github_creds):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = github_creds
        print("✅ Usando credenciales de GitHub Actions")
        return
    
    # 2. PC Local (Windows)
    local_creds = r"C:\PyScripts\lookerstudio-consolidacion-c10dd284ce9d.json"
    if os.path.exists(local_creds):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_creds
        print("✅ Usando credenciales locales")
        return
    
    # 3. Google Colab (usa autenticación nativa)
    try:
        from google.colab import auth
        auth.authenticate_user()
        print("✅ Usando autenticación de Google Colab")
        return
    except ImportError:
        pass
    
    # Si no encontró ninguna credencial
    print("❌ No se encontraron credenciales de GCP")
    raise EnvironmentError("No se pudo configurar la autenticación con Google Cloud")

# Configurar credenciales según el entorno
configurar_credenciales()

# -----------------------------
# Configuración de parámetros
# -----------------------------
file_id = "1RMOvX2uvcTKhH4_0H7FSCM-P9t3PJ8GD"
url_csv = f"https://drive.google.com/uc?id={file_id}"
archivo_csv = "DetalladoCanje.csv"

proyecto_bq = "lookerstudio-consolidacion"
dataset_bq = "DatosLooker_USC_V2"
tabla_bq = "DetalladoDeCanje"

# -----------------------------
# 1. Descargar CSV
# -----------------------------
print("📥 Descargando CSV...")
try:
    gdown.download(url_csv, archivo_csv, quiet=False)
    print(f"✅ Archivo descargado: {archivo_csv}")
except Exception as e:
    print(f"❌ Error al descargar el archivo: {e}")
    exit(1)

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
