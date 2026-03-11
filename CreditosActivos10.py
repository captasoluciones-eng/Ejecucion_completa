import google.auth
from google.cloud import bigquery
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
        return github_creds

    local_creds = r"C:\PyScripts\lookerstudio-consolidacion-c10dd284ce9d.json"
    if os.path.exists(local_creds):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_creds
        print("✅ Usando credenciales locales")
        return local_creds

    try:
        from google.colab import auth
        auth.authenticate_user()
        print("✅ Usando autenticación de Google Colab")
        return None
    except ImportError:
        pass

    raise EnvironmentError("❌ No se pudo configurar la autenticación con Google Cloud")

creds_path = configurar_credenciales()

# -----------------------------
# Configuración de parámetros
# -----------------------------
file_id = "1Tu1__f_w-s7V2UCyP5RFVNRMFtnJ0zfE"
drive_uri = f"https://drive.google.com/open?id={file_id}"

proyecto_bq = "lookerstudio-consolidacion"
dataset_bq = "DatosLooker_USC_V2"
tabla_bq = "Full"
tabla_destino = f"{proyecto_bq}.{dataset_bq}.{tabla_bq}"

# -----------------------------
# 1. Autenticación con scope de Drive
# -----------------------------
credentials = service_account.Credentials.from_service_account_file(
    creds_path,
    scopes=[
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/bigquery",
    ]
)

client = bigquery.Client(project=proyecto_bq, credentials=credentials)

# -----------------------------
# 2. Configurar tabla externa apuntando a Drive
# -----------------------------
external_config = bigquery.ExternalConfig("CSV")
external_config.source_uris = [drive_uri]
external_config.autodetect = True
external_config.options.skip_leading_rows = 1
external_config.options.allow_quoted_newlines = True
external_config.options.allow_jagged_rows = True
external_config.options.quote_character = '"'

# -----------------------------
# 3. Leer desde Drive y escribir directo en BigQuery
# -----------------------------
print("🚀 Leyendo desde Drive y cargando en BigQuery...")

job_config = bigquery.QueryJobConfig(
    table_definitions={"tabla_drive": external_config},
    destination=client.dataset(dataset_bq).table(tabla_bq),
    write_disposition="WRITE_TRUNCATE",
    create_disposition="CREATE_IF_NEEDED",
)

try:
    job = client.query("SELECT * FROM tabla_drive", job_config=job_config)
    job.result()
    tabla_final = client.get_table(tabla_destino)
    print(f"✅ Datos cargados correctamente en {tabla_destino}")
    print(f"   Filas: {tabla_final.num_rows:,} | Columnas: {len(tabla_final.schema)}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

print("🎉 Proceso completado exitosamente")
