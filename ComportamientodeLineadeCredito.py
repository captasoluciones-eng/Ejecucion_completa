import google.auth
from google.cloud import bigquery
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
file_id = "1TP2sdPjpy5jdP8xmdwmIv9HbCkKz92gn"
drive_uri = f"https://drive.google.com/open?id={file_id}"
proyecto_bq = "lookerstudio-consolidacion"
dataset_bq = "DatosLooker_USC_V2"
tabla_bq = "ComportamientodeLineadeCredito"
tabla_destino = f"{proyecto_bq}.{dataset_bq}.{tabla_bq}"

# -----------------------------
# 1. Autenticación con scopes de Drive + BigQuery
# -----------------------------
credentials, _ = google.auth.load_credentials_from_file(
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
    scopes=[
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/bigquery",
    ]
)
client = bigquery.Client(project=proyecto_bq, credentials=credentials)

# -----------------------------
# 2. Definir tabla externa apuntando a Drive
# -----------------------------
external_config = bigquery.ExternalConfig("CSV")
external_config.source_uris = [drive_uri]
external_config.autodetect = True
external_config.options.skip_leading_rows = 1
external_config.options.allow_quoted_newlines = True
external_config.options.allow_jagged_rows = True
external_config.options.quote_character = '"'

# -----------------------------
# 3. Query con limpieza de columnas con formato "$ valor"
# -----------------------------
query = """
SELECT
    Codigo,
    NombreCompleto,
    SAFE_CAST(REGEXP_REPLACE(CAST(Limite                     AS STRING), r'[^0-9.]', '') AS FLOAT64) AS Limite,
    SAFE_CAST(REGEXP_REPLACE(CAST(LineaCredito               AS STRING), r'[^0-9.]', '') AS FLOAT64) AS LineaCredito,
    SAFE_CAST(REGEXP_REPLACE(CAST(CapacidadPago              AS STRING), r'[^0-9.]', '') AS FLOAT64) AS CapacidadPago,
    SAFE_CAST(REGEXP_REPLACE(CAST(LimiteCreditoPersonal      AS STRING), r'[^0-9.]', '') AS FLOAT64) AS LimiteCreditoPersonal,
    SAFE_CAST(REGEXP_REPLACE(CAST(FactorLiberacionDisponible AS STRING), r'[^0-9.]', '') AS FLOAT64) AS FactorLiberacionDisponible,
    SAFE_CAST(REGEXP_REPLACE(CAST(MontoLimiteExtendido       AS STRING), r'[^0-9.]', '') AS FLOAT64) AS MontoLimiteExtendido,
    SAFE_CAST(REGEXP_REPLACE(CAST(Proyectado                 AS STRING), r'[^0-9.]', '') AS FLOAT64) AS Proyectado,
    Comentario,
    EstatusEnlace,
    TipoBaja,
    UsuarioModifico,
    FechaModificacion,
    FechaExpiracion,
    FechaEvaluacion,
    FechaAlta,
    ExtenderLimite,
    Sucursal,
    SinAval,
    Nivel,
    Coordinador,
    SaldoCapitalActual,
    SaldoCapital,
    SaldoaLiberar,
    SaldoProximomesLiberar,
    DiasVencimiento,
    TempDisponible,
    CanjeActual1,
    MontoEnTransito,
    CanjeEnTransito,
    TipodeLiberacion1,
    CapitalAnticipado1,
    CapitalAnticipado,
    TipodeLiberacion,
    CanjeActual,
    DisponibleTotal,
    Disponible,
    EstatusLinea,
    Sugerencia
FROM tabla_drive
"""

# -----------------------------
# 4. Insertar en tabla nativa (WRITE_TRUNCATE)
# -----------------------------
print("📥 Leyendo desde Drive y cargando en BigQuery...")
job_config = bigquery.QueryJobConfig(
    table_definitions={"tabla_drive": external_config},
    destination=client.dataset(dataset_bq).table(tabla_bq),
    write_disposition="WRITE_TRUNCATE",
    create_disposition="CREATE_IF_NEEDED",
)
try:
    job = client.query(query, job_config=job_config)
    job.result()
    tabla_final = client.get_table(tabla_destino)
    print(f"✅ Datos cargados correctamente en {tabla_destino}")
    print(f"   Filas: {tabla_final.num_rows} | Columnas: {len(tabla_final.schema)}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

print("🎉 Proceso completado exitosamente")
