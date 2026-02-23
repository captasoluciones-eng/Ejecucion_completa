# Analisis de quitas y convenios -> consolida, calcula TIR y sube a BigQuery
# ✅ VERSIÓN CON DESCARGA AUTOMÁTICA DESDE GOOGLE DRIVE

import os, re
import pandas as pd
import numpy as np
import gdown  # ✅ Habilitado para descargar archivos
from google.cloud import bigquery
from scipy.optimize import brentq, minimize_scalar
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ----------------------------
# CONFIG
# ----------------------------
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\PyScripts\lookerstudio-consolidacion-c10dd284ce9d.json"
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

# ----------------------------
# Archivos Drive y locales
# ----------------------------
ARCHIVOS_DRIVE = {
    "abono": "1hl0_jOf4equItPWr-_z5G80iyeAJd1hb",
    "bloques": "19Jlhet-osFE8rWQ8rzDetT21G_pNzOhh",
    "canje": "1Bz_ZtLrhMQ1HXA3TSEYo1kridSfZb05O"
}
ARCHIVOS_LOCAL = {k: f"{k}.csv" for k in ARCHIVOS_DRIVE}

PROYECTO_BQ = "lookerstudio-consolidacion"
DATASET_BQ = "DatosLooker_USC_V2"
TABLA_CONSOLIDADO = "ConsolidadoConvenios"

# ----------------------------
# ✅ DESCARGAR ARCHIVOS (SOBRESCRIBIR SIEMPRE)
# ----------------------------
print("⬇️ Descargando archivos desde Google Drive...")
print(f"📅 Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

for key, fid in ARCHIVOS_DRIVE.items():
    url = f"https://drive.google.com/uc?id={fid}"
    out = ARCHIVOS_LOCAL[key]
    
    # ✅ Eliminar archivo viejo si existe
    if os.path.exists(out):
        os.remove(out)
        print(f"  🗑️ {out} eliminado (descargando versión actualizada)")
    
    try:
        # ✅ Descargar con fuzzy=False para evitar problemas
        print(f"  ⏳ Descargando {key}.csv...")
        gdown.download(url, out, quiet=False, fuzzy=False)
        
        # Verificar que se descargó correctamente
        if os.path.exists(out) and os.path.getsize(out) > 0:
            size_mb = os.path.getsize(out) / (1024 * 1024)
            print(f"  ✅ {key}.csv descargado exitosamente ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ ERROR: {key}.csv no se descargó correctamente")
            raise Exception(f"Archivo {out} vacío o no existe")
            
    except Exception as e:
        print(f"  ❌ ERROR descargando {key}: {str(e)}")
        print(f"     URL: {url}")
        raise  # Detener el script si falla la descarga

# Verificar fechas de modificación
print("\n📅 Fechas de los archivos descargados:")
for key, filename in ARCHIVOS_LOCAL.items():
    if os.path.exists(filename):
        mtime = os.path.getmtime(filename)
        fecha = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {filename}: {fecha}")

# ----------------------------
# Lectura CSVs
# ----------------------------
print("\n📥 Leyendo CSVs...")
df_abono = pd.read_csv(ARCHIVOS_LOCAL["abono"], on_bad_lines='skip', encoding='utf-8-sig', low_memory=False)
df_bloques = pd.read_csv(ARCHIVOS_LOCAL["bloques"], on_bad_lines='skip', encoding='utf-8-sig', low_memory=False)
df_canje = pd.read_csv(ARCHIVOS_LOCAL["canje"], on_bad_lines='skip', encoding='utf-8-sig', low_memory=False)

print(f"  ✓ abono.csv: {len(df_abono):,} filas")
print(f"  ✓ bloques.csv: {len(df_bloques):,} filas")
print(f"  ✓ canje.csv: {len(df_canje):,} filas")

# ----------------------------
# Funciones auxiliares
# ----------------------------
def normalize_codigo_series(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

MESES = {"Ene":"01","Feb":"02","Mar":"03","Abr":"04","May":"05","Jun":"06",
         "Ago":"08","Sep":"09","Oct":"10","Nov":"11","Dic":"12"}

def normalize_period_value(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "" or s.lower() in ["nan","none","na","n/a"]:
        return None
    m = re.match(r"^(\d{4})[-/\. ]?(\d{1,2})$", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    s2 = re.sub(r"[\.\/]", "-", s)
    m2 = re.match(r"^(\d{4})[-_ ]?([A-Za-z]{3,})", s2)
    if m2:
        y = m2.group(1)
        mon = m2.group(2)[:3].title()
        if mon in MESES:
            return f"{y}-{MESES[mon]}"
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_period("M").strftime("%Y-%m")
    except:
        return None

def get_periodo_series(df, prefer_cols):
    for col in prefer_cols:
        if col in df.columns:
            ser = df[col].astype(str)
            normalized = ser.map(normalize_period_value)
            if normalized.notna().sum() > 0:
                return normalized
    for col in df.columns:
        if "fecha" in col.lower():
            ser = pd.to_datetime(df[col].astype(str), dayfirst=True, errors="coerce")
            if ser.notna().sum() > 0:
                return ser.dt.to_period("M").astype(str)
    return pd.Series([None]*len(df))

def first_value_safe(grupo, col):
    if col in grupo.columns:
        s = grupo[col].dropna()
        if len(s) > 0:
            return s.iloc[0]
    return None

# ----------------------------
# Función principal de TIR tipo Excel TIR.NO.PER
# ----------------------------
def calcular_xirr(grupo):
    """
    Calcula la TIR similar a Excel TIR.NO.PER:
    - Ignora ceros iniciales
    - Retorna 0 si no hay suficientes flujos positivos para generar TIR
    - Usa meses como periodo (DIAS360 / 30)
    """
    diffs = grupo[grupo["TipoDato"] == "Diferencia"].copy()
    diffs["PeriodoYMD"] = pd.to_datetime(diffs["PeriodoYMD"], errors="coerce")
    diffs = diffs.dropna(subset=["PeriodoYMD", "Monto"])
    diffs = diffs.sort_values("PeriodoYMD")

    flujos = diffs["Monto"].astype(float).tolist()
    fechas = diffs["PeriodoYMD"].tolist()

    # 🔹 Ignorar ceros iniciales como Excel
    for i, v in enumerate(flujos):
        if v != 0:
            flujos = flujos[i:]
            fechas = fechas[i:]
            break

    if len(flujos) < 2:
        return 0.0  # Excel devuelve 0 si no hay suficientes flujos

    # Validación: debe haber al menos un flujo positivo y uno negativo
    if not (any(v < 0 for v in flujos) and any(v > 0 for v in flujos)):
        return 0.0

    # XNPV tipo Excel usando meses como periodo
    def xnpv(rate, values, dates):
        t0 = dates[0]
        meses = [(d.year - t0.year) * 12 + (d.month - t0.month) for d in dates]
        return sum(v / (1 + rate) ** (m / 12.0) for v, m in zip(values, meses))

    # Intentar encontrar la raíz con Brentq
    try:
        tir = brentq(lambda r: xnpv(r, flujos, fechas), -0.99, 10)
    except ValueError:
        # Si no hay raíz positiva, retornar 0
        tir = 0.0
    except Exception:
        tir = 0.0

    # Limitar tir al rango [0, 200%] como máximo
    tir = max(0.0, tir)
    tir = min(2.0, tir)

    return tir

# ----------------------------
# Normalizaciones
# ----------------------------
print("\n🔄 Normalizando datos...")
for d in (df_abono, df_bloques, df_canje):
    if "Codigo" in d.columns:
        d["Codigo"] = normalize_codigo_series(d["Codigo"])

df_abono["PeriodoYM"] = get_periodo_series(df_abono, ["Periodo","FechaRegistro"])
df_canje["PeriodoYM"] = get_periodo_series(df_canje, ["Periodo","FechaInicioPago"])

# ----------------------------
# Mapeo NombreCompleto / Sucursal
# ----------------------------
print("🗺️ Creando mapeos...")
mapeo_list = []
for df_src in [df_canje, df_abono, df_bloques]:
    if "Codigo" in df_src.columns:
        cols = [c for c in ["Codigo","NombreCompleto","Sucursal"] if c in df_src.columns]
        if cols:
            mapeo_list.append(df_src[cols].drop_duplicates(subset=["Codigo"]))
df_mapa = (pd.concat(mapeo_list, ignore_index=True) if mapeo_list else pd.DataFrame(columns=["Codigo","NombreCompleto","Sucursal"]))
df_mapa = df_mapa.drop_duplicates(subset=["Codigo"], keep="first").set_index("Codigo")

# ----------------------------
# Agregaciones
# ----------------------------
print("➕ Agregando datos...")
if "MontoPagado" not in df_abono.columns:
    df_abono["MontoPagado"] = 0
ab_agg = df_abono.groupby(["Codigo","PeriodoYM"], as_index=False)["MontoPagado"].sum().rename(columns={"MontoPagado":"Abono_Sum"})

if "TotalCapital" not in df_canje.columns:
    df_canje["TotalCapital"] = 0
ca_agg = df_canje.groupby(["Codigo","PeriodoYM"], as_index=False)["TotalCapital"].sum().rename(columns={"TotalCapital":"Canje_Sum"})

merged = pd.merge(ab_agg, ca_agg, on=["Codigo","PeriodoYM"], how="outer")
merged["Abono_Sum"] = merged["Abono_Sum"].fillna(0.0)
merged["Canje_Sum"] = merged["Canje_Sum"].fillna(0.0)
merged["Diferencia"] = (merged["Canje_Sum"]*-1) + merged["Abono_Sum"]

# ----------------------------
# Construir tabla larga
# ----------------------------
print("📊 Construyendo tabla larga...")
rows = []
for _, r in merged.iterrows():
    codigo, periodo = r["Codigo"], r["PeriodoYM"]
    ab, ca, diff = float(r["Abono_Sum"]), float(r["Canje_Sum"]), float(r["Diferencia"])
    rows.append({"Codigo": codigo,"PeriodoYM": periodo,"TipoDato": "Abono","Monto": ab})
    rows.append({"Codigo": codigo,"PeriodoYM": periodo,"TipoDato": "Canje","Monto": ca})
    rows.append({"Codigo": codigo,"PeriodoYM": periodo,"TipoDato": "Diferencia","Monto": diff})
df_long = pd.DataFrame(rows)

# Añadir columnas adicionales
df_long["NombreCompleto"] = df_long["Codigo"].map(df_mapa["NombreCompleto"] if "NombreCompleto" in df_mapa.columns else pd.Series())
df_long["Sucursal"] = df_long["Codigo"].map(df_mapa["Sucursal"] if "Sucursal" in df_mapa.columns else pd.Series())
cols_bloques = [c for c in ["Codigo","Bloque","TotalCapitalTape","TotalCobranzaEstimada"] if c in df_bloques.columns]
df_long = df_long.merge(df_bloques[cols_bloques], on="Codigo", how="left")

# Crear PeriodoYMD
df_long["PeriodoYMD"] = pd.to_datetime(df_long["PeriodoYM"].astype(str) + "-01", errors="coerce")

# Orden columnas
cols_order = ["Codigo","NombreCompleto","Sucursal","PeriodoYM","PeriodoYMD","Bloque","TipoDato","Monto","TotalCapitalTape","TotalCobranzaEstimada"]
for c in cols_order:
    if c not in df_long.columns:
        df_long[c] = pd.NA
df_long = df_long[cols_order]
df_long["Codigo"] = df_long["Codigo"].astype(str).str.replace(r"\.0$","",regex=True)

# ----------------------------
# Calcular TIR por Codigo y unir
# ----------------------------
print("📈 Calculando TIR por código...")
tir_por_codigo = df_long.groupby("Codigo", group_keys=False).apply(calcular_xirr).reset_index(name="TIR")
df_long = df_long.merge(tir_por_codigo, on="Codigo", how="left")

# ----------------------------
# Guardar CSV y subir a BigQuery
# ----------------------------
OUT_CSV_CONSOL = "ConsolidadoPorPeriodo.csv"
df_long.to_csv(OUT_CSV_CONSOL, index=False, encoding="utf-8-sig")
print(f"\n✅ CSV generado (sobrescrito): {OUT_CSV_CONSOL}")
print(f"   Total de filas: {len(df_long):,}")

print("\n⬆️ Subiendo a BigQuery...")
client = bigquery.Client(project=PROYECTO_BQ)

def subir_a_bq(df, tabla):
    df_bq = df.copy()
    df_bq.columns = (
        df_bq.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
    )
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
    job = client.load_table_from_dataframe(df_bq, f"{PROYECTO_BQ}.{DATASET_BQ}.{tabla}", job_config=job_config)
    job.result()
    print(f"🎉 Datos subidos a {tabla}")

subir_a_bq(df_long, TABLA_CONSOLIDADO)

print("\n✅ Proceso completado exitosamente")