# simulador_quitas_resumen.py - VERSIÓN CORREGIDA
# Genera Resumen_Simulado con lógica completa de quitas y TIR

import os
import pandas as pd
import numpy as np
import gdown
from datetime import datetime
from scipy.optimize import brentq, minimize_scalar
from google.cloud import bigquery
import warnings

warnings.filterwarnings('ignore')

# ----------------------------
# CONFIG - Ruta credenciales
# ----------------------------
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\PyScripts\lookerstudio-consolidacion-c10dd284ce9d.json"

def configurar_credenciales():
    # 0. Variable de entorno ya seteada externamente
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print("✅ Usando credenciales de variable de entorno")
        return

    # 1. GitHub Actions
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

    # 3. Google Colab
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

# MES_ACTUAL: primer día del mes actual
MES_ACTUAL = pd.to_datetime(datetime.today().strftime("%Y-%m-01"))

# ----------------------------
# Descargar archivos
# ----------------------------
print("⬇️ Descargando archivos...")
for key, fid in ARCHIVOS_DRIVE.items():
    url = f"https://drive.google.com/uc?id={fid}"
    out = ARCHIVOS_LOCAL[key]
    try:
        gdown.download(url, out, quiet=True)
        print(f"  ✓ {key} descargado")
    except Exception:
        print(f"  ℹ {key} ya existe o no se pudo descargar")

# ----------------------------
# Lectura CSVs
# ----------------------------
print("📥 Leyendo CSVs...")
df_abono = pd.read_csv(ARCHIVOS_LOCAL["abono"], on_bad_lines='skip', encoding='utf-8-sig', low_memory=False)
df_bloques = pd.read_csv(ARCHIVOS_LOCAL["bloques"], on_bad_lines='skip', encoding='utf-8-sig', low_memory=False)
df_canje = pd.read_csv(ARCHIVOS_LOCAL["canje"], on_bad_lines='skip', encoding='utf-8-sig', low_memory=False)

# ----------------------------
# Helpers
# ----------------------------
def normalize_codigo_series(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

def normalize_period_value(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
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
    return pd.Series([None]*len(df))

def first_value_safe(grupo, col):
    if col in grupo.columns:
        s = grupo[col].dropna()
        if len(s) > 0:
            return s.iloc[0]
    return None

def parse_debajo(val):
    """Convierte DebajoDel100 a decimal (5 -> 0.05, 100 -> 1.0)"""
    if pd.isna(val):
        return None
    try:
        return float(str(val).replace('%','').replace(',', '.')) / 100.0
    except:
        return None

# ----------------------------
# Funciones TIR (XIRR) - Estilo Excel
# ----------------------------
def xnpv_excel(rate, values, dates):
    """XNPV usando meses como periodo, similar a TIR.NO.PER de Excel"""
    t0 = dates[0]
    meses = [(d.year - t0.year) * 12 + (d.month - t0.month) for d in dates]
    return sum(v / (1 + rate) ** (m / 12.0) for v, m in zip(values, meses))

def xirr_excel(values, dates):
    """Calcula TIR estilo Excel: 0 si no hay al menos un flujo positivo y uno negativo"""
    if not (any(v > 0 for v in values) and any(v < 0 for v in values)):
        return 0.0

    f = lambda r: xnpv_excel(r, values, dates)

    # Intentar encontrar raíz usando brentq
    try:
        # Verificar cambio de signo
        f_low = f(-0.9999)
        f_high = f(10)
        if f_low * f_high < 0:
            return brentq(f, -0.9999, 10)
    except:
        pass

    # Si no hay raíz, devolver 0
    return 0.0

def calcular_xirr(grupo):
    """Calcula la TIR de un grupo tipo Excel TIR.NO.PER"""
    diffs = grupo[grupo["TipoDato"] == "Diferencia"].copy()
    diffs["PeriodoYMD"] = pd.to_datetime(diffs["PeriodoYMD"], errors="coerce")
    diffs = diffs.dropna(subset=["PeriodoYMD", "Monto"])
    diffs = diffs.sort_values("PeriodoYMD")

    if len(diffs) < 2:
        return 0.0

    # Ignorar ceros iniciales
    flujos = diffs["Monto"].astype(float).tolist()
    fechas = diffs["PeriodoYMD"].tolist()
    
    for i, v in enumerate(flujos):
        if v != 0:
            flujos = flujos[i:]
            fechas = fechas[i:]
            break

    if len(flujos) < 2:
        return 0.0

    # Validación: debe haber al menos un flujo positivo y uno negativo
    if not (any(v < 0 for v in flujos) and any(v > 0 for v in flujos)):
        return 0.0

    # Debug para códigos específicos
    codigo = grupo["Codigo"].iloc[0] if "Codigo" in grupo.columns else "N/A"
    if codigo in ["60272", "80379"]:
        print(f"\n🔍 DEBUG Código {codigo}:")
        print("Flujos y fechas usados para TIR:")
        for v, d in zip(flujos, fechas):
            print(f"  {d.date()} -> ${v:,.2f}")

    # Calcular TIR
    try:
        tir = xirr_excel(flujos, fechas)
    except Exception as e:
        print(f"⚠️ Error calculando TIR para código {codigo}: {e}")
        tir = 0.0

    # Limitar tir entre 0 y 200%
    tir = max(0.0, min(2.0, tir))

    if codigo in ["60272", "80379"]:
        print(f"  TIR calculada: {tir:.4f} ({tir*100:.2f}%)")

    return tir

# ----------------------------
# Normalizaciones y agregaciones
# ----------------------------
print("🔄 Normalizando datos...")
for d in (df_abono, df_bloques, df_canje):
    if "Codigo" in d.columns:
        d["Codigo"] = normalize_codigo_series(d["Codigo"])

df_abono["PeriodoYM"] = get_periodo_series(df_abono, ["Periodo","FechaRegistro"])
df_canje["PeriodoYM"] = get_periodo_series(df_canje, ["Periodo","FechaInicioPago"])

# Mapas
mapeo_list = []
for df_src in [df_canje, df_abono, df_bloques]:
    if "Codigo" in df_src.columns:
        cols = [c for c in ["Codigo","NombreCompleto","Sucursal"] if c in df_src.columns]
        if cols:
            mapeo_list.append(df_src[cols].drop_duplicates(subset=["Codigo"]))

df_mapa = pd.concat(mapeo_list, ignore_index=True) if mapeo_list else pd.DataFrame(columns=["Codigo","NombreCompleto","Sucursal"])
if not df_mapa.empty:
    df_mapa = df_mapa.drop_duplicates(subset=["Codigo"], keep="first").set_index("Codigo")

# Agregaciones
if "MontoPagado" not in df_abono.columns:
    df_abono["MontoPagado"] = 0
ab_agg = df_abono.groupby(["Codigo","PeriodoYM"], as_index=False)["MontoPagado"].sum().rename(columns={"MontoPagado":"Abono_Sum"})

if "TotalCapital" not in df_canje.columns:
    df_canje["TotalCapital"] = 0
ca_agg = df_canje.groupby(["Codigo","PeriodoYM"], as_index=False)["TotalCapital"].sum().rename(columns={"TotalCapital":"Canje_Sum"})

merged = pd.merge(ab_agg, ca_agg, on=["Codigo","PeriodoYM"], how="outer").fillna(0)
merged["Diferencia"] = merged["Abono_Sum"] - merged["Canje_Sum"]

# Tabla larga
print("📊 Construyendo tabla larga...")
rows = []
for _, r in merged.iterrows():
    rows.append({"Codigo": r["Codigo"], "PeriodoYM": r["PeriodoYM"], "TipoDato": "Abono", "Monto": r["Abono_Sum"]})
    rows.append({"Codigo": r["Codigo"], "PeriodoYM": r["PeriodoYM"], "TipoDato": "Canje", "Monto": r["Canje_Sum"]})
    rows.append({"Codigo": r["Codigo"], "PeriodoYM": r["PeriodoYM"], "TipoDato": "Diferencia", "Monto": r["Diferencia"]})

df_long = pd.DataFrame(rows)

if not df_mapa.empty:
    df_long["NombreCompleto"] = df_long["Codigo"].map(df_mapa["NombreCompleto"]) 
    df_long["Sucursal"] = df_long["Codigo"].map(df_mapa["Sucursal"])
else:
    df_long["NombreCompleto"] = None
    df_long["Sucursal"] = None

cols_bloques = [c for c in ["Codigo","Bloque","TotalCapitalTape","TotalCobranzaEstimada"] if c in df_bloques.columns]
if cols_bloques:
    df_long = df_long.merge(df_bloques[cols_bloques], on="Codigo", how="left")
else:
    for c in ["Bloque","TotalCapitalTape","TotalCobranzaEstimada"]:
        if c not in df_long.columns:
            df_long[c] = np.nan

df_long["PeriodoYMD"] = pd.to_datetime(df_long["PeriodoYM"].astype(str) + "-01", errors="coerce")

print("📈 Calculando TIR inicial por código...")
tir_por_codigo = df_long.groupby("Codigo", group_keys=False).apply(calcular_xirr).reset_index(name="TIR")
df_long = df_long.merge(tir_por_codigo, on="Codigo", how="left")

# ----------------------------
# ✅ CORRECCIÓN: Generar escenarios DebajoDel100
# ----------------------------
def generar_escenarios(df):
    """
    Genera escenarios de quitas según TIR:
    - Si TIR ≤ 0 o NaN: genera 5, 50 y 100
    - Si TIR > 0: aplica reglas y SIEMPRE genera escenario 100
    """
    escenarios = []
    
    for _, row in df.iterrows():
        tir = row["TIR"]
        
        # Convertir a número válido
        try:
            tir = float(tir)
        except:
            tir = np.nan

        # Siempre agregar escenario base (sin quita)
        base = row.to_dict()
        base["DebajoDel100"] = np.nan
        escenarios.append(base)

        # ✅ LÓGICA CORREGIDA
        if pd.isna(tir) or tir <= 0:
            # Si TIR es NaN, 0 o negativa → generar los tres escenarios
            for quita in [5, 50, 100]:
                r = row.to_dict()
                r["DebajoDel100"] = quita
                escenarios.append(r)
        else:
            # Si TIR > 0, aplicar reglas
            if tir <= 0.04:  # ≤ 4%
                r5 = row.to_dict()
                r5["DebajoDel100"] = 5
                escenarios.append(r5)
            
            if tir <= 0.49:  # ≤ 49%
                r50 = row.to_dict()
                r50["DebajoDel100"] = 50
                escenarios.append(r50)
            
            # ✅ SIEMPRE generar escenario 100
            r100 = row.to_dict()
            r100["DebajoDel100"] = 100
            escenarios.append(r100)

    return pd.DataFrame(escenarios)

print("🎯 Generando escenarios de quitas...")
df_escenarios = generar_escenarios(df_long)
if "DebajoDel100" not in df_escenarios.columns:
    df_escenarios["DebajoDel100"] = np.nan

# ----------------------------
# Funciones robustas para buscar la quita
# ----------------------------
def objetivo_quita(monto_quita, grupo, tir_objetivo):
    """Calcula la diferencia entre TIR resultante y TIR objetivo"""
    sim = grupo.copy()
    
    # Buscar si ya existe una fila del mes actual
    mask = (sim["PeriodoYMD"] == MES_ACTUAL) & (sim["TipoDato"] == "Diferencia")
    
    if mask.any():
        sim.loc[mask, "Monto"] = monto_quita
    else:
        # Agregar nueva fila para el mes actual
        new_row = {
            "Codigo": first_value_safe(grupo, "Codigo") or grupo["Codigo"].iloc[0],
            "NombreCompleto": first_value_safe(grupo, "NombreCompleto"),
            "Sucursal": first_value_safe(grupo, "Sucursal"),
            "Bloque": first_value_safe(grupo, "Bloque"),
            "TotalCapitalTape": first_value_safe(grupo, "TotalCapitalTape"),
            "TotalCobranzaEstimada": first_value_safe(grupo, "TotalCobranzaEstimada"),
            "PeriodoYM": MES_ACTUAL.strftime("%Y-%m"),
            "PeriodoYMD": MES_ACTUAL,
            "TipoDato": "Diferencia",
            "Monto": monto_quita,
            "DebajoDel100": grupo["DebajoDel100"].iloc[0] if "DebajoDel100" in grupo.columns else np.nan
        }
        sim = pd.concat([sim, pd.DataFrame([new_row])], ignore_index=True)
    
    tir = calcular_xirr(sim)
    
    if pd.isna(tir):
        return 1e12  # Valor alto si no se puede calcular
    
    return tir - tir_objetivo

def buscar_quita_robusta(grupo, tir_objetivo):
    """
    Busca el monto de quita que logra la TIR objetivo
    Usa múltiples estrategias para encontrar la solución
    """
    codigo = first_value_safe(grupo, "Codigo")
    tape = first_value_safe(grupo, "TotalCapitalTape")
    
    try:
        tape = float(tape) if tape is not None else 100000.0
    except:
        tape = 100000.0
    
    # Estrategia 1: Búsqueda de raíz con brentq
    a = 0.0
    b = max(tape * 2.0, 100000.0)  # Rango inicial más amplio
    
    try:
        fa = objetivo_quita(a, grupo, tir_objetivo)
        fb = objetivo_quita(b, grupo, tir_objetivo)
        
        # Si hay cambio de signo, usar brentq
        if not (np.isnan(fa) or np.isnan(fb) or np.isinf(fa) or np.isinf(fb)):
            if fa * fb < 0:
                resultado = brentq(lambda x: objetivo_quita(x, grupo, tir_objetivo), a, b, maxiter=500)
                if codigo in ["60272", "80379"]:
                    print(f"  ✓ Quita encontrada (brentq): ${resultado:,.2f}")
                return resultado
    except Exception as e:
        if codigo in ["60272", "80379"]:
            print(f"  ⚠️ Brentq falló: {e}")
    
    # Estrategia 2: Expandir rango de búsqueda
    try:
        for multiplicador in [5, 10, 20, 50, 100]:
            b = tape * multiplicador
            fa = objetivo_quita(a, grupo, tir_objetivo)
            fb = objetivo_quita(b, grupo, tir_objetivo)
            
            if not (np.isnan(fa) or np.isnan(fb) or np.isinf(fa) or np.isinf(fb)):
                if fa * fb < 0:
                    resultado = brentq(lambda x: objetivo_quita(x, grupo, tir_objetivo), a, b, maxiter=500)
                    if codigo in ["60272", "80379"]:
                        print(f"  ✓ Quita encontrada (expandido x{multiplicador}): ${resultado:,.2f}")
                    return resultado
    except Exception as e:
        if codigo in ["60272", "80379"]:
            print(f"  ⚠️ Búsqueda expandida falló: {e}")
    
    # Estrategia 3: Minimización escalar
    try:
        upper = max(tape * 100.0, 500000.0)
        res = minimize_scalar(
            lambda x: abs(objetivo_quita(x, grupo, tir_objetivo)),
            bounds=(0.0, upper),
            method='bounded',
            options={'maxiter': 500, 'xatol': 1.0}
        )
        
        if res.success:
            # Verificar que la solución sea razonable
            tir_resultante = calcular_xirr(grupo) + objetivo_quita(res.x, grupo, tir_objetivo)
            if abs(tir_resultante - tir_objetivo) < 0.05:  # Tolerancia de 5%
                if codigo in ["60272", "80379"]:
                    print(f"  ✓ Quita encontrada (minimización): ${res.x:,.2f}")
                return float(res.x)
    except Exception as e:
        if codigo in ["60272", "80379"]:
            print(f"  ⚠️ Minimización falló: {e}")
    
    # Si todo falla, retornar None
    if codigo in ["60272", "80379"]:
        print(f"  ❌ No se pudo encontrar quita para TIR objetivo {tir_objetivo*100:.0f}%")
    
    return None

# ----------------------------
# Simular por grupo
# ----------------------------
def simular_grupo(grupo):
    """Aplica simulación de quita para alcanzar TIR objetivo"""
    grupo = grupo.copy()
    codigo = first_value_safe(grupo, "Codigo")
    
    tir_obj = parse_debajo(grupo["DebajoDel100"].iloc[0] if "DebajoDel100" in grupo.columns else np.nan)

    
    if tir_obj is None:
        # Sin escenario de quita, solo calcular TIR
        grupo["TIR_Simulada"] = calcular_xirr(grupo)
        grupo["QuitaSimulada"] = np.nan
        return grupo
    
    if codigo in ["60272", "80379"]:
        print(f"\n🎯 Simulando código {codigo} para TIR objetivo: {tir_obj*100:.0f}%")
    
    # Buscar la quita necesaria
    quita = buscar_quita_robusta(grupo, tir_obj)
    
    if quita is not None and quita > 0:
        # Aplicar la quita
        mask = (grupo["PeriodoYMD"] == MES_ACTUAL) & (grupo["TipoDato"] == "Diferencia")
        
        if mask.any():
            grupo.loc[mask, "Monto"] = quita
            grupo.loc[mask, "QuitaSimulada"] = quita
        else:
            new_row = {
                "Codigo": codigo,
                "NombreCompleto": first_value_safe(grupo, "NombreCompleto"),
                "Sucursal": first_value_safe(grupo, "Sucursal"),
                "Bloque": first_value_safe(grupo, "Bloque"),
                "TotalCapitalTape": first_value_safe(grupo, "TotalCapitalTape"),
                "TotalCobranzaEstimada": first_value_safe(grupo, "TotalCobranzaEstimada"),
                "PeriodoYM": MES_ACTUAL.strftime("%Y-%m"),
                "PeriodoYMD": MES_ACTUAL,
                "TipoDato": "Diferencia",
                "Monto": quita,
                "DebajoDel100": grupo["DebajoDel100"].iloc[0],
                "QuitaSimulada": quita
            }
            grupo = pd.concat([grupo, pd.DataFrame([new_row])], ignore_index=True)
    else:
        grupo["QuitaSimulada"] = np.nan
    
    # Recalcular TIR simulada
    grupo["TIR_Simulada"] = calcular_xirr(grupo)
    
    if codigo in ["60272", "80379"]:
        print(f"  TIR Simulada: {grupo['TIR_Simulada'].iloc[0]*100:.2f}%")
    
    return grupo

print("🔁 Ejecutando simulación de quitas...")
df_simulado = df_escenarios.groupby(["Codigo","DebajoDel100"], group_keys=False).apply(simular_grupo).reset_index(drop=True)

# ----------------------------
# Ajustar quita según escenarios y recalcular TIR
# ----------------------------
def calcular_quita_ajustada(grupo):
    """Ajusta la quita según límites del escenario y recalcula TIR"""
    grupo = grupo.copy()
    
    # ✅ Definir quita ajustada según escenario
    def regla(row):
        quita_sim = row["QuitaSimulada"]
        if pd.isna(quita_sim):
            return np.nan
        
        cobranza = row["TotalCobranzaEstimada"]
        tape = row["TotalCapitalTape"]
        
        # Valores por defecto si son NaN
        if pd.isna(cobranza):
            cobranza = float('inf')
        if pd.isna(tape):
            tape = float('inf')
        
        if row["DebajoDel100"] == 5:
            return min(quita_sim, cobranza, tape * 0.7)
        elif row["DebajoDel100"] == 50:
            return min(quita_sim, cobranza, tape * 0.6)
        elif row["DebajoDel100"] == 100:
            return min(quita_sim, cobranza, tape * 0.5)
        else:
            return quita_sim
    
    grupo["QuitaAjustada"] = grupo.apply(regla, axis=1)

    # Aplicar QuitaAjustada y recalcular TIR
    mask = (grupo["TipoDato"] == "Diferencia") & (grupo["PeriodoYMD"] == MES_ACTUAL)
    
    if mask.any() and not pd.isna(grupo["QuitaAjustada"].iloc[0]):
        grupo.loc[mask, "Monto"] = grupo.loc[mask, "QuitaAjustada"]
        
        # Recalcular TIR_Ajustada
        tir_ajustada = calcular_xirr(grupo)
        
        # Validar rango razonable
        if pd.notna(tir_ajustada) and 0.0 <= tir_ajustada <= 2.0:
            grupo["TIR_Ajustada"] = tir_ajustada
        else:
            grupo["TIR_Ajustada"] = grupo["TIR_Simulada"]
    else:
        # Sin ajuste, mantener TIR simulada
        grupo["TIR_Ajustada"] = grupo["TIR_Simulada"]

    return grupo

print("⚙️ Ajustando quitas según límites...")
df_simulado = df_simulado.groupby(["Codigo","DebajoDel100"], group_keys=False).apply(calcular_quita_ajustada).reset_index(drop=True)

# ----------------------------
# Generar resumen
# ----------------------------
print("📋 Generando resumen...")
mes_actual_max = pd.to_datetime(df_simulado["PeriodoYMD"]).max()
df_resumen = df_simulado[
    (pd.to_datetime(df_simulado["PeriodoYMD"]) == mes_actual_max) &
    (df_simulado["TipoDato"] == "Diferencia")
][[
    "Codigo","NombreCompleto","Sucursal","Bloque","TotalCapitalTape","TotalCobranzaEstimada",
    "TIR","DebajoDel100","TIR_Simulada","QuitaSimulada","QuitaAjustada","TIR_Ajustada"
]].drop_duplicates(subset=["Codigo","DebajoDel100"])

# Guardar CSV
df_resumen.to_csv("Resumen_Simulado.csv", index=False, encoding="utf-8-sig")
print(f"✅ Resumen_Simulado.csv generado: {len(df_resumen)} filas")

# Mostrar muestra para código 60272
if "60272" in df_resumen["Codigo"].values:
    print("\n📊 Resultados para código 60272:")
    print(df_resumen[df_resumen["Codigo"] == "60272"][["Codigo","TIR","DebajoDel100","TIR_Simulada","QuitaSimulada","QuitaAjustada","TIR_Ajustada"]].to_string(index=False))

# ----------------------------
# Subir a BigQuery
# ----------------------------
print("\n⬆️ Subiendo Resumen_Simulado.csv a BigQuery...")
PROYECTO_BQ = "lookerstudio-consolidacion"
DATASET_BQ = "DatosLooker_USC_V2"
TABLA_SIMULACION = "Simulacion_de_quitas"

try:
    client = bigquery.Client(project=PROYECTO_BQ)
    table_id = f"{PROYECTO_BQ}.{DATASET_BQ}.{TABLA_SIMULACION}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True
    )

    with open("Resumen_Simulado.csv", "rb") as source_file:
        job = client.load_table_from_file(source_file, table_id, job_config=job_config)
        job.result()
    
    print(f"✅ Tabla {table_id} actualizada: {job.output_rows} filas")
except Exception as e:
    print(f"⚠️ Error subiendo a BigQuery: {e}")


print("\n🎉 Proceso completado exitosamente")


