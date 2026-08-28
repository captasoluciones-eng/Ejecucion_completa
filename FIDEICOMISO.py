#!/usr/bin/env python3
"""
Conciliacion FIDEICOMISO vs CAPTAVALE
=====================================

Cruza el reporte del FIDEICOMISO contra los reportes de CAPTAVALE y
clasifica los documentos que no se encuentran, segun las 7 reglas de
negocio.

REGLAS
------
1. BUSQUEDA        Comparar el "Numero de Documento" de cada fila del
                   REPORTE FIDEICOMISO contra el "IdentificadorUnicoFlujo"
                   de REPORTE CAPTAVALE 1 y 2.
2. NO ENCONTRADOS  Conservar las filas de ambas hojas del FIDEICOMISO cuyo
                   numero de documento no exista en ninguno de los dos
                   reportes de CAPTAVALE.
3. DIFERENCIAS     Para los registros encontrados, comparar
                   "CarteraController Total VN" contra el importe SLD
                   ("MontoDelFlujo" de CAPTAVALE).
4. POR CONTRATO    Agrupar los no encontrados por "Numero de Contrato". Las
                   decisiones siguientes consideran el conjunto completo de
                   documentos del contrato.
5. CSTI            Si todos los documentos del contrato inician con C, I, T
                   o S, existen las cuatro letras, y comparten una
                   terminacion de dos digitos para el mismo conjunto,
                   clasificar el contrato como CSTI.
6. STI             El resto de contratos cuyos documentos solo inician con
                   C, I, T, S pero no cumplen la regla 5.
7. INTEGRIDAD      La suma de documentos CSTI, STI y excluidos debe ser
                   igual al total de documentos no encontrados.

USO
---
    python FIDEICOMISO.py <carpeta> [--clave F12540] [--salida ./salida]

La carpeta debe contener:
    REPORTE CAPTA 1 <clave>.csv
    REPORTE CAPTA 2 <clave>.csv
    REPORTE FIDEICOMISO <clave>.xlsx     (tambien acepta .csv)

Ejemplo:
    python FIDEICOMISO.py "G:/Mi unidad/.../F12540" --clave F12540

SALIDA
------
    CONCILIACION <clave>.xlsx    entregable principal, cuatro hojas:
                                 Resumen, No encontrados, Diferencias
                                 importe y Contratos.
    *.csv, resumen_<clave>.json  mismos datos, para uso automatizado.

DEPENDENCIAS
------------
    pip install openpyxl
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------

# Diferencia maxima para considerar que dos importes son iguales (regla 3).
TOLERANCIA_IMPORTE = 0.01

# Nombres de columna en cada origen. Se localizan por nombre, asi que el
# orden real de las columnas puede cambiar sin romper el proceso.
COLS_CAPTA = {
    "contrato": "NumeroDeContrato",
    "documento": "IdentificadorUnicoFlujo",
    "importe": "MontoDelFlujo",
}

COLS_FIDEICOMISO = {
    "contrato": "Numero de Contrato",
    "documento": "Numero de Documento",
    "importe": "CarteraController Total VN",
    "concepto": "Concepto",
    "subtipo": "Subtipo Producto",
}

LETRAS_VALIDAS = {"C", "I", "T", "S"}


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def a_numero(valor):
    """Convierte a float tolerando None, cadenas vacias y separadores."""
    if valor is None or valor == "":
        return 0.0
    if isinstance(valor, (int, float)):
        return float(valor)
    limpio = str(valor).replace("$", "").replace(",", "").strip()
    try:
        return float(limpio)
    except ValueError:
        return 0.0


def indices_de(encabezado, columnas):
    """Mapea {clave logica: indice} a partir de la fila de encabezado."""
    posicion = {}
    for i, nombre in enumerate(encabezado):
        if nombre is not None:
            posicion[str(nombre).strip().lstrip("\ufeff")] = i

    indices = {}
    faltantes = []
    for clave, nombre in columnas.items():
        if nombre in posicion:
            indices[clave] = posicion[nombre]
        else:
            faltantes.append(nombre)

    if faltantes:
        raise SystemExit(
            "Faltan columnas: {}\nEncabezado encontrado: {}".format(
                ", ".join(faltantes), " | ".join(str(h) for h in encabezado)
            )
        )
    return indices


# --------------------------------------------------------------------------
# Lectura de los insumos
# --------------------------------------------------------------------------

def cargar_captavale(rutas):
    """
    Regla 1: construye el indice {numero de documento: importe SLD} a partir
    de los reportes de CAPTAVALE.

    Nota: CAPTA 1 y CAPTA 2 no son reportes independientes, son un mismo
    export partido por el limite de 1,000,000 de filas. Por eso se cargan
    juntos en un solo indice.
    """
    indice = {}
    duplicados = 0
    total = 0

    for ruta in rutas:
        print(f"  Leyendo {Path(ruta).name} ...", flush=True)
        with open(ruta, "r", encoding="utf-8-sig", newline="") as fh:
            lector = csv.reader(fh)
            try:
                encabezado = next(lector)
            except StopIteration:
                continue
            idx = indices_de(encabezado, COLS_CAPTA)

            for fila in lector:
                if len(fila) <= idx["documento"]:
                    continue
                documento = fila[idx["documento"]].strip()
                if not documento:
                    continue
                total += 1
                if documento in indice:
                    duplicados += 1
                    continue
                indice[documento] = a_numero(fila[idx["importe"]])

    print(f"  CAPTAVALE: {total:,} filas -> {len(indice):,} documentos unicos "
          f"({duplicados:,} duplicados ignorados)")
    return indice


def iter_fideicomiso(ruta):
    """
    Recorre el reporte del FIDEICOMISO fila por fila, sin cargarlo entero en
    memoria. Acepta .xlsx (todas las hojas, regla 2) y .csv.

    Entrega tuplas: (contrato, documento, importe, concepto, subtipo)
    """
    ruta = Path(ruta)

    if ruta.suffix.lower() == ".csv":
        yield from _iter_fideicomiso_csv(ruta)
    else:
        yield from _iter_fideicomiso_xlsx(ruta)


def _iter_fideicomiso_csv(ruta):
    with open(ruta, "r", encoding="utf-8-sig", newline="") as fh:
        lector = csv.reader(fh)
        encabezado = next(lector)
        idx = indices_de(encabezado, COLS_FIDEICOMISO)
        for fila in lector:
            if len(fila) <= idx["documento"]:
                continue
            documento = fila[idx["documento"]].strip()
            if not documento:
                continue
            yield (
                fila[idx["contrato"]].strip(),
                documento,
                a_numero(fila[idx["importe"]]),
                fila[idx["concepto"]],
                fila[idx["subtipo"]],
            )


def _iter_fideicomiso_xlsx(ruta):
    try:
        from openpyxl import load_workbook
    except ImportError:
        raise SystemExit(
            "Falta la dependencia openpyxl. Instalala con:\n"
            "    pip install openpyxl"
        )

    # read_only=True activa el modo streaming: openpyxl no materializa la
    # hoja completa, que aqui son mas de un millon de filas.
    libro = load_workbook(ruta, read_only=True, data_only=True)

    try:
        for hoja in libro.worksheets:
            print(f"  Hoja '{hoja.title}' ...", flush=True)
            idx = None
            for fila in hoja.iter_rows(values_only=True):
                if fila is None:
                    continue

                # El encabezado no siempre esta en la primera fila (en estos
                # reportes viene en la segunda), asi que se busca.
                if idx is None:
                    if any(c == COLS_FIDEICOMISO["documento"] for c in fila if c):
                        idx = indices_de(list(fila), COLS_FIDEICOMISO)
                    continue

                if len(fila) <= idx["documento"]:
                    continue
                documento = fila[idx["documento"]]
                if documento is None or str(documento).strip() == "":
                    continue

                yield (
                    str(fila[idx["contrato"]] or "").strip(),
                    str(documento).strip(),
                    a_numero(fila[idx["importe"]]),
                    fila[idx["concepto"]] or "",
                    fila[idx["subtipo"]] or "",
                )

            if idx is None:
                print(f"    (sin encabezado reconocible, hoja omitida)")
    finally:
        libro.close()


# --------------------------------------------------------------------------
# Reglas de clasificacion
# --------------------------------------------------------------------------

def clasificar_contrato(documentos):
    """
    Reglas 5 y 6. Recibe todos los documentos NO ENCONTRADOS de un contrato.

    CSTI     todos inician con C/I/T/S, estan las cuatro letras, y cada
             conjunto que comparte terminacion de dos digitos esta completo
             con las cuatro.
    STI      resto de contratos que solo usan C/I/T/S.
    EXCLUIDO algun documento inicia con otra letra.

    Criterio adoptado: se exige que TODOS los conjuntos por terminacion esten
    completos ("todos los documentos del contrato"). Si existe un conjunto
    completo pero sobran documentos sueltos con otra terminacion, el contrato
    cae en STI.
    """
    letras = {d["letra"] for d in documentos}
    if not letras or not letras <= LETRAS_VALIDAS:
        return "EXCLUIDO"

    por_terminacion = defaultdict(set)
    for d in documentos:
        por_terminacion[d["terminacion"]].add(d["letra"])

    todos_completos = all(
        conjunto >= LETRAS_VALIDAS for conjunto in por_terminacion.values()
    )
    estan_las_cuatro = letras >= LETRAS_VALIDAS

    return "CSTI" if (todos_completos and estan_las_cuatro) else "STI"


# --------------------------------------------------------------------------
# Proceso principal
# --------------------------------------------------------------------------

def conciliar(indice_capta, ruta_fideicomiso):
    """Aplica las reglas 1 a 4 recorriendo el FIDEICOMISO una sola vez."""
    total = 0
    encontrados = 0
    diferencias = []
    no_encontrados_por_contrato = defaultdict(list)

    for contrato, documento, importe, concepto, subtipo in iter_fideicomiso(ruta_fideicomiso):
        total += 1
        if total % 250_000 == 0:
            print(f"    {total:,} filas procesadas ...", flush=True)

        if documento in indice_capta:          # Regla 1
            encontrados += 1
            importe_sld = indice_capta[documento]
            diferencia = abs(importe - importe_sld)
            if diferencia > TOLERANCIA_IMPORTE:  # Regla 3
                diferencias.append({
                    "contrato": contrato,
                    "documento": documento,
                    "cartera_vn": importe,
                    "importe_sld": importe_sld,
                    "diferencia": round(diferencia, 4),
                })
        else:                                   # Regla 2
            no_encontrados_por_contrato[contrato].append({  # Regla 4
                "documento": documento,
                "letra": documento[0].upper() if documento else "",
                "terminacion": documento[-2:],
                "concepto": concepto,
                "subtipo": subtipo,
                "importe": importe,
            })

    return total, encontrados, diferencias, no_encontrados_por_contrato


def escribir_csv(ruta, columnas, filas):
    with open(ruta, "w", encoding="utf-8-sig", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


# --------------------------------------------------------------------------
# Salida en Excel
# --------------------------------------------------------------------------

# Ancho de columna y formato numerico por nombre de campo.
ANCHOS = {
    "contrato": 18, "documento": 20, "letra": 8, "terminacion": 13,
    "concepto": 12, "subtipo": 20, "cartera_vn": 16, "importe_sld": 18,
    "diferencia": 14, "clasificacion_contrato": 22, "documentos": 14,
    "clasificacion": 16,
}
FORMATO_IMPORTE = {"cartera_vn", "importe_sld", "diferencia"}

TITULOS = {
    "contrato": "Numero de Contrato",
    "documento": "Numero de Documento",
    "letra": "Letra",
    "terminacion": "Terminacion",
    "concepto": "Concepto",
    "subtipo": "Subtipo Producto",
    "cartera_vn": "CarteraController Total VN",
    "importe_sld": "Importe SLD (CAPTAVALE)",
    "diferencia": "Diferencia",
    "clasificacion_contrato": "Clasificacion del contrato",
    "documentos": "Documentos no encontrados",
    "clasificacion": "Clasificacion",
}


def _hoja_de_datos(libro, titulo, columnas, filas):
    """Agrega una hoja con encabezado fijo, filtro y formato de importes."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    hoja = libro.create_sheet(titulo)

    encabezado = [TITULOS.get(c, c) for c in columnas]
    hoja.append(encabezado)
    for celda in hoja[1]:
        celda.font = Font(bold=True, color="FFFFFF")
        celda.fill = PatternFill("solid", fgColor="1F4E5F")
        celda.alignment = Alignment(vertical="center")
    hoja.row_dimensions[1].height = 22

    for fila in filas:
        hoja.append([fila.get(c, "") for c in columnas])

    for i, clave in enumerate(columnas, start=1):
        letra = get_column_letter(i)
        hoja.column_dimensions[letra].width = ANCHOS.get(clave, 16)
        if clave in FORMATO_IMPORTE:
            for celda in hoja[letra][1:]:
                celda.number_format = "#,##0.00"

    hoja.freeze_panes = "A2"
    if filas:
        hoja.auto_filter.ref = f"A1:{get_column_letter(len(columnas))}{len(filas) + 1}"
    return hoja


def escribir_excel(ruta, resumen, detalle_no_enc, diferencias, detalle_contratos):
    """
    Genera un unico archivo .xlsx con cuatro hojas, pensado para abrirse
    directamente en Excel sin ningun paso intermedio.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    libro = Workbook()
    libro.remove(libro.active)          # quita la hoja vacia por defecto

    # ---- Hoja Resumen ----
    hoja = libro.create_sheet("Resumen")
    hoja.column_dimensions["A"].width = 42
    hoja.column_dimensions["B"].width = 20

    hoja["A1"] = f"CONCILIACION {resumen['clave']}"
    hoja["A1"].font = Font(bold=True, size=15, color="1F4E5F")
    hoja["A2"] = "FIDEICOMISO contra CAPTAVALE"
    hoja["A2"].font = Font(italic=True, color="666666")

    renglones = [
        ("", ""),
        ("Documentos revisados", resumen["filas_fideicomiso"]),
        ("", ""),
        ("Conciliados", resumen["encontrados"]),
        ("   con diferencia de importe", resumen["diferencias_importe"]),
        ("", ""),
        ("Sin contraparte", resumen["no_encontrados"]),
        ("   contratos afectados", resumen["contratos_no_encontrados"]),
        ("   documentos CSTI", resumen["documentos_csti"]),
        ("   documentos STI", resumen["documentos_sti"]),
        ("   documentos excluidos", resumen["documentos_excluidos"]),
        ("", ""),
        ("Control de integridad (regla 7)", resumen["control_integridad"]),
    ]

    fila_actual = 3
    for etiqueta, valor in renglones:
        hoja.cell(row=fila_actual, column=1, value=etiqueta)
        if valor != "":
            celda = hoja.cell(row=fila_actual, column=2, value=valor)
            if isinstance(valor, int):
                celda.number_format = "#,##0"
                celda.alignment = Alignment(horizontal="right")
        if etiqueta and not etiqueta.startswith("   "):
            hoja.cell(row=fila_actual, column=1).font = Font(bold=True)
            hoja.cell(row=fila_actual, column=2).font = Font(bold=True)
        fila_actual += 1

    # Resalta el control de integridad: es lo primero que hay que mirar.
    fila_ctrl = fila_actual - 1
    cuadra = resumen["control_integridad"] == "OK"
    for col in (1, 2):
        celda = hoja.cell(row=fila_ctrl, column=col)
        celda.font = Font(bold=True, color="FFFFFF")
        celda.fill = PatternFill("solid", fgColor="1B6E45" if cuadra else "B03A2E")

    # ---- Hojas de detalle ----
    _hoja_de_datos(libro, "No encontrados",
                   ["contrato", "documento", "letra", "terminacion", "concepto",
                    "subtipo", "cartera_vn", "clasificacion_contrato"],
                   detalle_no_enc)
    _hoja_de_datos(libro, "Diferencias importe",
                   ["contrato", "documento", "cartera_vn", "importe_sld", "diferencia"],
                   diferencias)
    _hoja_de_datos(libro, "Contratos",
                   ["contrato", "documentos", "clasificacion"],
                   detalle_contratos)

    libro.save(ruta)


def main():
    parser = argparse.ArgumentParser(
        description="Conciliacion FIDEICOMISO vs CAPTAVALE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("carpeta", help="Carpeta que contiene los tres archivos")
    parser.add_argument("--clave", default="F12540",
                        help="Clave del fideicomiso en los nombres de archivo (default: F12540)")
    parser.add_argument("--salida", default="salida",
                        help="Carpeta donde escribir los resultados (default: ./salida)")
    args = parser.parse_args()

    carpeta = Path(args.carpeta)
    if not carpeta.is_dir():
        raise SystemExit(f"No existe la carpeta: {carpeta}")

    capta = [
        carpeta / f"REPORTE CAPTA 1 {args.clave}.csv",
        carpeta / f"REPORTE CAPTA 2 {args.clave}.csv",
    ]

    # El FIDEICOMISO puede venir como .xlsx o .csv.
    fideicomiso = carpeta / f"REPORTE FIDEICOMISO {args.clave}.xlsx"
    if not fideicomiso.exists():
        alterno = carpeta / f"REPORTE FIDEICOMISO {args.clave}.csv"
        if alterno.exists():
            fideicomiso = alterno

    for ruta in capta + [fideicomiso]:
        if not ruta.exists():
            raise SystemExit(f"No se encontro el archivo: {ruta}")

    print(f"\nCONCILIACION {args.clave}")
    print("=" * 60)

    print("\n[1/3] Cargando CAPTAVALE ...")
    indice = cargar_captavale(capta)

    print(f"\n[2/3] Recorriendo {fideicomiso.name} ...")
    total, encontrados, diferencias, no_enc = conciliar(indice, fideicomiso)

    print("\n[3/3] Clasificando contratos no encontrados ...")
    detalle_no_enc = []
    detalle_contratos = []
    conteo = {"CSTI": 0, "STI": 0, "EXCLUIDO": 0}
    no_encontrados = 0

    for contrato, documentos in no_enc.items():
        clase = clasificar_contrato(documentos)          # Reglas 5 y 6
        conteo[clase] += len(documentos)
        no_encontrados += len(documentos)

        detalle_contratos.append({
            "contrato": contrato,
            "documentos": len(documentos),
            "clasificacion": clase,
        })
        for d in documentos:
            detalle_no_enc.append({
                "contrato": contrato,
                "documento": d["documento"],
                "letra": d["letra"],
                "terminacion": d["terminacion"],
                "concepto": d["concepto"],
                "subtipo": d["subtipo"],
                "cartera_vn": d["importe"],
                "clasificacion_contrato": clase,
            })

    # Regla 7: control de integridad
    suma = conteo["CSTI"] + conteo["STI"] + conteo["EXCLUIDO"]
    cuadra = (suma == no_encontrados)

    resumen = {
        "clave": args.clave,
        "filas_fideicomiso": total,
        "encontrados": encontrados,
        "diferencias_importe": len(diferencias),
        "no_encontrados": no_encontrados,
        "contratos_no_encontrados": len(no_enc),
        "documentos_csti": conteo["CSTI"],
        "documentos_sti": conteo["STI"],
        "documentos_excluidos": conteo["EXCLUIDO"],
        "control_integridad": "OK" if cuadra else "DESCUADRE",
    }

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    escribir_csv(salida / f"no_encontrados_{args.clave}.csv",
                 ["contrato", "documento", "letra", "terminacion", "concepto",
                  "subtipo", "cartera_vn", "clasificacion_contrato"],
                 detalle_no_enc)
    escribir_csv(salida / f"diferencias_importe_{args.clave}.csv",
                 ["contrato", "documento", "cartera_vn", "importe_sld", "diferencia"],
                 diferencias)
    escribir_csv(salida / f"contratos_{args.clave}.csv",
                 ["contrato", "documentos", "clasificacion"],
                 detalle_contratos)
    with open(salida / f"resumen_{args.clave}.json", "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)

    # Entregable principal: un solo Excel con las cuatro hojas.
    ruta_excel = salida / f"CONCILIACION {args.clave}.xlsx"
    escribir_excel(ruta_excel, resumen, detalle_no_enc, diferencias, detalle_contratos)

    print("\n" + "=" * 60)
    print(f"RESUMEN {args.clave}")
    print("=" * 60)
    print(f"Filas del FIDEICOMISO            {total:>12,}")
    print(f"  Encontrados                    {encontrados:>12,}")
    print(f"    con diferencia de importe    {len(diferencias):>12,}")
    print(f"  No encontrados                 {no_encontrados:>12,}")
    print(f"    contratos afectados          {len(no_enc):>12,}")
    print(f"    documentos CSTI              {conteo['CSTI']:>12,}")
    print(f"    documentos STI               {conteo['STI']:>12,}")
    print(f"    documentos excluidos         {conteo['EXCLUIDO']:>12,}")
    print("-" * 60)
    print(f"Regla 7 (integridad): {suma:,} == {no_encontrados:,} -> "
          f"{'OK' if cuadra else 'DESCUADRE'}")
    print("=" * 60)
    print(f"\nEXCEL: {ruta_excel.resolve()}")
    print(f"Detalle en CSV y resumen JSON: {salida.resolve()}")

    if not cuadra:
        sys.exit(1)


if __name__ == "__main__":
    main()
