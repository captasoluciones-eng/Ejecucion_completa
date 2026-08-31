#!/usr/bin/env python3
"""
Entrada y salida con Google Drive para el workflow de conciliacion.

Se ejecuta dentro de GitHub Actions: baja los insumos de la carpeta de Drive,
y al terminar sube el Excel de resultados a esa misma carpeta.

USO
---
    python drive_io.py descargar <carpeta_id> <destino> --clave F12540
    python drive_io.py subir <carpeta_id> <archivo>
    python drive_io.py listar <carpeta_id>

CREDENCIALES
------------
Variable de entorno GOOGLE_SERVICE_ACCOUNT_JSON con el contenido completo del
JSON de la cuenta de servicio. La carpeta de Drive debe estar compartida con
el correo de esa cuenta (el campo client_email del JSON).

DEPENDENCIAS
------------
    pip install google-api-python-client google-auth
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

ALCANCES = ["https://www.googleapis.com/auth/drive"]


def servicio():
    """Construye el cliente de Drive a partir de la cuenta de servicio."""
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        raise SystemExit(
            "Faltan dependencias. Instala con:\n"
            "    pip install google-api-python-client google-auth"
        )

    crudo = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not crudo:
        raise SystemExit(
            "Falta la variable de entorno GOOGLE_SERVICE_ACCOUNT_JSON.\n"
            "En GitHub se define como secreto del repositorio."
        )

    try:
        info = json.loads(crudo)
    except json.JSONDecodeError as e:
        raise SystemExit(f"GOOGLE_SERVICE_ACCOUNT_JSON no es un JSON valido: {e}")

    credenciales = service_account.Credentials.from_service_account_info(
        info, scopes=ALCANCES)
    return build("drive", "v3", credentials=credenciales, cache_discovery=False)


def listar(drive, carpeta_id):
    """Devuelve los archivos de una carpeta: [{id, name, size, mimeType}]."""
    archivos = []
    pagina = None
    while True:
        resp = drive.files().list(
            q=f"'{carpeta_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, size, mimeType, modifiedTime)",
            pageSize=1000,
            pageToken=pagina,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        archivos.extend(resp.get("files", []))
        pagina = resp.get("nextPageToken")
        if not pagina:
            break
    return archivos


def resolver_carpeta(drive, carpeta_id, clave):
    """
    Acepta indistintamente la carpeta del fideicomiso o la carpeta que la
    contiene.

    Los archivos se organizan en subcarpetas por fideicomiso:

        FIDEICOMISO-CONCILIACION/
            F12540/   REPORTE CAPTA 1 F12540.csv, ...
            F1551/    REPORTE CAPTA 1 F1551.csv, ...

    Si dentro de carpeta_id hay una subcarpeta llamada como la clave, se baja
    a ella. Asi basta configurar una sola carpeta raiz, y los fideicomisos que
    se agreguen despues funcionan sin tocar nada.
    """
    resp = drive.files().list(
        q=(f"'{carpeta_id}' in parents and "
           "mimeType = 'application/vnd.google-apps.folder' and "
           f"name = '{clave}' and trashed = false"),
        fields="files(id, name)",
        pageSize=10,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    subcarpetas = resp.get("files", [])
    if subcarpetas:
        print(f"  Usando la subcarpeta '{subcarpetas[0]['name']}'")
        return subcarpetas[0]["id"]

    return carpeta_id


def descargar_archivo(drive, archivo, destino):
    """Baja un archivo en trozos, para no cargar 80 MB en memoria de golpe."""
    from googleapiclient.http import MediaIoBaseDownload

    destino = Path(destino)
    peticion = drive.files().get_media(fileId=archivo["id"], supportsAllDrives=True)

    with open(destino, "wb") as fh:
        bajador = MediaIoBaseDownload(fh, peticion, chunksize=16 * 1024 * 1024)
        terminado = False
        ultimo_reportado = -10
        while not terminado:
            estado, terminado = bajador.next_chunk()
            if estado:
                pct = int(estado.progress() * 100)
                if pct - ultimo_reportado >= 10:
                    print(f"      {pct}%", flush=True)
                    ultimo_reportado = pct

    mb = destino.stat().st_size / 1024 / 1024
    print(f"    {destino.name}  ({mb:,.1f} MB)")
    return destino


def cmd_descargar(args):
    """Baja los tres insumos que la conciliacion necesita."""
    drive = servicio()
    destino = Path(args.destino)
    destino.mkdir(parents=True, exist_ok=True)

    carpeta_id = resolver_carpeta(drive, args.carpeta_id, args.clave)
    existentes = {a["name"]: a for a in listar(drive, carpeta_id)}

    # El reporte del fideicomiso puede venir en xlsx o en csv.
    requeridos = [
        f"REPORTE CAPTA 1 {args.clave}.csv",
        f"REPORTE CAPTA 2 {args.clave}.csv",
    ]
    fideicomiso = None
    for extension in ("xlsx", "csv"):
        candidato = f"REPORTE FIDEICOMISO {args.clave}.{extension}"
        if candidato in existentes:
            fideicomiso = candidato
            break

    if fideicomiso is None:
        print("Archivos encontrados en la carpeta:", file=sys.stderr)
        for nombre in sorted(existentes):
            print(f"  - {nombre}", file=sys.stderr)
        raise SystemExit(
            f"\nNo se encontro 'REPORTE FIDEICOMISO {args.clave}.xlsx' ni .csv"
        )
    requeridos.append(fideicomiso)

    faltantes = [n for n in requeridos if n not in existentes]
    if faltantes:
        print("Archivos encontrados en la carpeta:", file=sys.stderr)
        for nombre in sorted(existentes):
            print(f"  - {nombre}", file=sys.stderr)
        raise SystemExit("\nFaltan archivos:\n  " + "\n  ".join(faltantes))

    print(f"Descargando {len(requeridos)} archivos de Drive ...")
    for nombre in requeridos:
        descargar_archivo(drive, existentes[nombre], destino / nombre)

    print(f"\nListo en: {destino.resolve()}")


def cmd_subir(args):
    """
    Sube el resultado a la carpeta de Drive. Si ya existe un archivo con el
    mismo nombre, se actualiza en lugar de crear un duplicado, para que el
    enlace que la gente tenga guardado siga sirviendo.
    """
    from googleapiclient.http import MediaFileUpload

    drive = servicio()
    ruta = Path(args.archivo)
    if not ruta.exists():
        raise SystemExit(f"No existe el archivo: {ruta}")

    tipo = ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if ruta.suffix.lower() == ".xlsx" else "application/octet-stream")
    medio = MediaFileUpload(str(ruta), mimetype=tipo, resumable=True,
                            chunksize=16 * 1024 * 1024)

    # El resultado se deja junto a los insumos, en la subcarpeta del fideicomiso.
    carpeta_id = resolver_carpeta(drive, args.carpeta_id, args.clave)
    existentes = {a["name"]: a for a in listar(drive, carpeta_id)}
    previo = existentes.get(ruta.name)

    if previo:
        print(f"Actualizando '{ruta.name}' en Drive ...")
        archivo = drive.files().update(
            fileId=previo["id"], media_body=medio,
            fields="id, webViewLink", supportsAllDrives=True,
        ).execute()
    else:
        print(f"Subiendo '{ruta.name}' a Drive ...")
        archivo = drive.files().create(
            body={"name": ruta.name, "parents": [carpeta_id]},
            media_body=medio, fields="id, webViewLink",
            supportsAllDrives=True,
        ).execute()

    print(f"Listo: {archivo.get('webViewLink', archivo['id'])}")

    # Lo dejamos en el resumen del workflow para que quede a la vista.
    resumen = os.environ.get("GITHUB_STEP_SUMMARY")
    if resumen:
        with open(resumen, "a", encoding="utf-8") as fh:
            fh.write(f"\n**Resultado:** [{ruta.name}]"
                     f"({archivo.get('webViewLink', '')})\n")


def cmd_listar(args):
    drive = servicio()
    carpeta_id = args.carpeta_id
    if args.clave:
        carpeta_id = resolver_carpeta(drive, carpeta_id, args.clave)

    archivos = listar(drive, carpeta_id)
    if not archivos:
        print("La carpeta esta vacia o la cuenta de servicio no tiene acceso.")
        return

    CARPETA = "application/vnd.google-apps.folder"
    for a in sorted(archivos, key=lambda x: (x["mimeType"] != CARPETA, x["name"])):
        if a["mimeType"] == CARPETA:
            print(f"  [carpeta] {a['name']}")
        else:
            tam = int(a.get("size", 0)) / 1024 / 1024 if a.get("size") else 0
            print(f"            {a['name']:52} {tam:>8,.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Drive I/O para la conciliacion")
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("descargar", help="Baja los insumos de la carpeta")
    p.add_argument("carpeta_id")
    p.add_argument("destino")
    p.add_argument("--clave", default="F12540")
    p.set_defaults(func=cmd_descargar)

    p = sub.add_parser("subir", help="Sube un archivo de resultados")
    p.add_argument("carpeta_id")
    p.add_argument("archivo")
    p.add_argument("--clave", default="F12540")
    p.set_defaults(func=cmd_subir)

    p = sub.add_parser("listar", help="Lista el contenido de la carpeta")
    p.add_argument("carpeta_id")
    p.add_argument("--clave", default=None,
                   help="Si se indica, lista dentro de la subcarpeta con ese nombre")
    p.set_defaults(func=cmd_listar)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
