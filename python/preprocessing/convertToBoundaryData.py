#!/usr/bin/env python3
from pathlib import Path

import numpy as np


def leer_valores_vtk(lineas, inicio, n_valores):
    valores = []
    i = inicio
    while len(valores) < n_valores and i < len(lineas):
        valores.extend(float(x) for x in lineas[i].strip().split())
        i += 1
    if len(valores) < n_valores:
        raise ValueError(f"VTK incompleto: se esperaban {n_valores} valores y se leyeron {len(valores)}")
    return np.array(valores[:n_valores]), i


def leer_valores_enteros_vtk(lineas, inicio, n_valores):
    valores = []
    i = inicio
    while len(valores) < n_valores and i < len(lineas):
        valores.extend(int(x) for x in lineas[i].strip().split())
        i += 1
    if len(valores) < n_valores:
        raise ValueError(f"VTK incompleto: se esperaban {n_valores} enteros y se leyeron {len(valores)}")
    return np.array(valores[:n_valores]), i


def leer_vtk_field_format(vtk_file):
    print(f"Leyendo {vtk_file}...")

    with open(vtk_file, "r", encoding="latin-1") as f:
        lineas = f.readlines()

    puntos = None
    poly = []
    data_loc = None
    n_data = None
    data_campos = {}

    i = 0
    while i < len(lineas):
        line = lineas[i].strip()

        if line.startswith("POINTS"):
            parts = line.split()
            n_puntos = int(parts[1])
            puntos_sp, i = leer_valores_vtk(lineas, i + 1, n_puntos * 3)
            puntos = puntos_sp.reshape(-1, 3)
            print(f"  POINTS geometria: {len(puntos)}")
            continue

        if line.startswith("POLYGONS") or line.startswith("CELLS"):
            parts = line.split()
            n_poly = int(parts[1])
            n_ints = int(parts[2])
            conect_sp, i = leer_valores_enteros_vtk(lineas, i + 1, n_ints)

            poly = []
            pos = 0
            for _ in range(n_poly):
                n_vertices = conect_sp[pos]
                pos += 1
                poly.append(conect_sp[pos:pos + n_vertices])
                pos += n_vertices
            print(f"  POLYGONS/CELLS: {len(poly)}")
            continue

        if line.startswith("POINT_DATA") or line.startswith("CELL_DATA"):
            parts = line.split()
            data_loc = parts[0]
            n_data = int(parts[1])
            print(f"  {data_loc}: {n_data}")
            i += 1
            continue

        if line.startswith("FIELD"):
            parts = line.split()
            n_campos = int(parts[2])
            print(f"  FIELD attributes: {n_campos}")
            i += 1

            for _ in range(n_campos):
                linea_campo = lineas[i].strip().split()
                nom_campo = linea_campo[0]
                n_comp = int(linea_campo[1])
                n_valor = int(linea_campo[2])
                n_raw = n_valor * n_comp
                i += 1

                raw_values, i = leer_valores_vtk(lineas, i, n_raw)
                if n_comp == 1:
                    data_campos[nom_campo] = raw_values
                elif n_comp == 3:
                    data_campos[nom_campo] = raw_values.reshape(-1, 3)
                else:
                    print(f"  ADVERTENCIA: campo {nom_campo} con {n_comp} componentes ignorado")
                    continue

                print(f"  Campo {nom_campo}: {n_valor} valores, {n_comp} componente(s)")
            continue

        i += 1

    if puntos is None:
        raise ValueError("No se encontró la seccion POINTS en el VTK")
    if data_loc is None or n_data is None:
        raise ValueError("No se encontró POINT_DATA ni CELL_DATA en el VTK")

    if data_loc == "POINT_DATA":
        pts = puntos[:n_data]
    elif data_loc == "CELL_DATA":
        if not poly:
            raise ValueError("El VTK tiene CELL_DATA, pero no se encontraron POLYGONS/CELLS")
        if len(poly) != n_data:
            raise ValueError(f"CELL_DATA={n_data}, pero hay {len(poly)} poligonos")
        pts = np.array([puntos[cell].mean(axis=0) for cell in poly])
        print("  Usando centros de celda como puntos para boundaryData")
    else:
        raise ValueError(f"Ubicacion de datos no soportada: {data_loc}")

    for nom_campo, data_cmp in data_campos.items():
        if len(data_cmp) != len(pts):
            raise ValueError(
                f"Campo {nom_campo} tiene {len(data_cmp)} valores, pero hay {len(pts)} puntos"
            )

    print("\nResumen:")
    print(f"  Puntos para boundaryData: {len(pts)}")
    for nom_campo, data_cmp in data_campos.items():
        kind = "vectorial" if data_cmp.ndim == 2 else "escalar"
        print(f"  Campo {nom_campo}: {len(data_cmp)} valores ({kind})")

    return pts, data_campos


def write_openfoam_list(filepath, valores, formatter):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{len(valores)}\n(\n")
        for valor in valores:
            f.write(formatter(valor) + "\n")
        f.write(")\n")


def write_points_file(filepath, puntos):
    write_openfoam_list(
        filepath,
        puntos,
        lambda pt: f"({pt[0]:.8e} {pt[1]:.8e} {pt[2]:.8e})",
    )


def write_openfoam_vector_field(filepath, valores):
    write_openfoam_list(
        filepath,
        valores,
        lambda val: f"({val[0]:.8e} {val[1]:.8e} {val[2]:.8e})",
    )


def write_openfoam_scalar_field(filepath, valores):
    write_openfoam_list(filepath, valores, lambda val: f"{val:.8e}")


# Configuracion
dir_caso = Path(__file__).resolve().parent
vtk_file = dir_caso / "postProcessing/sampleDict/600/planoHelice.vtk"
output_dir = dir_caso / "constant/boundaryData/inlet"

print("=" * 70)
print("Conversion VTK a boundaryData para OpenFOAM")
print("=" * 70)

if not vtk_file.exists():
    raise FileNotFoundError(f"No se encuentra {vtk_file}")

points, fields = leer_vtk_field_format(vtk_file)

dir_tiempo = output_dir / "0"
dir_tiempo.mkdir(parents=True, exist_ok=True)

print("\nEscribiendo archivos OpenFOAM")
write_points_file(output_dir / "points", points)
print(f"  points: {len(points)}")

for nom_campo, data_cmp in fields.items():
    if data_cmp.ndim == 2 and data_cmp.shape[1] == 3:
        write_openfoam_vector_field(dir_tiempo / nom_campo, data_cmp)
        mean_val = np.mean(data_cmp, axis=0)
        print(
            f"  {nom_campo}: vector, media=({mean_val[0]:.6g}, {mean_val[1]:.6g}, {mean_val[2]:.6g})"
        )
    else:
        write_openfoam_scalar_field(dir_tiempo / nom_campo, data_cmp)
        print(f"  {nom_campo}: escalar, media={np.mean(data_cmp):.6g}")

print("\nConversion completada.")
print(f"Archivos creados en: {output_dir}")
