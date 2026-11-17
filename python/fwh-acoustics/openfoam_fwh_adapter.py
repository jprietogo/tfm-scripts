import os
import glob
import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

# LECTURA DE LOS DATOS DESDE EL ARCHIVO VTK
def _leer_polydata(archivo_vtk):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(archivo_vtk)
    reader.Update()
    polydata = reader.GetOutput()

    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise ValueError(f"Invalid VTK file: {archivo_vtk}")

    return polydata

# EXTRACCIÓN DE LOS DATOS DE LA SUPERFICIE PARA CÁLCULO FWH 1A
def _get_campo(polydata, nom_campo):
    arr = polydata.GetCellData().GetArray(nom_campo)
    if arr is not None:
        return vtk_to_numpy(arr), "celda"

    arr = polydata.GetPointData().GetArray(nom_campo)
    if arr is not None:
        return vtk_to_numpy(arr), "punto"

    raise ValueError(f"Campo '{nom_campo}' no encontrado")

# CÁLCULO DEL VECTOR ÁREA DE UN POLÍGONO A PARTIR DE SUS PUNTOS
def _vector_area_poly(puntos):
    vec_area = np.zeros(3)

    for i in range(1, len(puntos) - 1):
        vec_area += 0.5 * np.cross(
            puntos[i] - puntos[0],
            puntos[i + 1] - puntos[0]
        )

    return vec_area

# LECTURA DE LOS DATOS DE LA SUPERFICIE DESDE UN VTK DE OPENFOAM
def leer_openfoam_surface_vtk(archivo_vtk, campo_press="p"):
    polydata = _leer_polydata(archivo_vtk)
    puntos = vtk_to_numpy(polydata.GetPoints().GetData())
    press_raw, press_loc = _get_campo(polydata, campo_press)

    centros = []
    normales = []
    areas = []
    presiones = []

    for id_celda in range(polydata.GetNumberOfCells()):
        celda = polydata.GetCell(id_celda)
        ids = [celda.GetPointId(i) for i in range(celda.GetNumberOfPoints())]

        if len(ids) < 3:
            continue

        pts = puntos[ids]
        centre = pts.mean(axis=0)

        vec_area = _vector_area_poly(pts)
        area = np.linalg.norm(vec_area)

        if area <= 0:
            continue

        normal = vec_area / area

        if press_loc == "celda":
            pressure = press_raw[id_celda]
        else:
            pressure = press_raw[ids].mean(axis=0)

        centros.append(centre)
        normales.append(normal)
        areas.append(area)
        presiones.append(pressure)

    return (
        np.asarray(centros),
        np.asarray(normales),
        np.asarray(areas),
        np.asarray(presiones),
    )

# EXTRACCIÓN DE LOS DATOS DE UN VTK
def find_serie_temporal_vtk(dir_surface, surfNom_cont=None):
    candidatos = glob.glob(
        os.path.join(dir_surface, "**", "*.vtk"),
        recursive=True
    )
    
    print("Buscando VTK en:", dir_surface)
    print("VTK encontrados: {len(candidatos)}")

    if surfNom_cont is not None:
        candidatos = [
            f for f in candidatos
            if surfNom_cont.lower() in os.path.basename(f).lower()
        ]
    
    #print("VTK tras filtro:", candidatos)

    tiempos = []
    files = []

    for file_path in candidatos:
        parts = os.path.normpath(file_path).split(os.sep)

        t_value = None
        for part in reversed(parts):
            try:
                t_value = float(part)
                break
            except ValueError:
                pass

        if t_value is not None:
            tiempos.append(t_value)
            files.append(file_path)

    if not files:
        raise FileNotFoundError(f"No VTK time series found in {dir_surface}")

    order = np.argsort(tiempos)

    return np.asarray(tiempos)[order], [files[i] for i in order]

# CARGA DE LOS DATOS DE LA SUPERFICIE DESDE UNA SERIE TEMPORAL DE VTKs DE OPENFOAM
def cargar_openfoam_fwh_surface(
    dir_surface,
    campo_press="p",
    surfNom_cont=None,
    t_ini=None,
    t_fin=None,
):
    tiempos, files = find_serie_temporal_vtk(dir_surface, surfNom_cont)

    keep = np.ones_like(tiempos, dtype=bool)

    if t_ini is not None:
        keep &= tiempos >= t_ini

    if t_fin is not None:
        keep &= tiempos <= t_fin

    tiempos = tiempos[keep]
    files = [f for f, k in zip(files, keep) if k]

    list_y = []
    list_n = []
    list_area = []
    list_p = []

    n_caras_ref = None

    for vtk_file in files:
        y, n, area, p = leer_openfoam_surface_vtk(vtk_file, campo_press)

        if n_caras_ref is None:
            n_caras_ref = len(area)
        elif len(area) != n_caras_ref:
            raise ValueError("El número de caras de la superficie cambia entre pasos temporales")

        list_y.append(y)
        list_n.append(n)
        list_area.append(area)
        list_p.append(p)

    return (
        tiempos,
        np.asarray(list_y),
        np.asarray(list_n),
        np.asarray(list_area),
        np.asarray(list_p),
    )

# CÁLCULO DE LOS VECTORES DE VELOCIDAD DE LA SUPERFICIE A PARTIR DE LOS DATOS DE MOVIMIENTO
def vel_surface_from_motion(tiempos, y):
    return np.gradient(y, tiempos, axis=0, edge_order=2)

# CÁLCULO DE LOS VECTORES DE VELOCIDAD DE LA SUPERFICIE A PARTIR DE LOS DATOS DE ROTACIÓN
def vel_surface_from_rot(
    y,
    rpm,
    eje=(1.0, 0.0, 0.0),
    origen=(0.0, 0.0, 0.0),
):
    eje = np.asarray(eje, dtype=float)
    eje = eje / np.linalg.norm(eje)

    origen = np.asarray(origen, dtype=float)

    omega_mag = 2.0 * np.pi * rpm / 60.0
    omega = omega_mag * eje

    r = y - origen[None, None, :]

    return np.cross(omega[None, None, :], r)


def fwh1a_arrays_from_openfoam(
    dir_surface,
    campo_press="p",
    surfNom_cont=None,
    t_ini=None,
    t_fin=None,
    vel_mode="from_motion",
    rpm=None,
    eje=(1.0, 0.0, 0.0),
    origen=(0.0, 0.0, 0.0),
):
    tiempos, y, n, area, p = cargar_openfoam_fwh_surface(
        dir_surface=dir_surface,
        campo_press=campo_press,
        surfNom_cont=surfNom_cont,
        t_ini=t_ini,
        t_fin=t_fin,
    )

    if vel_mode == "movimiento":
        v = vel_surface_from_motion(tiempos, y)

    elif vel_mode == "rotacion":
        if rpm is None:
            raise ValueError("Se debe proporcionar la velocidad en rpm para el modo de rotación")

        v = vel_surface_from_rot(
            y=y,
            rpm=rpm,
            eje=eje,
            origen=origen,
        )

    else:
        raise ValueError("vel_mode debe ser 'movimiento' o 'rotacion'")

    return tiempos, y, n, area, p, v