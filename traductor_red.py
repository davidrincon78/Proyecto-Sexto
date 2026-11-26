"""
traductor_red.py  ·  Traductor de lengua de señas con red CNN + LSTM MULTIMODAL
=================================================================================
Hace lo mismo que traductor.py (cámara / video → segmentar gesto → palabra →
frase → voz), pero el clasificador es una red neuronal CNN + LSTM multimodal
(red_cnn_lstm.py) que ENTRENAS con:

    · videos        (.mp4 .avi .mov ...)   → se procesan con MediaPipe
    · landmarks     (landmarks.csv / .xlsx de resultados/, el análisis de completo.py)
                    → se convierten al mismo vector por frame que la captura en vivo
    · dataset/      (.npz de completo.py)  → opcional

Con todos esos gestos se reserva un 80 % para entrenar y un 20 % para
validar, y la precisión que se muestra es la medida en ese 20 %.

AGREGAR DATOS DESPUÉS DE CREAR EL MODELO
    · Los gestos ya procesados se guardan en DATOS_PATH (datos_cnn_lstm.npz) cada
      vez que entrenas, y se vuelven a abrir solos al iniciar la app. Para añadir
      datos: abre la app, agrega los videos / landmarks nuevos y pulsa «Entrenar red».
      No hace falta volver a procesar los videos anteriores con MediaPipe.
    · Cada gesto recuerda si quedó en entrenamiento o en validación: al añadir datos
      solo se reparten los gestos nuevos, así la validación sigue siendo honesta y
      las precisiones de modelos sucesivos son comparables.
    · «💾 Guardar datos…» / «📂 Abrir datos…» guardan o abren otros conjuntos.
    · Casilla «Partir del modelo actual (ajuste fino)»: en vez de crear una red nueva,
      sigue entrenando la red cargada (más rápido; admite señas nuevas). Conviene
      tener abiertos también los datos antiguos para que no "olvide" lo aprendido.

Reutiliza de traductor.py: la interfaz, el motor de cámara, el segmentador de
gestos y el pipeline de MediaPipe. Por eso traductor.py, completo.py y
red_cnn_lstm.py deben estar en la misma carpeta.

La red separa las columnas del vector por frame en MODALIDADES (cuerpo, mano
izquierda, mano derecha), pasa cada una por su propia CNN temporal (Conv1D),
fusiona los resultados frame a frame y una LSTM recorre la secuencia.

Dependencias extra:  pip install openpyxl     (solo para leer landmarks.xlsx)
"""
import copy
import csv
import json
import os
import queue
import re
import shutil
import threading
import tkinter as tk
import unicodedata
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np

import completo as c
import traductor as T
from red_cnn_lstm import RedCNNLSTM, matriz_confusion, modalidades_por_nombre

# ───────────────────────────── Configuración ─────────────────────────────
FRAC_VAL = 0.20                 # 20 % de los gestos para validar
EPOCAS = 150
# Arquitectura CNN + LSTM
FILTROS_CNN = (32, 32)          # filtros de cada capa Conv1D (la misma CNN en cada modalidad)
KERNEL_CNN = 5                  # frames que mira cada filtro (impar)
UNIDADES_LSTM = 64              # tamaño del estado oculto de la LSTM
DROPOUT = 0.3                   # dropout de la fusión y de la salida
DROPOUT_MODALIDAD = 0.15        # prob. de apagar una modalidad entera al entrenar
EXTRA_RECORTES = 6              # recortes aleatorios extra por gesto (aumento de datos)
MODELO_PATH = "modelo_cnn_lstm.npz"  # se guarda aquí tras entrenar y se carga al abrir
DATOS_PATH = "datos_cnn_lstm.npz"    # gestos procesados: se guardan al entrenar y se abren al iniciar
FORMATO_DATOS = "datos_lsc_gestos_v1"
LR_AJUSTE = 5e-4                # tasa de aprendizaje del ajuste fino (menor que la de entrenar desde cero)
FPS_RESPALDO = 30.0

# landmarks.csv / landmarks.xlsx: mismas reglas que la captura en vivo de completo.py
# (construir_vector); si completo.py no trae alguna constante se usa el mismo valor por defecto.
SUAVIZAR_ANGULOS = True         # suavizado exponencial de los ángulos de dedos, igual que en vivo
_VIS_MIN = getattr(c, "VIS_MIN", 0.5)              # visibilidad mínima de un punto del cuerpo
_MARGEN = getattr(c, "MARGEN_BORDE", 0.02)         # puntos pegados al borde se descartan
_ESC_MIN = getattr(c, "ESCALA_MIN", 0.05)          # hombros más juntos que esto (fracción del ancho)
_ALPHA = getattr(c, "ALPHA", 0.4)                  # suavizado de ángulos
_PUNTOS_POSE = [i for i, _ in getattr(c, "PUNTOS_VECTOR", [])] or [0, 7, 8, 11, 12, 13, 14, 15, 16]
_DEDOS = getattr(c, "VIVO_DEDOS", {"Indice": [5, 6, 7, 8], "Medio": [9, 10, 11, 12],
                                   "Anular": [13, 14, 15, 16], "Menique": [17, 18, 19, 20]})
_PULGAR = getattr(c, "PULGAR", [1, 2, 3, 4])
_NOMBRES_MANO = getattr(c, "NOMBRES_MANO", [f"{d} {k}" for d in _DEDOS for k in ("MCP", "PIP", "DIP")]
                        + ["Pulgar MCP", "Pulgar IP"])


# ───────────────────────────── Utilidades de datos ─────────────────────────────
def nombre_sena(texto):
    return re.sub(r"\s+", "_", texto.strip())


def sena_de_archivo(ruta):
    """hola_1.mp4 → hola"""
    stem = os.path.splitext(os.path.basename(ruta))[0]
    return nombre_sena(re.sub(r"[\s_\-]*\d+$", "", stem) or stem)


def extraer_gestos(t, X):
    """
    Pasa una secuencia por el MISMO segmentador que usa el traductor en vivo.
    Devuelve ([(t, X), ...], sin_actividad). Si no detecta ningún gesto, usa la
    secuencia completa.
    """
    seg, salida = T.Segmentador(), []
    for ti, vi in zip(t, X):
        r = seg.agregar(float(ti), vi, T.es_activo(vi))
        if r is not None:
            salida.append(r)
    r = seg.cerrar()
    if r is not None:
        salida.append(r)
    sin_actividad = not salida
    if sin_actividad:
        salida = [(np.asarray(t, float), np.asarray(X, np.float32))]
    validos = [(np.asarray(a, float), np.asarray(b, np.float32)) for a, b in salida
               if len(a) >= 3 and a[-1] - a[0] > 0]
    return validos, sin_actividad


def leer_npz(ruta):
    d = np.load(ruta)
    if "x_raw" in d.files:
        return d["t_raw"], d["x_raw"]
    X = d["x"]
    return np.linspace(0, c.DURACION_S, len(X)), X


def leer_tablas(ruta):
    """Devuelve [(nombre_hoja, filas)] de un .xlsx o .csv."""
    ext = os.path.splitext(ruta)[1].lower()
    if ext in (".csv", ".txt", ".tsv"):
        with open(ruta, newline="", encoding="utf-8-sig") as f:
            # El separador se deduce de la primera línea. csv.Sniffer falla con los CSV
            # "españoles" (separador ; y coma decimal): confunde la coma decimal con el separador.
            primera = f.readline()
            f.seek(0)
            sep = ";" if ";" in primera else ("\t" if "\t" in primera else ",")
            return [("csv", list(csv.reader(f, delimiter=sep)))]
    try:
        import openpyxl
    except ImportError:
        raise RuntimeError("Falta openpyxl para leer .xlsx:  pip install openpyxl")
    wb = openpyxl.load_workbook(ruta, read_only=True, data_only=True)
    try:
        return [(ws.title, [list(r) for r in ws.iter_rows(values_only=True)])
                for ws in wb.worksheets]
    finally:
        wb.close()


def _a_float(v):
    """Celda → float; vacía o no numérica → NaN (en landmarks.csv vacío = no detectado)."""
    if v is None:
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return np.nan
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return np.nan


def _flexion(a, b, d):
    """Igual que completo.flexion: 0° = extendido; mayor valor = más flexionado."""
    ba, bd = a - b, d - b
    cos = np.dot(ba, bd) / (np.linalg.norm(ba) * np.linalg.norm(bd) + 1e-9)
    return 180.0 - float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _angulos_mano(pts):
    """Igual que completo.vivo_angulos_mano: 14 flexiones en el orden de NOMBRES_MANO."""
    res = {}
    for nombre, (mcp, pip, dip, tip) in _DEDOS.items():
        res[f"{nombre} MCP"] = _flexion(pts[0], pts[mcp], pts[pip])
        res[f"{nombre} PIP"] = _flexion(pts[mcp], pts[pip], pts[dip])
        res[f"{nombre} DIP"] = _flexion(pts[pip], pts[dip], pts[tip])
    cmc, mcp, ip, tip = _PULGAR
    res["Pulgar MCP"] = _flexion(pts[cmc], pts[mcp], pts[ip])
    res["Pulgar IP"] = _flexion(pts[mcp], pts[ip], pts[tip])
    return np.array([res[n] for n in _NOMBRES_MANO])


def es_tabla_landmarks(filas):
    """¿La hoja tiene el formato de landmarks.csv (resultados/ de completo.py)?"""
    if not filas or not filas[0]:
        return False
    cab = {str(h).strip().lower() for h in filas[0] if h is not None}
    return {"tiempo_ms", "pose_11_x", "manoi_0_x", "manod_0_x"} <= cab


def landmarks_a_secuencia(filas):
    """
    Convierte una hoja de landmarks.csv / landmarks.xlsx en (t, X, pct_validos), con X de
    forma (frames, N_COLUMNAS): el MISMO vector por frame que arma completo.construir_vector
    en la captura en vivo.
      · x, y normalizados (0-1) → píxeles con img_ancho_px / img_alto_px → origen en el
        punto medio de los hombros y escala = ancho de hombros.
      · Cuerpo: 9 puntos (nariz, orejas, hombros, codos, muñecas). Un punto con visibilidad
        < VIS_MIN o pegado al borde va en 0 con visible = 0.
      · Manos: 21 puntos x, y, z (z de imagen relativa a la muñeca, a la escala del ancho);
        mano_IZQ = manoI (mano izquierda de la persona), mano_DER = manoD.
      · Ángulos de dedos: con las coordenadas mundo de la mano (metros), como en vivo,
        suavizados (ALPHA) mientras la mano sigue detectada, y divididos entre 180.
      · Frame sin los dos hombros visibles → vector en 0 (igual que un video sin hombros).
    """
    filas = [f for f in filas if f and any(x not in (None, "") for x in f)]
    if len(filas) < 4:
        raise ValueError("la hoja tiene muy pocas filas")
    cab = {str(h).strip().lower(): i for i, h in enumerate(filas[0]) if h is not None}
    faltan = [k for k in ("tiempo_ms", "img_ancho_px", "img_alto_px") if k not in cab]
    if faltan:
        raise ValueError("no tiene el formato de landmarks.csv (faltan: " + ", ".join(faltan) + ")")
    datos = np.array([[_a_float(f[i]) if i < len(f) else np.nan for i in range(len(filas[0]))]
                      for f in filas[1:]], float)
    n = len(datos)

    def C(nombre):
        i = cab.get(nombre)
        return datos[:, i] if i is not None else np.full(n, np.nan)

    t = C("tiempo_ms") / 1000.0
    W, H = C("img_ancho_px"), C("img_alto_px")
    if np.all(np.isnan(W)) or np.all(np.isnan(H)):
        raise ValueError("faltan img_ancho_px / img_alto_px")
    W = np.where(np.isnan(W), np.nanmedian(W), W)
    H = np.where(np.isnan(H), np.nanmedian(H), H)
    pose = {i: (C(f"pose_{i}_x"), C(f"pose_{i}_y"), C(f"pose_{i}_vis"))
            for i in set(_PUNTOS_POSE) | {11, 12}}
    manos = {}
    for lado, pref in (("IZQ", "manoi"), ("DER", "manod")):
        img = np.stack([np.stack([C(f"{pref}_{j}_{e}") for e in "xyz"], 1) for j in range(21)], 1)
        mundo = np.stack([np.stack([C(f"mundo_{pref}_{j}_{e}") for e in "xyz"], 1) for j in range(21)], 1)
        manos[lado] = (img, mundo, C(f"{pref}_presente"))

    n_ang = len(_NOMBRES_MANO)
    X = np.zeros((n, c.N_COLUMNAS), np.float32)
    validos = np.zeros(n, bool)
    suave = {"IZQ": None, "DER": None}
    for k in range(n):
        w, h = W[k], H[k]
        # manos: ángulos (con su suavizado) en todos los frames, como en vivo
        datos_mano = {}
        for lado, (img, mundo, pres) in manos.items():
            ok = np.all(np.isfinite(img[k])) and (np.isnan(pres[k]) or pres[k] >= 0.5)
            if not ok:
                suave[lado] = None
                continue
            base = mundo[k] if np.all(np.isfinite(mundo[k])) else img[k] * np.array([w, h, w])
            ang = _angulos_mano(base)
            if SUAVIZAR_ANGULOS and suave[lado] is not None:
                ang = _ALPHA * ang + (1 - _ALPHA) * suave[lado]
            suave[lado] = ang
            datos_mano[lado] = (img[k], ang)

        def visible(i):
            x, y, v = pose[i][0][k], pose[i][1][k], pose[i][2][k]
            if np.isnan(x) or np.isnan(y) or (not np.isnan(v) and v < _VIS_MIN):
                return False
            return _MARGEN <= x <= 1 - _MARGEN and _MARGEN <= y <= 1 - _MARGEN

        if not (visible(11) and visible(12)):
            continue                                        # vector en 0
        P = {i: np.array([pose[i][0][k] * w, pose[i][1][k] * h]) for i in pose}
        origen = (P[11] + P[12]) / 2
        escala = float(np.linalg.norm(P[11] - P[12]))
        if escala < _ESC_MIN * w:
            continue
        vec = []
        vis = []
        for i in _PUNTOS_POSE:
            if visible(i):
                vec += list((P[i] - origen) / escala)
                vis.append(1.0)
            else:
                vec += [0.0, 0.0]
                vis.append(0.0)
        vec += vis
        presente, angs = [], []
        for lado in ("IZQ", "DER"):
            if lado in datos_mano:
                pts_img, ang = datos_mano[lado]
                pts = pts_img * np.array([w, h, w])
                pts[:, :2] -= origen
                pts /= escala
                vec += list(pts.ravel())
                presente.append(1.0)
                angs += list(ang / 180.0)
            else:
                vec += [0.0] * 63
                presente.append(0.0)
                angs += [0.0] * n_ang
        vec += presente + angs
        if len(vec) != c.N_COLUMNAS:
            raise ValueError(f"el vector armado tiene {len(vec)} valores y completo.py usa {c.N_COLUMNAS}: "
                             "esta versión de completo.py tiene otro formato de vector")
        X[k] = vec
        validos[k] = True

    orden = np.isfinite(t)                                  # tiempos válidos y crecientes
    t, X, validos = t[orden], X[orden], validos[orden]
    crece = np.concatenate([[True], np.diff(t) > 0])
    t, X, validos = t[crece], X[crece], validos[crece]
    if len(t) < 4 or not validos.any():
        raise ValueError("no se ven los dos hombros en ningún frame")
    return t - t[0], X, 100.0 * validos.mean()


def sena_de_landmarks(ruta):
    """resultados/ACCIDENTE/Anonimo_rep1/landmarks.csv → accidente (si no, como sena_de_archivo)."""
    carpeta = os.path.dirname(os.path.abspath(ruta))
    if os.path.splitext(os.path.basename(ruta))[0].lower() == "landmarks":
        sena = os.path.basename(os.path.dirname(carpeta))
        if sena:
            return nombre_sena(sena).lower()
    return sena_de_archivo(ruta)


def nombre_landmarks(ruta):
    """Nombre corto para la tabla: ACCIDENTE/Anonimo_rep1/landmarks.csv"""
    partes = os.path.normpath(os.path.abspath(ruta)).split(os.sep)
    return "/".join(partes[-3:])


def buscar_landmarks(raiz):
    """Todos los landmarks.csv (o landmarks.xlsx si no hay .csv) dentro de raiz, una vez por carpeta."""
    rutas = []
    for carpeta, _, archivos in os.walk(raiz):
        bajos = {a.lower(): a for a in archivos}
        for nombre in ("landmarks.csv", "landmarks.xlsx"):
            if nombre in bajos:
                rutas.append(os.path.join(carpeta, bajos[nombre]))
                break
    return sorted(rutas)


def gestos_de_video(ruta, proc, progreso=None):
    """Procesa un video con MediaPipe y devuelve ([(t, X)...], nota)."""
    cap, fps, total = c.abrir_video(ruta)
    if cap is None:
        raise RuntimeError("no se pudo abrir el video")
    fps = fps if fps and fps > 0 else FPS_RESPALDO
    proc.reiniciar()                 # MediaPipe exige timestamps que empiecen en 0
    seg, gestos, todo, idx = T.Segmentador(), [], [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = c.ajustar_frame_video(frame)
            t = idx / fps
            _, vec = proc.procesar(frame, int(t * 1000), False)
            vec = vec if vec is not None else np.zeros(c.N_COLUMNAS, np.float32)
            todo.append((t, vec))
            r = seg.agregar(t, vec, T.es_activo(vec))
            if r is not None:
                gestos.append(r)
            idx += 1
            if progreso and idx % 10 == 0:
                progreso(min(idx / max(total, 1), 1.0))
    finally:
        cap.release()
    r = seg.cerrar()
    if r is not None:
        gestos.append(r)
    nota = ""
    if not gestos:
        if 3 <= len(todo) and todo[-1][0] <= T.MAX_SEG_S:
            gestos = [(np.array([a for a, _ in todo]), np.array([b for _, b in todo]))]
            nota = "sin mano levantada detectada: se usó el video completo"
        else:
            raise RuntimeError("no se detectó ningún gesto (¿se ven los hombros y levantas la mano?)")
    gestos = [(np.asarray(a, float), np.asarray(b, np.float32)) for a, b in gestos
              if len(a) >= 3 and a[-1] - a[0] > 0]
    return gestos, nota


# ───────────────────────────── Datos procesados (guardar / abrir) ─────────────────────────────
def guardar_datos(ruta, items):
    """
    Guarda en un .npz los gestos ya extraídos de cada archivo cargado, para no tener
    que volver a procesarlos. Estructura:
        meta     : JSON con formato, N_COLUMNAS y la lista de archivos (tipo, origen,
                   nombre, seña, nota) con sus gestos (id y si son de validación)
        t_<id>   : tiempos (s) del gesto  ·  X_<id> : (frames, N_COLUMNAS) del gesto
    Escribe primero un archivo temporal para no dejar un archivo a medias si falla.
    Devuelve (ruta, número de gestos).
    """
    meta = {"formato": FORMATO_DATOS, "n_columnas": int(c.N_COLUMNAS), "items": []}
    arrays, k = {}, 0
    for it in items:
        ids = []
        for g in it["gestos"]:
            arrays[f"t_{k}"] = np.asarray(g["t"], np.float64)
            arrays[f"X_{k}"] = np.asarray(g["X"], np.float32)
            v = g.get("val")
            ids.append({"id": k, "val": None if v is None else bool(v)})
            k += 1
        meta["items"].append({"tipo": it.get("tipo", ""), "origen": it["origen"],
                              "nombre": it.get("nombre") or os.path.basename(it["origen"]),
                              "sena": it["sena"], "nota": it.get("nota", ""), "gestos": ids})
    if not ruta.lower().endswith(".npz"):
        ruta += ".npz"
    tmp = ruta[:-4] + "_tmp.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    os.replace(tmp, ruta)
    return ruta, k


def leer_datos(ruta):
    """Abre un archivo de guardar_datos(). Devuelve la lista de items del panel."""
    with np.load(ruta) as d:
        if "meta" not in d.files:
            raise ValueError("no es un archivo de datos de entrenamiento")
        meta = json.loads(str(d["meta"]))
        if meta.get("formato") != FORMATO_DATOS:
            raise ValueError("formato de datos desconocido")
        if meta.get("n_columnas") != c.N_COLUMNAS:
            raise ValueError(f"los gestos tienen {meta.get('n_columnas')} columnas por frame y "
                             f"completo.py usa {c.N_COLUMNAS}")
        items = []
        for it in meta["items"]:
            gestos = [{"sena": it["sena"], "t": d[f"t_{g['id']}"], "X": d[f"X_{g['id']}"],
                       "val": g["val"]} for g in it["gestos"]]
            items.append({"tipo": it["tipo"], "origen": it["origen"], "nombre": it["nombre"],
                          "sena": it["sena"], "nota": it["nota"], "gestos": gestos})
    return items


def asignar_validacion(gestos, frac_val=FRAC_VAL, semilla=0):
    """
    Reparto 80 % / 20 % POR GESTO y estratificado por seña, que se conserva entre
    entrenamientos: g["val"] = True (validación) / False (entrenamiento) / None (nuevo).
    Solo se reparten los gestos nuevos, completando el ~20 % de validación de cada
    seña; los que ya tenían lado no se mueven (la red nunca entrena con un gesto que
    antes se usó para validar). Una seña con un solo gesto va entera a entrenamiento.
    Devuelve (indices_train, indices_val).
    """
    rng = np.random.default_rng(semilla)
    por_sena = {}
    for i, g in enumerate(gestos):
        por_sena.setdefault(g["sena"], []).append(i)
    for sena in sorted(por_sena):
        ids = por_sena[sena]
        n = len(ids)
        objetivo = 0 if n < 2 else min(max(int(round(n * frac_val)), 1), n - 1)
        ya_val = sum(1 for i in ids if gestos[i].get("val") is True)
        libres = [i for i in ids if gestos[i].get("val") is None]
        libres = [libres[j] for j in rng.permutation(len(libres))]
        faltan = max(0, objetivo - ya_val)
        for j, i in enumerate(libres):
            gestos[i]["val"] = j < faltan
        if all(gestos[i]["val"] for i in ids):      # p. ej. se quitaron sus gestos de entrenamiento
            gestos[ids[0]]["val"] = False
    tr = [i for i, g in enumerate(gestos) if not g["val"]]
    va = [i for i, g in enumerate(gestos) if g["val"]]
    return tr, va


# ───────────────────────────── Entrenamiento ─────────────────────────────
def modalidades_vector():
    """
    Modalidades (grupos de columnas) del vector por frame de completo.py:
    cuerpo / mano_izq / mano_der. Si completo.py no expone los nombres de las
    columnas, la red usa una sola rama con todas las columnas.
    """
    try:
        nombres = c.nombres_columnas()
    except AttributeError:
        return None
    if len(nombres) != c.N_COLUMNAS:
        return None
    return modalidades_por_nombre(nombres)


def entrenar_modelo(gestos, epocas, avance, semilla=0, base=None, clases_previas=None):
    """
    gestos : lista de dicts {"sena", "t", "X", "val"}.
    base   : None = red NUEVA desde cero.  Un RedCNNLSTM ya entrenado = AJUSTE FINO:
             se copia, se le añaden las señas nuevas y se sigue entrenando.
    clases_previas : señas del modelo actual (solo para avisar si alguna se pierde).
    Reparte 80 % / 20 % por gesto (reparto persistente), entrena con el 80 %
    (+ aumento de datos) y mide con el 20 %. Devuelve (red, lineas_del_informe).
    """
    rng = np.random.default_rng(semilla)
    senas_datos = sorted({g["sena"] for g in gestos})
    nuevas, red = [], None
    if base is not None:
        red = copy.deepcopy(base)       # copia: la app sigue traduciendo con el modelo actual
        nuevas = red.ampliar_clases(senas_datos)
        clases = red.clases
    else:
        clases = senas_datos
    if len(clases) < 2:
        raise RuntimeError("Hacen falta al menos 2 señas distintas.")
    idx = {k: i for i, k in enumerate(clases)}

    tr, va = asignar_validacion(gestos, FRAC_VAL, semilla)
    if not va:
        raise RuntimeError("Hacen falta al menos 2 gestos de alguna seña para reservar el 20 % de validación.")

    Xtr, ytr = [], []
    for i in tr:
        g = gestos[i]
        recortes = list(T.JITTER) + [(rng.uniform(0, 0.15), rng.uniform(0, 0.15))
                                      for _ in range(EXTRA_RECORTES)]
        for a, b in recortes:
            Xtr.append(T.segmento_a_feature(g["t"], g["X"], a, b))
            ytr.append(idx[gestos[i]["sena"]])
    Xva = [T.segmento_a_feature(gestos[i]["t"], gestos[i]["X"]) for i in va]
    yva = [idx[gestos[i]["sena"]] for i in va]
    Xtr, ytr, Xva, yva = np.array(Xtr), np.array(ytr), np.array(Xva), np.array(yva)
    if Xtr.shape[1] != c.N_FRAMES * c.N_COLUMNAS:
        raise RuntimeError(f"segmento_a_feature devuelve {Xtr.shape[1]} valores y la CNN+LSTM espera "
                           f"N_FRAMES×N_COLUMNAS = {c.N_FRAMES}×{c.N_COLUMNAS}")

    modalidades = None
    if base is None:
        modalidades = modalidades_vector()
        red = RedCNNLSTM(c.N_FRAMES, c.N_COLUMNAS, clases, modalidades=modalidades,
                         filtros=FILTROS_CNN, kernel=KERNEL_CNN, lstm=UNIDADES_LSTM,
                         dropout=DROPOUT, dropout_modalidad=DROPOUT_MODALIDAD, semilla=semilla)
        red.entrenar(Xtr, ytr, Xva, yva, epocas=epocas, callback=avance)
    else:
        red.entrenar(Xtr, ytr, Xva, yva, epocas=epocas, lr=LR_AJUSTE, normalizar=False,
                     callback=avance)
    red.calibrar_rechazo(Xtr, ytr, Xva, yva)

    pred = red.probabilidades(Xva).argmax(1)
    acc_val = float(np.mean(pred == yva))
    acc_tr = float(np.mean(red.probabilidades(Xtr).argmax(1) == ytr))
    red.precision = acc_val

    k = len(clases)
    n_tr, n_va = len(tr), len(va)
    inf = [red.resumen()]
    if base is not None:
        inf.append("Ajuste fino del modelo anterior · señas nuevas: "
                   + (", ".join(T.bonito(s) for s in nuevas) if nuevas else "ninguna"))
        sin_datos = [s for s in clases if s not in senas_datos]
        if sin_datos:
            inf.append("⚠ Señas del modelo sin datos abiertos: " + ", ".join(T.bonito(s) for s in sin_datos)
                       + ". Se conservan, pero la red puede olvidarlas en parte: abre también sus datos.")
    else:
        if modalidades is None:
            inf.append("⚠ completo.py no da los nombres de las columnas: se usó una sola rama "
                       "(sin separar modalidades).")
        perdidas = [s for s in (clases_previas or []) if s not in clases]
        if perdidas:
            inf.append("⚠ El modelo anterior reconocía " + ", ".join(T.bonito(s) for s in perdidas)
                       + " y no hay datos de esas señas: el modelo nuevo ya no las reconoce.")
    inf += [f"Red entrenada · {k} señas · {n_tr + n_va} gestos "
           f"({100 * n_tr / (n_tr + n_va):.0f}% entrenamiento / {100 * n_va / (n_tr + n_va):.0f}% validación)",
           f"Precisión en entrenamiento: {acc_tr * 100:.0f}%   ·   en VALIDACIÓN: {acc_val * 100:.0f}%  ({n_va} gestos)"]
    if acc_tr - acc_val > 0.25:
        inf.append("⚠ Mucha diferencia entre entrenamiento y validación (sobreajuste): "
                   "graba/carga más gestos por seña, con variaciones.")
    cm = matriz_confusion(yva, pred, k)
    inf.append("Aciertos en validación por seña:")
    for i, nombre in enumerate(clases):
        tot = int(cm[i].sum())
        if tot:
            inf.append(f"   {T.bonito(nombre):<16} {cm[i, i]}/{tot}")
        else:
            inf.append(f"   {T.bonito(nombre):<16} sin gestos de validación")
    errores = [(clases[i], clases[j], int(cm[i, j])) for i in range(k) for j in range(k)
               if i != j and cm[i, j]]
    if errores:
        inf.append("Confusiones: " + ", ".join(f"{T.bonito(a)}→{T.bonito(b)} ({n})"
                                               for a, b, n in errores[:8]))
    return red, inf


# ───────────────────────────── Panel de datos ─────────────────────────────
class PanelDatos(tk.Toplevel):
    """Ventana para cargar videos / landmarks / dataset, entrenar y guardar la red."""

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.items = {}              # iid de la tabla → {"tipo","origen","sena","gestos"}
        self.ocupado = False
        self.title("Datos y entrenamiento de la red CNN + LSTM")
        self.configure(bg=T.BG)
        self.minsize(820, 600)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)   # ocultar, no destruir

        est = ttk.Style(self)
        est.configure("Treeview", background=T.PANEL2, fieldbackground=T.PANEL2,
                      foreground=T.TEXTO, rowheight=24, borderwidth=0)
        est.configure("Treeview.Heading", background=T.PANEL, foreground=T.TEXTO)
        est.map("Treeview", background=[("selected", T.AZUL)])

        b = app._boton
        fila = tk.Frame(self, bg=T.BG)
        fila.pack(fill="x", padx=14, pady=(14, 6))
        self.botones = [
            b(fila, "🎬 Agregar videos…", self._agregar_videos),
            b(fila, "🧍 Agregar landmarks…", self._agregar_landmarks),
            b(fila, "📁 Cargar dataset/ (.npz)", self._cargar_dataset),
        ]
        for w in self.botones:
            w.pack(side="left", padx=(0, 8))
        btn_vaciar = b(fila, "Vaciar", self._vaciar)
        btn_vaciar.pack(side="right")
        btn_quitar = b(fila, "Quitar seleccionado", self._quitar)
        btn_quitar.pack(side="right", padx=8)
        self.botones += [btn_vaciar, btn_quitar]

        cols = ("tipo", "origen", "sena", "gestos")
        marco_tabla = tk.Frame(self, bg=T.BG)
        marco_tabla.pack(fill="both", expand=True, padx=14)
        self.tabla = ttk.Treeview(marco_tabla, columns=cols, show="headings", height=9)
        for col, txt, ancho in (("tipo", "Tipo", 80), ("origen", "Archivo / hoja", 430),
                                ("sena", "Seña", 150), ("gestos", "Gestos", 70)):
            self.tabla.heading(col, text=txt)
            self.tabla.column(col, width=ancho, anchor="w" if col != "gestos" else "center")
        barra_tabla = ttk.Scrollbar(marco_tabla, orient="vertical", command=self.tabla.yview)
        self.tabla.configure(yscrollcommand=barra_tabla.set)
        barra_tabla.pack(side="right", fill="y")
        self.tabla.pack(side="left", fill="both", expand=True)

        self.lbl_resumen = tk.Label(self, text="", bg=T.BG, fg=T.GRIS, anchor="w",
                                    justify="left", wraplength=780, font=("Segoe UI", 10))
        self.lbl_resumen.pack(fill="x", padx=14, pady=(6, 0))

        fd = tk.Frame(self, bg=T.BG)
        fd.pack(fill="x", padx=14, pady=(8, 0))
        for texto, cmd in (("💾 Guardar datos…", self._guardar_datos),
                           ("📂 Abrir datos…", self._abrir_datos)):
            w = b(fd, texto, cmd)
            w.pack(side="left", padx=(0, 8))
            self.botones.append(w)
        self.var_ajuste = tk.BooleanVar(value=False)
        tk.Checkbutton(fd, text="Partir del modelo actual (ajuste fino)", variable=self.var_ajuste,
                       bg=T.BG, fg=T.TEXTO, selectcolor=T.PANEL2, activebackground=T.BG,
                       activeforeground=T.TEXTO, font=("Segoe UI", 10)).pack(side="right")

        ent = tk.Frame(self, bg=T.BG)
        ent.pack(fill="x", padx=14, pady=10)
        tk.Label(ent, text=f"Reparto: {100 - int(FRAC_VAL * 100)}% entrenar / "
                           f"{int(FRAC_VAL * 100)}% validar     Épocas:",
                 bg=T.BG, fg=T.TEXTO, font=("Segoe UI", 10)).pack(side="left")
        self.sp_epocas = tk.Spinbox(ent, from_=20, to=1000, increment=10, width=6)
        self.sp_epocas.delete(0, "end")
        self.sp_epocas.insert(0, str(EPOCAS))
        self.sp_epocas.pack(side="left", padx=6)
        self.btn_entrenar = b(ent, "🧠 Entrenar red", app._entrenar)
        self.btn_entrenar.pack(side="left", padx=12)
        b(ent, "💾 Guardar modelo…", self._guardar).pack(side="left")
        b(ent, "📂 Cargar modelo…", self._cargar).pack(side="left", padx=8)

        self.barra = ttk.Progressbar(self, maximum=1.0)
        self.barra.pack(fill="x", padx=14)
        self.txt = tk.Text(self, height=11, bg=T.PANEL2, fg=T.TEXTO, relief="flat",
                           font=("Consolas", 10), wrap="word", state="disabled")
        self.txt.pack(fill="both", expand=True, padx=14, pady=(8, 14))
        self.log("Agrega videos, landmarks o el dataset/ y pulsa «Entrenar red». Los datos se guardan "
                 f"solos al entrenar ({DATOS_PATH}) y se vuelven a abrir al iniciar.")
        if os.path.isfile(DATOS_PATH):
            self.abrir_datos(DATOS_PATH, auto=True)

    # ---- utilidades de la ventana
    def log(self, texto):
        self.txt.config(state="normal")
        self.txt.insert("end", texto + "\n")
        self.txt.see("end")
        self.txt.config(state="disabled")

    def epocas(self):
        try:
            return max(5, int(self.sp_epocas.get()))
        except ValueError:
            return EPOCAS

    def todos_los_gestos(self):
        return [g for it in self.items.values() for g in it["gestos"]]

    def set_ocupado(self, ocupado):
        self.ocupado = ocupado
        estado = "disabled" if ocupado else "normal"
        for w in self.botones + [self.btn_entrenar]:
            w.config(state=estado)
        if not ocupado:
            self.barra["value"] = 0

    def actualizar_resumen(self):
        cuenta = {}
        for it in self.items.values():
            cuenta[it["sena"]] = cuenta.get(it["sena"], 0) + len(it["gestos"])
        if not cuenta:
            self.lbl_resumen.config(text="Sin datos cargados.")
            return
        detalle = ", ".join(f"{T.bonito(k)}: {v}" for k, v in sorted(cuenta.items()))
        self.lbl_resumen.config(text=f"{len(cuenta)} señas · {sum(cuenta.values())} gestos  —  {detalle}")

    def agregar_item(self, it, log=True):
        """Añade un archivo (con sus gestos) a la tabla. Devuelve False si ya estaba."""
        if any(x["origen"] == it["origen"] for x in self.items.values()):
            if log:
                self.log(f"Ya estaba cargado: {os.path.basename(it['origen'])}")
            return False
        nombre = it.get("nombre") or os.path.basename(it["origen"])
        iid = self.tabla.insert("", "end", values=(it["tipo"], nombre, T.bonito(it["sena"]),
                                                   len(it["gestos"])))
        self.items[iid] = it
        if log:
            extra = f" ({it['nota']})" if it.get("nota") else ""
            self.log(f"✓ {nombre} → «{T.bonito(it['sena'])}»: {len(it['gestos'])} gesto(s){extra}")
        self.actualizar_resumen()
        return True

    def _quitar(self):
        for iid in self.tabla.selection():
            self.items.pop(iid, None)
            self.tabla.delete(iid)
        self.actualizar_resumen()

    def _vaciar(self):
        if self.items and messagebox.askyesno("Vaciar", "¿Quitar todos los datos cargados?", parent=self):
            self.items.clear()
            self.tabla.delete(*self.tabla.get_children())
            self.actualizar_resumen()

    def _pedir_sena(self, n):
        s = simpledialog.askstring(
            "Nombre de la seña",
            f"¿Qué seña muestran estos {n} archivo(s)?\n\nDéjalo vacío para deducirla de la carpeta "
            "(resultados/ACCIDENTE/… → accidente)\no del nombre del archivo (hola_1.mp4 → hola).", parent=self)
        return None if s is None else nombre_sena(s)

    def _trabajo(self, funcion):
        """Ejecuta una carga en un hilo para no congelar la ventana."""
        if self.ocupado:
            return
        self.set_ocupado(True)
        q = self.app.q_datos

        def run():
            try:
                funcion(q)
            except Exception as e:
                q.put(("log", f"ERROR: {e}"))
            finally:
                q.put(("fin_carga",))
        threading.Thread(target=run, daemon=True).start()

    # ---- cargas
    def _agregar_videos(self):
        rutas = filedialog.askopenfilenames(
            parent=self, title="Elige los videos de una misma seña",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv *.webm *.m4v"), ("Todos", "*.*")])
        if not rutas:
            return
        sena = self._pedir_sena(len(rutas))
        if sena is None:
            return

        def tarea(q):
            proc = T.Procesador()
            try:
                for k, ruta in enumerate(rutas):
                    etiqueta = sena or sena_de_archivo(ruta)
                    q.put(("log", f"Procesando {os.path.basename(ruta)} ({k + 1}/{len(rutas)})…"))
                    try:
                        gs, nota = gestos_de_video(
                            ruta, proc, lambda f: q.put(("progreso", (k + f) / len(rutas))))
                    except Exception as e:
                        q.put(("log", f"✗ {os.path.basename(ruta)}: {e}"))
                        continue
                    q.put(("item", {"tipo": "video", "origen": ruta, "sena": etiqueta, "nota": nota,
                                    "gestos": [{"sena": etiqueta, "t": t, "X": X} for t, X in gs]}))
            finally:
                proc.cerrar()
        self._trabajo(tarea)

    def _agregar_landmarks(self):
        """
        Carga landmarks.csv / landmarks.xlsx de resultados/ (análisis de completo.py) y los
        convierte al vector por frame de la captura en vivo. Se pueden elegir archivos sueltos
        o una carpeta entera (busca todos los landmarks que haya dentro).
        """
        carpeta = messagebox.askyesnocancel(
            "Agregar landmarks",
            "¿Buscar en una CARPETA?\n\n"
            "Sí: eliges una carpeta (p. ej. resultados/) y se cargan todos los landmarks.csv / "
            "landmarks.xlsx que haya dentro. La seña sale del nombre de la carpeta "
            "(resultados/ACCIDENTE/Anonimo_rep1 → accidente).\n\n"
            "No: eliges archivos sueltos.", parent=self)
        if carpeta is None:
            return
        if carpeta:
            raiz = filedialog.askdirectory(parent=self, title="Carpeta con los landmarks (p. ej. resultados/)")
            if not raiz:
                return
            rutas = buscar_landmarks(raiz)
            if not rutas:
                messagebox.showinfo("Agregar landmarks", "No hay ningún landmarks.csv ni landmarks.xlsx "
                                    "dentro de esa carpeta.", parent=self)
                return
            sena = ""                       # en carpeta, cada archivo toma la seña de su carpeta
            self.log(f"Encontrados {len(rutas)} archivos de landmarks en {raiz}")
        else:
            rutas = filedialog.askopenfilenames(
                parent=self, title="Elige los landmarks.csv / landmarks.xlsx",
                filetypes=[("Landmarks", "*.csv *.xlsx"), ("Todos", "*.*")])
            if not rutas:
                return
            sena = self._pedir_sena(len(rutas))
            if sena is None:
                return

        def tarea(q):
            for k, ruta in enumerate(rutas):
                nombre = nombre_landmarks(ruta)
                q.put(("progreso", k / len(rutas)))
                try:
                    tablas = [(h, f) for h, f in leer_tablas(ruta) if es_tabla_landmarks(f)]
                except Exception as e:
                    q.put(("log", f"✗ {nombre}: {e}"))
                    continue
                if not tablas:
                    q.put(("log", f"✗ {nombre}: no tiene el formato de landmarks.csv "
                                  "(columnas tiempo_ms, pose_*, manoI_*, manoD_*)"))
                    continue
                for hoja, filas in tablas:
                    try:
                        t, X, pct = landmarks_a_secuencia(filas)
                    except ValueError as e:
                        q.put(("log", f"✗ {nombre}: {e}"))
                        continue
                    etiqueta = sena or sena_de_landmarks(ruta)
                    gs, sin = extraer_gestos(t, X)
                    if not gs:
                        q.put(("log", f"✗ {nombre}: sin gestos válidos"))
                        continue
                    # mismo origen para el .csv y el .xlsx de una repetición: no se cargan dos veces
                    origen = os.path.splitext(os.path.abspath(ruta))[0] + "::landmarks" \
                        + (f"::{hoja}" if len(tablas) > 1 else "")
                    nota = f"hombros visibles en el {pct:.0f}% de los frames"
                    if sin:
                        nota += "; sin mano levantada: se usó todo el archivo"
                    q.put(("item", {"tipo": "landmarks", "origen": origen, "nombre": nombre,
                                    "sena": etiqueta, "nota": nota,
                                    "gestos": [{"sena": etiqueta, "t": a, "X": b} for a, b in gs]}))
        self._trabajo(tarea)

    def _cargar_dataset(self):
        carpeta = c.CARPETA_DATASET

        def tarea(q):
            if not os.path.isdir(carpeta):
                raise RuntimeError(f"No existe la carpeta «{carpeta}».")
            for sena in sorted(os.listdir(carpeta)):
                ruta_s = os.path.join(carpeta, sena)
                if not os.path.isdir(ruta_s):
                    continue
                gestos, tomas = [], 0
                for f in sorted(os.listdir(ruta_s)):
                    if not f.endswith(".npz"):
                        continue
                    try:
                        t, X = leer_npz(os.path.join(ruta_s, f))
                    except Exception as e:
                        q.put(("log", f"✗ {sena}/{f}: {e}"))
                        continue
                    gs, _ = extraer_gestos(t, X)
                    gestos += [{"sena": sena, "t": a, "X": b} for a, b in gs]
                    tomas += 1
                if gestos:
                    q.put(("item", {"tipo": "dataset", "origen": ruta_s, "sena": sena,
                                    "nombre": f"{sena}/ ({tomas} tomas)", "nota": "", "gestos": gestos}))
        self._trabajo(tarea)

    # ---- datos procesados
    def guardar_datos_en(self, ruta, respaldo=False):
        """Guarda los datos del panel en ruta. respaldo=True copia antes el archivo anterior
        a <nombre>_anterior.npz (así un «Vaciar» seguido de entrenar no borra nada para siempre)."""
        if not self.items:
            return False
        try:
            if respaldo and os.path.isfile(ruta):
                shutil.copy2(ruta, ruta[:-4] + "_anterior.npz")
            ruta, n = guardar_datos(ruta, list(self.items.values()))
        except Exception as e:
            self.log(f"ERROR al guardar los datos: {e}")
            return False
        self.log(f"Datos guardados en {ruta} ({len(self.items)} archivos, {n} gestos).")
        return True

    def abrir_datos(self, ruta, auto=False):
        try:
            items = leer_datos(ruta)
        except Exception as e:
            self.log(f"✗ No se pudieron abrir los datos de {os.path.basename(ruta)}: {e}")
            return
        nuevos = [it for it in items if self.agregar_item(it, log=not auto)]
        n_g = sum(len(it["gestos"]) for it in nuevos)
        ya = len(items) - len(nuevos)
        titulo = "Datos del último entrenamiento abiertos" if auto else f"Datos abiertos de {os.path.basename(ruta)}"
        self.log(f"{titulo}: {len(nuevos)} archivos, {n_g} gestos"
                 + (f" ({ya} ya estaban cargados)" if ya else "") + ". Agrega más datos y vuelve a entrenar.")

    def _guardar_datos(self):
        if not self.items:
            messagebox.showinfo("Guardar datos", "No hay datos cargados.", parent=self)
            return
        ruta = filedialog.asksaveasfilename(parent=self, defaultextension=".npz", initialfile=DATOS_PATH,
                                            filetypes=[("Datos de entrenamiento", "*.npz")])
        if ruta:
            self.guardar_datos_en(ruta)

    def _abrir_datos(self):
        ruta = filedialog.askopenfilename(parent=self, filetypes=[("Datos de entrenamiento", "*.npz")])
        if ruta:
            self.abrir_datos(ruta)

    # ---- modelo
    def _guardar(self):
        if self.app.clf is None:
            messagebox.showinfo("Guardar", "Todavía no hay un modelo entrenado.", parent=self)
            return
        ruta = filedialog.asksaveasfilename(parent=self, defaultextension=".npz",
                                            initialfile=MODELO_PATH, filetypes=[("Modelo", "*.npz")])
        if ruta:
            self.app.clf.guardar(ruta)
            self.log(f"Modelo guardado en {ruta}")

    def _cargar(self):
        ruta = filedialog.askopenfilename(parent=self, filetypes=[("Modelo", "*.npz")])
        if ruta:
            self.app._cargar_modelo(ruta)


# ───────────────────────────── Lista de probabilidades con scroll ─────────────────────────────
def _sin_tildes(texto):
    t = unicodedata.normalize("NFD", str(texto).lower())
    return "".join(ch for ch in t if unicodedata.category(ch) != "Mn")


def _texto(w):
    try:
        return str(w.cget("text"))
    except tk.TclError:
        return ""


def _descendientes(w):
    for h in w.winfo_children():
        yield h
        yield from _descendientes(h)


def _buscar_etiqueta(raiz, contiene):
    for w in _descendientes(raiz):
        if contiene in _texto(w).upper():
            return w
    return None


def _tiene_porcentajes(w):
    """¿Hay dentro de w etiquetas tipo «0%» (las barras originales de traductor.py)?"""
    return any(re.fullmatch(r"\s*\d+(\.\d+)?\s*%\s*", _texto(h)) for h in _descendientes(w))


def _ocultar(w):
    gestor = w.winfo_manager()
    if gestor == "pack":
        w.pack_forget()
    elif gestor == "grid":
        w.grid_forget()
    elif gestor == "place":
        w.place_forget()


class ListaProbabilidades(tk.Frame):
    """
    Lista con SCROLL de todas las señas del modelo, con barra y porcentaje.
      · Ordenada por probabilidad (la más probable arriba y resaltada) o alfabética.
      · Buscador: filtra por nombre, sin importar tildes ni mayúsculas.
      · Rueda del ratón o barra lateral para desplazarse.
    """

    def __init__(self, master, bg):
        super().__init__(master, bg=bg)
        self.bg = bg
        self.filas = {}                  # seña → (fila, etiqueta_nombre, barra, etiqueta_pct)
        self.probs = {}
        self.top, self.fuera = None, False
        self._orden_actual = []

        # cabecera: número de señas · buscador · orden
        cab = tk.Frame(self, bg=bg)
        cab.pack(fill="x", pady=(2, 8))
        self.lbl_info = tk.Label(cab, text="Sin modelo cargado", bg=bg, fg=T.GRIS,
                                 font=("Segoe UI", 9))
        self.lbl_info.pack(side="left")
        self.var_orden = tk.BooleanVar(value=True)
        tk.Checkbutton(cab, text="Por probabilidad", variable=self.var_orden, command=self._reordenar,
                       bg=bg, fg=T.GRIS, selectcolor=T.PANEL2, activebackground=bg,
                       activeforeground=T.TEXTO, font=("Segoe UI", 9), bd=0,
                       highlightthickness=0).pack(side="right", padx=(8, 0))
        self.var_buscar = tk.StringVar()
        self.var_buscar.trace_add("write", lambda *_: self._reordenar())
        caja = tk.Frame(cab, bg=T.PANEL2)
        caja.pack(side="right")
        tk.Label(caja, text="🔍", bg=T.PANEL2, fg=T.GRIS, font=("Segoe UI", 9)).pack(side="left", padx=(6, 0))
        tk.Entry(caja, textvariable=self.var_buscar, bg=T.PANEL2, fg=T.TEXTO, insertbackground=T.TEXTO,
                 relief="flat", bd=0, width=16, font=("Segoe UI", 10)).pack(side="left", ipady=4, padx=(4, 6))

        # zona con scroll: Canvas + Frame interior + barra vertical
        cuerpo = tk.Frame(self, bg=bg)
        cuerpo.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(cuerpo, bg=bg, highlightthickness=0, bd=0)
        barra = ttk.Scrollbar(cuerpo, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=barra.set)
        barra.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.interior = tk.Frame(self.canvas, bg=bg)
        self._ventana = self.canvas.create_window(0, 0, window=self.interior, anchor="nw")
        self.interior.bind("<Configure>",
                           lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._ventana, width=e.width))
        self.lbl_vacio = tk.Label(self.interior, text="", bg=bg, fg=T.GRIS, font=("Segoe UI", 10))

        # rueda del ratón: solo cuando el puntero está sobre la lista
        self.bind_all("<MouseWheel>", self._rueda, add="+")
        self.bind_all("<Button-4>", self._rueda, add="+")     # Linux
        self.bind_all("<Button-5>", self._rueda, add="+")

    # ---- scroll
    def _rueda(self, e):
        try:
            w = self.winfo_containing(e.x_root, e.y_root)
        except (tk.TclError, KeyError):
            return
        while w is not None and w is not self.canvas:
            w = w.master
        if w is None:
            return
        if getattr(e, "num", None) == 4:
            paso = -1
        elif getattr(e, "num", None) == 5:
            paso = 1
        else:
            paso = -1 if e.delta > 0 else 1
        self.canvas.yview_scroll(paso * 2, "units")

    # ---- contenido
    def poner_clases(self, clases):
        """Crea una fila por seña (se llama cuando cambia el modelo)."""
        for fila, *_ in self.filas.values():
            fila.destroy()
        self.filas, self._orden_actual = {}, []
        self.probs = {s: 0.0 for s in clases}
        self.top, self.fuera = None, False
        for s in clases:
            fila = tk.Frame(self.interior, bg=self.bg)
            nombre = tk.Label(fila, text=T.bonito(s), bg=self.bg, fg=T.TEXTO, anchor="w",
                              width=22, font=("Segoe UI", 10))
            pista = tk.Frame(fila, bg=T.PANEL2, height=10)
            barra = tk.Frame(pista, bg=T.GRIS)
            pct = tk.Label(fila, text="0%", bg=self.bg, fg=T.GRIS, anchor="e", width=5,
                           font=("Segoe UI", 9))
            nombre.grid(row=0, column=0, sticky="w", padx=(2, 10))
            pista.grid(row=0, column=1, sticky="ew")
            pct.grid(row=0, column=2, padx=(10, 4))
            fila.columnconfigure(1, weight=1)
            barra.place(x=0, y=0, relheight=1, relwidth=0)
            self.filas[s] = (fila, nombre, barra, pct)
        n = len(clases)
        self.lbl_info.config(text=f"{n} seña{'s' if n != 1 else ''} en el modelo" if n else "Sin modelo cargado")
        self.canvas.yview_moveto(0)
        self._reordenar()

    def mostrar(self, res):
        """Actualiza barras y porcentajes con una predicción de la red."""
        for s, p in res.get("probs", {}).items():
            if s in self.probs:
                self.probs[s] = float(p)
        self.top, self.fuera = res.get("sena"), bool(res.get("fuera"))
        for s, (_, nombre, barra, pct) in self.filas.items():
            p = self.probs.get(s, 0.0)
            es_top = s == self.top
            color = (T.AMBAR if self.fuera else T.AZUL) if es_top else T.GRIS
            barra.config(bg=color)
            barra.place_configure(relwidth=max(0.0, min(1.0, p)))
            pct.config(text=f"{p * 100:.0f}%", fg=T.TEXTO if es_top else T.GRIS)
            nombre.config(fg=T.TEXTO, font=("Segoe UI", 10, "bold") if es_top else ("Segoe UI", 10))
        self._reordenar()

    def _reordenar(self):
        filtro = _sin_tildes(self.var_buscar.get().strip())
        visibles = [s for s in self.filas if filtro in _sin_tildes(T.bonito(s))]
        if self.var_orden.get():
            visibles.sort(key=lambda s: (-self.probs.get(s, 0.0), _sin_tildes(T.bonito(s))))
        else:
            visibles.sort(key=lambda s: _sin_tildes(T.bonito(s)))
        if visibles == self._orden_actual:
            return
        for fila, *_ in self.filas.values():
            fila.pack_forget()
        self.lbl_vacio.pack_forget()
        for s in visibles:
            self.filas[s][0].pack(fill="x", pady=3)
        if self.filas and not visibles:
            self.lbl_vacio.config(text="Ninguna seña coincide con la búsqueda.")
            self.lbl_vacio.pack(pady=10)
        self._orden_actual = visibles


# ───────────────────────────── Aplicación ─────────────────────────────
class AppRed(T.App):
    """La misma interfaz de traductor.py, pero con la red CNN + LSTM como clasificador."""

    def __init__(self):
        # Se definen ANTES de super().__init__() porque este llama a self._entrenar().
        self.q_datos = queue.Queue()
        self.panel = None
        self.entrenando = False
        super().__init__()
        self.title("Traductor de lengua de señas — red CNN + LSTM multimodal")
        self.btn_ent.config(text="🔄 Reentrenar red")
        self._boton(self.btn_ent.master, "🧠 Datos y red…", self._abrir_panel).pack(side="right", padx=6)
        # lista de probabilidades con scroll (sustituye a las barras fijas de traductor.py)
        self.lista = None
        self._cont_lista = self._rama_titulo = self._ventana_lista = None
        self._clases_lista = None
        self._pred_nueva = None
        self._intentos_lista = 0
        self._ticks_lista = 0
        self.after(100, self._poll)
        self.after(300, self._refrescar_lista)
        self.after(400, self._abrir_panel)

    # ---- lista de probabilidades con scroll
    def _red_actual(self):
        clf = getattr(self, "clf", None)
        return clf if isinstance(clf, RedCNNLSTM) else None

    def _al_predecir(self, res):
        # Puede llamarse desde el hilo de la cámara: solo se guarda; la ventana se
        # actualiza en _refrescar_lista (hilo de la interfaz).
        self._pred_nueva = res

    def _instalar_lista(self, forzar=False):
        """
        Coloca la lista con scroll en la tarjeta «PROBABILIDAD POR SEÑA» de traductor.py
        y oculta las barras fijas de esa tarjeta (siguen existiendo y traductor.py las puede
        seguir actualizando, pero ya no se ven). Si no encuentra la tarjeta, abre la lista
        en una ventana aparte.
        """
        titulo = _buscar_etiqueta(self, "PROBABILIDAD")
        cont = None
        if titulo is not None:
            w = titulo.master
            while w is not None and w is not self:       # tarjeta más cercana que contiene las barras
                if _tiene_porcentajes(w):
                    cont = w
                    break
                w = w.master
            if cont is None:
                if not forzar:
                    return False                          # las barras aún no existen: reintentar
                cont = titulo.master
        if cont is not None:
            rama = titulo
            while rama.master is not cont:
                rama = rama.master
            gestor = rama.winfo_manager()
            try:
                bg = cont.cget("bg")
            except tk.TclError:
                bg = T.PANEL
            if gestor in ("pack", "grid"):
                self.lista = ListaProbabilidades(cont, bg)
                if gestor == "pack":
                    self.lista.pack(fill="both", expand=True, pady=(6, 0))
                else:
                    columnas, filas = cont.grid_size()
                    self.lista.grid(row=filas, column=0, columnspan=max(1, columnas),
                                    sticky="nsew", pady=(6, 0))
                    cont.rowconfigure(filas, weight=1)
                self._cont_lista, self._rama_titulo = cont, rama
                self._ocultar_barras_viejas()
                return True
        if not forzar:
            return False
        self._ventana_lista = tk.Toplevel(self)
        self._ventana_lista.title("Probabilidad por seña")
        self._ventana_lista.configure(bg=T.PANEL)
        self._ventana_lista.geometry("420x620")
        self._ventana_lista.protocol("WM_DELETE_WINDOW", self._ventana_lista.withdraw)
        self.lista = ListaProbabilidades(self._ventana_lista, T.PANEL)
        self.lista.pack(fill="both", expand=True, padx=14, pady=14)
        return True

    def _ocultar_barras_viejas(self):
        cont = self._cont_lista
        if cont is None or not cont.winfo_exists():
            return
        for w in cont.winfo_children():
            if w is not self._rama_titulo and w is not self.lista and w.winfo_manager():
                _ocultar(w)

    def _refrescar_lista(self):
        """Bucle de la interfaz (cada 150 ms): instala la lista, la conecta a la red y la actualiza."""
        try:
            if self.lista is None:
                self._intentos_lista += 1
                self._instalar_lista(forzar=self._intentos_lista > 20)   # ~3 s esperando las barras
            red = self._red_actual()
            if red is not None and getattr(red, "al_predecir", None) != self._al_predecir:
                red.al_predecir = self._al_predecir
            if self.lista is not None:
                clases = list(red.clases) if red is not None else []
                if clases != self._clases_lista:
                    self.lista.poner_clases(clases)
                    self._clases_lista = clases
                res, self._pred_nueva = self._pred_nueva, None
                if res is not None:
                    self.lista.mostrar(res)
                self._ticks_lista += 1
                if self._ticks_lista % 4 == 0:      # traductor.py puede volver a dibujar sus barras
                    self._ocultar_barras_viejas()
        except Exception as e:                      # nunca debe tumbar la app
            if not getattr(self, "_error_lista", False):
                self._error_lista = True
                print(f"Aviso (lista de probabilidades): {e}")
        self.after(150, self._refrescar_lista)

    def _abrir_panel(self):
        if self.panel is None or not self.panel.winfo_exists():
            self.panel = PanelDatos(self)
        self.panel.deiconify()
        self.panel.lift()

    # ---- modelo
    def _cargar_modelo(self, ruta):
        try:
            red = RedCNNLSTM.cargar(ruta)
        except Exception as e:
            self.eventos.put(("aviso", f"No se pudo cargar el modelo: {e}"))
            return False
        if (red.n_frames, red.n_columnas) != (c.N_FRAMES, c.N_COLUMNAS):
            self.eventos.put(("aviso", "Ese modelo se entrenó con otra configuración "
                                       "(N_FRAMES o columnas distintos): hay que reentrenar."))
            return False
        self.eventos.put(("modelo", red, [f"Modelo cargado de {ruta}"]))
        return True

    def _entrenar(self):
        """Llamado al abrir la app (carga el modelo guardado) y con «Reentrenar»."""
        if self.entrenando:
            return
        if self.panel is not None and self.panel.ocupado:
            self._log("Espera a que termine la carga de datos para entrenar.")
            return
        gestos = self.panel.todos_los_gestos() if self.panel is not None else []
        if not gestos:
            if os.path.isfile(MODELO_PATH) and self._cargar_modelo(MODELO_PATH):
                return
            self.lbl_modelo.config(text="Sin modelo: carga videos/landmarks en «🧠 Datos y red…» y entrena.",
                                   fg=T.AMBAR)
            if self.panel is not None:
                self._abrir_panel()
            return

        actual = self._red_actual()
        base = None
        if self.panel.var_ajuste.get():
            if actual is not None:
                base = actual
            else:
                self.panel.log("No hay un modelo CNN+LSTM cargado para ajustar: se entrena una red nueva.")
        previas = list(actual.clases) if actual is not None else None

        self.entrenando = True
        self.btn_ent.config(state="disabled")
        self.lbl_modelo.config(text="Entrenando la red CNN + LSTM…", fg=T.AMBAR)
        epocas, q = self.panel.epocas(), self.q_datos
        self.panel.set_ocupado(True)
        modo = "ajuste fino del modelo actual" if base is not None else "red nueva"
        q.put(("log_panel", f"— Entrenando ({modo}) con {len(gestos)} gestos ({epocas} épocas)…"))

        def avance(ep, n, m):
            v = "" if m["val_acc"] is None else f" · validación {m['val_acc'] * 100:.0f}%"
            q.put(("log_panel", f"Época {ep}/{n} · pérdida {m['loss']:.3f} · entrenamiento "
                                f"{m['acc'] * 100:.0f}%{v}"))
            q.put(("progreso", ep / n))

        def tarea():
            try:
                red, informe = entrenar_modelo(gestos, epocas, avance, base=base, clases_previas=previas)
                red.guardar(MODELO_PATH)
                informe.append(f"Modelo guardado en {MODELO_PATH} (se cargará solo al abrir la app).")
                for linea in informe:
                    q.put(("log_panel", linea))
                q.put(("guardar_datos",))           # se guarda en el hilo de la interfaz
                self.eventos.put(("modelo", red, informe))
            except Exception as e:
                if self.clf is None:
                    self.eventos.put(("error_modelo", f"no se pudo entrenar ({e})"))
                else:
                    self.eventos.put(("aviso", f"No se pudo entrenar: {e}"))
                q.put(("log_panel", f"ERROR: {e}"))
            finally:
                q.put(("fin_entreno",))
        threading.Thread(target=tarea, daemon=True).start()

    # ---- cola de mensajes de los hilos de carga / entrenamiento
    def _poll(self):
        while True:
            try:
                ev = self.q_datos.get_nowait()
            except queue.Empty:
                break
            p, tipo = self.panel, ev[0]
            if tipo == "log":
                self._log(ev[1])
                if p:
                    p.log(ev[1])
            elif tipo == "log_panel" and p:
                p.log(ev[1])
            elif tipo == "progreso" and p:
                p.barra["value"] = ev[1]
            elif tipo == "item" and p:
                p.agregar_item(ev[1])
            elif tipo == "fin_carga" and p:
                p.set_ocupado(False)
            elif tipo == "guardar_datos" and p:
                p.guardar_datos_en(DATOS_PATH, respaldo=True)
            elif tipo == "fin_entreno":
                self.entrenando = False
                self.btn_ent.config(state="normal")
                if p:
                    p.set_ocupado(False)
        self.after(100, self._poll)


if __name__ == "__main__":
    try:
        AppRed().mainloop()
    except Exception as e:
        messagebox.showerror("Error", str(e))
        raise