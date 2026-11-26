"""
sistema_lsc.py   SISTEMA DE ADQUISICION Y MEDICION DE DATOS PARA LENGUA DE SENAS COLOMBIANA (LSC)
                 VERSION UNIFICADA (un solo archivo)

Reune en un solo programa lo que antes estaba repartido en siete scripts:
    analizarvideo.py              extraccion de landmarks, calibracion, fases del gesto, figuras, datasets
    analisis_dedos.py             fases de los dedos, cinematica (velocidad / aceleracion), fotogramas HD
    fisiologicos.py               angulos fisiologicos 3D (convencion ISB) + autotest con esqueletos sinteticos
    diagnostico_fisiologicos.py   explicacion numerica de los avisos de fisiologicos
    calibrar_videos.py            calibracion automatica desde videos con toda la secuencia
    analisis_final.py             pipeline por lotes (un video o una carpeta)
    completo.py                   captura en tiempo real (camara / videos) con cara, contactos y vector ampliado

SIN ARGUMENTOS (boton Ejecutar de VS Code o doble clic) se abre un MENU: eliges los videos o la carpeta con una
ventana del explorador y el programa pregunta la sena y la persona.

Tambien se puede usar con SUBCOMANDOS (desde la carpeta del proyecto, con Python 3.12):

  ANALISIS DE VIDEOS GRABADOS (pipeline principal)
    py -3.12 sistema_lsc.py analizar videos_mediapipe                       toda la carpeta
    py -3.12 sistema_lsc.py analizar videos_mediapipe\\accidente001.mp4      un video
    py -3.12 sistema_lsc.py analizar videos_mediapipe --persona Nikol
    py -3.12 sistema_lsc.py analizar video.mp4 --sena AGUA --persona Carol --sin-blur
        opciones: --calib ARCHIVO  --sin-dataset  --sin-fisio  --tronco {vertical,cadera}
                  --detector {auto,holistic,pose+manos}

  CALIBRACION (una vez por persona / camara)
    py -3.12 sistema_lsc.py calibrar abierta.mp4 puno.mp4 brazos.mp4 --persona Nikol
    py -3.12 sistema_lsc.py calibrar-auto --persona Carol cal001.mp4 cal003.mp4 --solo-abierta cal002.mp4
    py -3.12 sistema_lsc.py calibrar-auto --persona Carol video.mp4 --abierta 2-4.5 --puno 6-8 --brazos 10-13
    py -3.12 sistema_lsc.py revisar-calib calibracion_carol.json

  DATOS Y VALIDACION
    py -3.12 sistema_lsc.py dataset             reconstruye dataset/ (angulos, landmarks, cinematica)
    py -3.12 sistema_lsc.py fisiologicos        angulos fisiologicos 3D de todas las repeticiones
    py -3.12 sistema_lsc.py autotest            verifica formulas y signos de los angulos 3D (correr PRIMERO)
    py -3.12 sistema_lsc.py definiciones        tabla de definiciones de los angulos 3D (para el metodo)
    py -3.12 sistema_lsc.py diagnostico         explica con numeros los avisos de 'fisiologicos'
    py -3.12 sistema_lsc.py reorganizar         copia la estructura antigua a resultados/ y arma el dataset

  CAPTURA EN TIEMPO REAL (lo que antes era completo.py)
    py -3.12 sistema_lsc.py vivo                camara en vivo + teclas (ver seccion 14)
    py -3.12 sistema_lsc.py actualizar-tomas    anade el vector ampliado a tomas viejas

Instalacion (una vez):
    6
    
ESTRUCTURA DE SALIDA
    resultados/SENA/Persona_repN/   landmarks.(csv|xlsx)   coordenadas + visibilidad/presencia de cada punto (crudos)
                                    angulos.(csv|xlsx)     angulos por frame (calibrados, mediana de 5 frames)
                                    variables.csv          TABLA PRINCIPAL: una fila por frame con control de calidad,
                                                           angulos estandarizados (0 = neutro, + = flexion) filtrados,
                                                           velocidades y aceleraciones, rasgos de mano (palma, apertura,
                                                           pinza, separacion de dedos), posicion XYZ respecto a hombros
                                                           y cara, y fases del gesto y de los dedos
                                    fases.(xlsx|png)       fases + hojas Eventos, Estadisticas, Sincronia, Calidad y
                                                           Definiciones de variables
                                    cuerpo.png  dedos_<lado>.(png|xlsx)  manos_espacial.png  (gris = rechazado por QC)
                                    descripcion_figuras.txt  textos de todas las figuras en espanol e ingles
                                    cuerpo_cinematica.csv  dedos_<lado>_cinematica.csv  angulos_fisiologicos.csv
                                    fotogramas/
    dataset/                        angulos_frames.csv  landmarks_frames.csv  resumen_repeticiones.csv
                                    secuencias.(csv|npz)  cinematica_*.csv/.npz  fisiologicos_*.csv
                                    variables_frames.csv  variables_resumen.csv  variables_definiciones.csv
    dataset/<sena>/*.npz            tomas de la captura en tiempo real (subcomando 'vivo')

MEDIAPIPE (API de TAREAS, mediapipe.tasks, en todo el archivo; no se usa mp.solutions)
    - Analisis offline: HolisticLandmarker si la version instalada lo trae; si no (p. ej. MediaPipe 0.10.3x, que ya
      no lo incluye en Python), PoseLandmarker + HandLandmarker con el MISMO formato de salida. El detector usado
      queda registrado en fases.xlsx (hoja Info). Los dos detectores NO dan resultados identicos: no mezcles
      repeticiones de ambos en un mismo analisis sin revisarlo.
    - Captura en vivo: PoseLandmarker + HandLandmarker + FaceLandmarker (como completo.py).

LIMITES (tambien van en el articulo)
    Con una sola camara RGB los angulos son ESTIMACIONES, no mediciones clinicas. Las coordenadas x, y de imagen son
    normalizadas (0-1); z de imagen es relativa; las coordenadas "mundo" son metros estimados por el modelo con origen
    en la cadera (pose) o en el centro de la mano (mano). El error aumenta con oclusion, movimiento hacia la camara,
    poca luz y desenfoque por movimiento. Valida con poses de angulo conocido y reporta tu propio error.
"""
import argparse
import collections
import contextlib
import csv
import glob
import json
import math
import os
import re
import shutil
import sys
import textwrap
import threading
import time
import traceback
import unicodedata
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # figuras a archivo, sin ventana (la captura en vivo usa cv2.imshow)
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.lines import Line2D        # noqa: E402
from matplotlib.patches import ConnectionPatch, Patch  # noqa: E402
from openpyxl import load_workbook         # noqa: E402
from openpyxl.formatting.rule import ColorScaleRule  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================================================================================================
# 1. CONFIGURACION GENERAL
# ======================================================================================================================
BASE = os.path.dirname(os.path.abspath(__file__))     # todas las rutas son relativas a la carpeta de este archivo
CARPETA_RESULTADOS = os.path.join(BASE, "resultados")
CARPETA_DATASET = os.path.join(BASE, "dataset")
CARPETA_VIEJA_CAPT = os.path.join(BASE, "capturas_senas")     # estructura antigua (solo 'reorganizar')
CARPETA_VIEJA_CARAC = os.path.join(BASE, "caracterizacion")   # estructura antigua (solo 'reorganizar')
CALIB_JSON = os.path.join(BASE, "calibracion.json")           # calibracion general (si no hay una por persona)
EXT_VIDEO = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v")

# --- modelos de MediaPipe (se descargan solos la primera vez) ---
_URL_MODELOS = "https://storage.googleapis.com/mediapipe-models/"
POSE_MODELO = "full"                                          # "full" (preciso) o "lite" (rapido); vivo y respaldo
MODELO_HOLISTIC = os.path.join(BASE, "holistic_landmarker.task")
URL_HOLISTIC = _URL_MODELOS + "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
MODELO_POSE = os.path.join(BASE, f"pose_landmarker_{POSE_MODELO}.task")
URL_POSE = _URL_MODELOS + f"pose_landmarker/pose_landmarker_{POSE_MODELO}/float16/1/pose_landmarker_{POSE_MODELO}.task"
MODELO_MANO = os.path.join(BASE, "hand_landmarker.task")
URL_MANO = _URL_MODELOS + "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODELO_CARA = os.path.join(BASE, "face_landmarker.task")
URL_CARA = _URL_MODELOS + "face_landmarker/face_landmarker/float16/1/face_landmarker.task"

DETECTOR = "auto"          # "auto" | "holistic" | "pose+manos"  (analisis offline; ver docstring)

# --- analisis offline ---
INVERTIR_LADOS = False     # True solo si el video esta en espejo (camara frontal de celular)
VISIBILIDAD_MINIMA = 0.5   # visibilidad minima de un punto de la pose para calcular sus angulos
DIST_MAX_MUNECA = 0.30     # distancia max. mano-muneca de la pose (fraccion del ancho) para asignar izq/der
N = 100                    # muestras del tiempo normalizado (figuras y fases)
T_SEQ = 30                 # instantes por secuencia en los datasets
UMBRAL_MOV = 0.20          # tramo activo: fraccion de la velocidad de referencia por encima de la cual hay movimiento
MIN_FASE = 5               # largo minimo de una fase (muestras de N); fases de dedos y fases del borde del gesto
PAD_S = 0.3                # segundos que se agregan antes y despues del tramo activo
# --- tramo activo y fases del gesto (version robusta) ---
MIN_ACTIVO_S = 0.15        # actividad mas corta que esto (picos al iniciar el seguimiento, temblor) no abre el gesto
MAX_PAUSA_ACTIVO_S = 1.5   # el gesto = bloque de actividad alrededor del pico; pausas mas largas lo cortan
UMBRAL_MOV_ENTRA = 0.25    # fases: empieza un movimiento cuando la velocidad supera esto (fraccion del p95 del tramo)
UMBRAL_MOV_SALE = 0.15     # ... y termina cuando baja de esto (histeresis: evita alternar por ruido en el umbral)
MIN_MOV_S = 0.12           # un movimiento mas corto que esto se considera ruido y pasa a sostenimiento
MIN_SOST_S = 0.15          # una pausa ENTRE dos movimientos mas corta que esto no es una fase (se une al movimiento)
ROM = {"cmc": 50, "mcp": 90, "pip": 100, "dip": 80, "ip": 80}  # rango anatomico de referencia (grados)
HOMBRO_REPOSO = 5          # angulo de hombro esperado con el brazo pegado al cuerpo (calibracion)
FILTRO_VENTANA = 5         # filtro de mediana de los angulos (frames)
PIE_EN_FIGURA = True       # False = figuras sin el texto largo al pie (para poner tu propio caption)
DPI_FIG = 200

# --- deteccion (punto 1) ---
CONF_DETECCION = 0.5       # confianza minima de deteccion de pose/cara (Holistic y Pose+Manos)
CONF_LANDMARKS_MANO = 0.5  # confianza minima de los landmarks de la mano
CONF_SEGUIMIENTO = 0.5     # confianza minima de seguimiento entre frames (solo Pose+Manos)
PREPROCESO_LUZ = False     # True = ecualiza la luminancia (CLAHE) ANTES de detectar. Util con poca luz o contraluz;
                           # cambia la imagen que ve el modelo: compara con y sin antes de adoptarlo.
GAP_MAX_S = 0.15           # huecos de deteccion mas cortos que esto se interpolan (y se marcan); los largos quedan NaN

# --- filtrado (punto 6): Butterworth pasa-bajos de 2o orden aplicado ida y vuelta (fase cero, sin retraso) ---
# Frecuencias de corte de partida: DEBEN validarse con tus datos (p. ej. analisis de residuos de Winter).
FC_CUERPO_HZ = 6.0         # angulos de codo / hombro / cabeza y posiciones del cuerpo
FC_DEDOS_HZ = 8.0          # angulos y rasgos de los dedos (movimientos mas rapidos)
FC_POSICION_HZ = 6.0       # posiciones de las manos (trayectorias)

# --- control de calidad por frame (punto 2) ---
QC_SALTO_SW_S = 15.0       # una mano que "viaja" mas rapido que esto (anchos de hombro / s) es un salto del seguimiento
QC_SOLAPE_MIN = 0.20       # fraccion del recuadro de la mano mas pequena cubierta por la otra -> manos solapadas
QC_SOLAPE_CARA_MIN = 0.30  # fraccion del recuadro de la mano dentro de la zona de la cara -> mano sobre la cara
QC_TOL_LONG = 0.30         # tolerancia de longitud de las falanges (coordenadas mundo) respecto a su mediana
QC_MAX_FALANGES_MAL = 3    # la MANO (palma, apertura, orientacion) se acepta si fallan como maximo 3 de sus 20 falanges.
                           # Cada DEDO se evalua aparte: sus angulos se aceptan si sus 4 falanges estan bien.
                           # Revisa este valor con tus datos (hoja Calidad + fotogramas) y reportalo en el articulo.

# --- fases sin punto objetivo definido (punto 8): umbrales de cambio dentro de una fase de movimiento ---
UMBRAL_ALTURA_SW = 0.30    # la mano sube / baja si su altura cambia mas de 0.3 anchos de hombro
UMBRAL_DIST_MANOS_SW = 0.30   # las manos se juntan / separan si su distancia cambia mas de 0.3 anchos de hombro
UMBRAL_CODO_DEG = 30.0     # el codo se flexiona / extiende si cambia mas de 30 grados

if INVERTIR_LADOS:
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 12, 11, 14, 13
    L_WRIST, R_WRIST, L_HIP, R_HIP = 16, 15, 24, 23
    L_EAR, R_EAR = 8, 7
else:
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 11, 12, 13, 14
    L_WRIST, R_WRIST, L_HIP, R_HIP = 15, 16, 23, 24
    L_EAR, R_EAR = 7, 8

# Landmarks de la mano (21 puntos): 0 = muneca; cada dedo = [base, articulacion 1, articulacion 2, punta]
WRIST = 0
DEDOS = {"pulgar": [1, 2, 3, 4], "indice": [5, 6, 7, 8], "medio": [9, 10, 11, 12],
         "anular": [13, 14, 15, 16], "menique": [17, 18, 19, 20]}
NOMBRES_DEDOS = list(DEDOS)
ARTICULACIONES = {"pulgar": ["cmc", "mcp", "ip"], "indice": ["mcp", "pip", "dip"],
                  "medio": ["mcp", "pip", "dip"], "anular": ["mcp", "pip", "dip"],
                  "menique": ["mcp", "pip", "dip"]}
ROM_TOTAL = {d: float(sum(ROM[a] for a in ARTICULACIONES[d])) for d in NOMBRES_DEDOS}
CLAVES_BRAZO = ["codo_izq", "codo_der", "hombro_izq", "hombro_der"]
CLAVES_CABEZA = ["cabeza_roll", "cabeza_pitch", "cabeza_yaw"]
DEDOS_FLEX = ["indice", "medio", "anular", "menique"]
CONEXIONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]

ESCALA = np.array([640.0, 480.0, 640.0])   # px; se actualiza con el tamano de cada video (z a escala del ancho)


class ErrorAnalisis(Exception):
    """El video no permite analizar el gesto (muy corto, sin cuerpo, sin movimiento...)."""


# ======================================================================================================================
# 2. UTILIDADES GENERALES
# ======================================================================================================================
def asegurar_modelo(url, ruta):
    """Descarga el modelo si falta. Se descarga a un .part y luego se renombra: una descarga cortada no deja un
    archivo corrupto que se reutilizaria siempre."""
    if os.path.exists(ruta):
        return
    print(f"Descargando {os.path.basename(ruta)} ...")
    tmp = ruta + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, ruta)
    except Exception as e:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise RuntimeError(f"No se pudo descargar {os.path.basename(ruta)} ({e}). Revisa la conexion o "
                           f"copialo a mano en {BASE}") from e
    print("Descargado.")


def importar_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as e:
        raise SystemExit("Falta MediaPipe:  py -3.12 -m pip install mediapipe") from e
    return mp


def fps_valido(fps):
    """FPS de la cabecera del video; 30 si viene en 0, NaN o un valor absurdo."""
    return float(fps) if fps and np.isfinite(fps) and 1 <= fps <= 240 else 30.0


def carpeta_rep(sena, persona, rep):
    return os.path.join(CARPETA_RESULTADOS, sena, f"{persona}_rep{rep}")


def siguiente_rep(sena, persona):
    usados = []
    for d in glob.glob(os.path.join(CARPETA_RESULTADOS, sena, f"{persona}_rep*")):
        m = re.search(r"_rep(\d+)$", d)
        if m:
            usados.append(int(m.group(1)))
    return max(usados, default=0) + 1


def repeticiones_guardadas(archivo):
    """Itera (carpeta, sena, persona, rep) de resultados/SENA/Persona_repN/<archivo>."""
    for ruta in sorted(glob.glob(os.path.join(CARPETA_RESULTADOS, "*", "*_rep*", archivo))):
        carpeta = os.path.dirname(ruta)
        m = re.match(r"(.+)_rep(\d+)$", os.path.basename(carpeta))
        if m:
            yield carpeta, os.path.basename(os.path.dirname(carpeta)), m.group(1), int(m.group(2))


def limpiar_nombre(s):
    """Nombre valido para carpeta: minusculas, sin tildes, espacios -> _."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9_]+", "_", s).strip("_")


def guardar_excel(ruta, hojas, max_ancho=48):
    """Varias hojas (dict nombre -> DataFrame) con ancho de columna automatico y encabezado fijo."""
    with pd.ExcelWriter(ruta, engine="openpyxl") as w:
        for nombre, df in hojas.items():
            df.to_excel(w, sheet_name=nombre[:31], index=False)
    wb = load_workbook(ruta)
    for ws in wb.worksheets:
        for c in range(1, ws.max_column + 1):
            letra = get_column_letter(c)
            ancho = max(len(str(x.value)) if x.value is not None else 0 for x in ws[letra][:40])
            ws.column_dimensions[letra].width = min(max(ancho + 2, 10), max_ancho)
        ws.freeze_panes = "A2"
    wb.save(ruta)


def _nanargmax_abs(v):
    """Indice del maximo valor absoluto ignorando NaN; None si todo es NaN."""
    a = np.abs(np.asarray(v, float))
    return int(np.nanargmax(a)) if np.isfinite(a).any() else None


# ======================================================================================================================
# 3. GEOMETRIA Y ANGULOS POR FRAME (analisis offline)
# ======================================================================================================================
# Sistema de coordenadas de los landmarks de IMAGEN: x, y normalizados 0-1 (x a la derecha de la imagen, y hacia
# abajo); z relativa (negativa = mas cerca de la camara). Para los angulos se pasan a pixeles con ESCALA (z a la escala
# del ancho), asi un angulo no se deforma por la proporcion del video.
# Coordenadas MUNDO (world landmarks): metros estimados por el modelo.
def punto(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x * ESCALA[0], lm.y * ESCALA[1], lm.z * ESCALA[2]])


def punto_mundo(lm, idx):
    p = lm[idx]
    return np.array([p.x, p.y, p.z])


def angulo_3puntos(a, b, c):
    """Angulo en b (grados, 0-180) con el producto punto: theta = arccos(u.v / |u||v|). NaN si falta un punto
    o un segmento mide ~0. El clip a [-1, 1] evita NaN por error de redondeo."""
    a, b, c = (np.array(v, dtype=float) for v in (a, b, c))
    if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
        return np.nan
    ba, bc = a - b, c - b
    norma = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norma < 1e-8:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / norma, -1.0, 1.0))))


def visible(pose, indices):
    """True si todos los puntos tienen visibilidad >= VISIBILIDAD_MINIMA (o el modelo no la reporta)."""
    for i in indices:
        v = getattr(pose[i], "visibility", None)
        if v is not None and v < VISIBILIDAD_MINIMA:
            return False
    return True


def angulo_pose(pose, i, j, k):
    if not visible(pose, (i, j, k)):
        return np.nan
    return angulo_3puntos(punto(pose, i), punto(pose, j), punto(pose, k))


def angulo_pose_mundo(pose, i, j, k):
    """Angulo con landmarks 3D en metros (pose_world_landmarks)."""
    if not visible(pose, (i, j, k)):
        return np.nan
    return angulo_3puntos(punto_mundo(pose, i), punto_mundo(pose, j), punto_mundo(pose, k))


def claves_mano(prefijo):
    return [f"{prefijo}_{d}_{a}" for d in DEDOS for a in ARTICULACIONES[d]]


CLAVES_MANO_I = claves_mano("manoI")
CLAVES_MANO_D = claves_mano("manoD")


def calcular_angulos_brazo(pose, pose_mundo=None):
    """Codo = hombro-codo-muneca (180 = estirado). Hombro = cadera-hombro-codo (0 = brazo pegado). Si hay
    coordenadas mundo se usan (3D, menos sensibles a la perspectiva)."""
    if not pose:
        return {k: np.nan for k in CLAVES_BRAZO}
    P, f = (pose_mundo, angulo_pose_mundo) if pose_mundo else (pose, angulo_pose)
    return {
        "codo_izq": f(P, L_SHOULDER, L_ELBOW, L_WRIST),
        "codo_der": f(P, R_SHOULDER, R_ELBOW, R_WRIST),
        "hombro_izq": f(P, L_HIP, L_SHOULDER, L_ELBOW),
        "hombro_der": f(P, R_HIP, R_SHOULDER, R_ELBOW),
    }


def calcular_angulos_mano(hand, prefijo):
    """Angulo interno de cada articulacion de los dedos (180 = recto). Coordenadas de imagen en px (3D)."""
    if not hand:
        return {k: np.nan for k in claves_mano(prefijo)}
    res = {}
    muneca = punto(hand, WRIST)
    for dedo, idxs in DEDOS.items():
        p1, p2, p3, p4 = [punto(hand, i) for i in idxs]
        art = ARTICULACIONES[dedo]
        res[f"{prefijo}_{dedo}_{art[0]}"] = angulo_3puntos(muneca, p1, p2)
        res[f"{prefijo}_{dedo}_{art[1]}"] = angulo_3puntos(p1, p2, p3)
        res[f"{prefijo}_{dedo}_{art[2]}"] = angulo_3puntos(p2, p3, p4)
    return res


def calcular_angulos_cabeza(pose):
    """Roll, pitch y yaw aproximados con nariz y orejas (0 = neutra)."""
    if not pose or not visible(pose, (0, L_EAR, R_EAR)):
        return {k: np.nan for k in CLAVES_CABEZA}
    n, oi, od = punto(pose, 0), punto(pose, L_EAR), punto(pose, R_EAR)
    mid = (oi + od) / 2
    dist = np.linalg.norm(oi[:2] - od[:2]) + 1e-6
    return {
        "cabeza_roll": float(np.degrees(np.arctan2(oi[1] - od[1], oi[0] - od[0]))),
        "cabeza_pitch": float(np.degrees(np.arctan2(n[1] - mid[1], dist))),
        "cabeza_yaw": float(np.degrees(np.arctan2(od[2] - oi[2], oi[0] - od[0]))),
    }


def bloque_landmarks(lista, n_puntos):
    if lista:
        return np.array([[lm.x, lm.y, lm.z] for lm in lista]).flatten()
    return np.full(n_puntos * 3, np.nan)


def encabezados_landmarks():
    c = ["frame", "tiempo_ms"]
    for i in range(33):
        c += [f"pose_{i}_x", f"pose_{i}_y", f"pose_{i}_z"]
    for pref in ("manoI", "manoD"):
        for i in range(21):
            c += [f"{pref}_{i}_x", f"{pref}_{i}_y", f"{pref}_{i}_z"]
    for i in range(33):
        c += [f"mundo_pose_{i}_x", f"mundo_pose_{i}_y", f"mundo_pose_{i}_z"]
    for pref in ("manoI", "manoD"):
        for i in range(21):
            c += [f"mundo_{pref}_{i}_x", f"mundo_{pref}_{i}_y", f"mundo_{pref}_{i}_z"]
    return c + encabezados_calidad()


def encabezados_calidad():
    """Confianza de deteccion por frame (al final de landmarks.csv; los archivos viejos no la traen):
    pose_i_vis = visibilidad (0-1, probabilidad de que el punto este visible y no ocluido);
    pose_i_pres = presencia (0-1, probabilidad de que el punto este dentro de la imagen);
    manoI/manoD_presente = 1 si el modelo devolvio esa mano; manoI/manoD_conf = confianza de la mano (solo detector
    Pose+Manos; Holistic no la entrega -> NaN); img_ancho_px / img_alto_px = resolucion del video (para medir distancias
    sin deformarlas en videos verticales)."""
    c = [f"pose_{i}_vis" for i in range(33)] + [f"pose_{i}_pres" for i in range(33)]
    return c + ["manoI_presente", "manoD_presente", "manoI_conf", "manoD_conf", "img_ancho_px", "img_alto_px"]


def _attr(lm, nombre):
    v = getattr(lm, nombre, None)
    return float(v) if v is not None else np.nan


def bloque_calidad(pose, izq, der, conf_izq, conf_der):
    if pose:
        vis = [_attr(p, "visibility") for p in pose]
        pres = [_attr(p, "presence") for p in pose]
    else:
        vis = pres = [np.nan] * 33
    return vis + pres + [1.0 if izq else 0.0, 1.0 if der else 0.0, conf_izq, conf_der,
                         float(ESCALA[0]), float(ESCALA[1])]


def encabezados_angulos():
    return (["frame", "tiempo_ms"] + ["ang_" + k for k in CLAVES_BRAZO]
            + ["ang_" + k for k in CLAVES_MANO_I] + ["ang_" + k for k in CLAVES_MANO_D]
            + ["ang_" + k for k in CLAVES_CABEZA])


# ======================================================================================================================
# 4. EXTRACCION DE LANDMARKS (MediaPipe Tasks: Holistic o Pose + Manos)
# ======================================================================================================================
def elegir_detector(mp, preferido=None):
    preferido = preferido or DETECTOR
    tiene_holistic = hasattr(mp.tasks.vision, "HolisticLandmarker")
    if preferido == "holistic":
        if not tiene_holistic:
            raise RuntimeError(f"MediaPipe {mp.__version__} no incluye HolisticLandmarker en Python. "
                               "Usa --detector auto o pose+manos, o instala una version que lo traiga.")
        return "holistic"
    if preferido == "pose+manos":
        return "pose+manos"
    return "holistic" if tiene_holistic else "pose+manos"


class _DetectorHolistic:
    """HolisticLandmarker (pose 33 + manos 21 + cara). Resultados con listas planas (una sola persona)."""
    nombre = "holistic"

    def __init__(self, mp):
        asegurar_modelo(URL_HOLISTIC, MODELO_HOLISTIC)
        v = mp.tasks.vision
        opciones = v.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODELO_HOLISTIC),
            running_mode=v.RunningMode.VIDEO,
            min_face_detection_confidence=CONF_DETECCION,
            min_pose_detection_confidence=CONF_DETECCION,
            min_hand_landmarks_confidence=CONF_LANDMARKS_MANO,
        )
        self._det = v.HolisticLandmarker.create_from_options(opciones)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._det.close()

    def detectar(self, imagen_mp, t_ms, ancho, alto):
        r = self._det.detect_for_video(imagen_mp, t_ms)
        return (r.pose_landmarks, r.pose_world_landmarks, r.left_hand_landmarks, r.right_hand_landmarks,
                getattr(r, "left_hand_world_landmarks", None), getattr(r, "right_hand_world_landmarks", None),
                np.nan, np.nan)                     # Holistic no entrega la confianza de cada mano


class _DetectorPoseManos:
    """Respaldo para versiones de MediaPipe sin HolisticLandmarker: PoseLandmarker + HandLandmarker.
    Cada mano se asigna a la muneca de la pose mas cercana (mas fiable que la etiqueta del modelo de mano)."""
    nombre = "pose+manos"

    def __init__(self, mp):
        asegurar_modelo(URL_POSE, MODELO_POSE)
        asegurar_modelo(URL_MANO, MODELO_MANO)
        v = mp.tasks.vision
        self._pose = v.PoseLandmarker.create_from_options(v.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODELO_POSE), running_mode=v.RunningMode.VIDEO,
            num_poses=1, min_pose_detection_confidence=CONF_DETECCION, min_pose_presence_confidence=CONF_DETECCION,
            min_tracking_confidence=CONF_SEGUIMIENTO))
        self._manos = v.HandLandmarker.create_from_options(v.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODELO_MANO), running_mode=v.RunningMode.VIDEO,
            num_hands=2, min_hand_detection_confidence=CONF_DETECCION,
            min_hand_presence_confidence=CONF_LANDMARKS_MANO, min_tracking_confidence=CONF_SEGUIMIENTO))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._pose.close()
        self._manos.close()

    def detectar(self, imagen_mp, t_ms, ancho, alto):
        rp = self._pose.detect_for_video(imagen_mp, t_ms)
        rm = self._manos.detect_for_video(imagen_mp, t_ms)
        pose = rp.pose_landmarks[0] if rp.pose_landmarks else None
        pose_w = rp.pose_world_landmarks[0] if rp.pose_world_landmarks else None
        manos = {"izq": (None, None, np.nan), "der": (None, None, np.nan)}
        for lado, k in asignar_manos(rm, pose, ancho, alto).items():
            manos[lado] = (rm.hand_landmarks[k], rm.hand_world_landmarks[k] if rm.hand_world_landmarks else None,
                           float(rm.handedness[k][0].score) if rm.handedness else np.nan)
        return (pose, pose_w, manos["izq"][0], manos["der"][0], manos["izq"][1], manos["der"][1],
                manos["izq"][2], manos["der"][2])


def asignar_manos(res_manos, pose, ancho, alto):
    """{'izq': k, 'der': k}: que mano detectada corresponde a la muneca izquierda (15) y derecha (16) de la pose.
    Respaldo: la etiqueta del modelo de mano, invertida (el modelo supone imagen en espejo y el video no lo esta)."""
    n = min(len(res_manos.hand_landmarks), 2)
    if n == 0:
        return {}
    esc = np.array([ancho, alto], float)
    pos = [np.array([h[0].x, h[0].y]) * esc for h in res_manos.hand_landmarks[:n]]
    munecas = {}
    if pose is not None:
        for lado, idx in (("izq", 15), ("der", 16)):
            vis = getattr(pose[idx], "visibility", None)
            if vis is None or vis >= VISIBILIDAD_MINIMA:
                munecas[lado] = np.array([pose[idx].x, pose[idx].y]) * esc
    umbral = DIST_MAX_MUNECA * ancho
    if munecas:
        if n == 1:
            lado, d = min(((l, float(np.linalg.norm(pos[0] - p))) for l, p in munecas.items()), key=lambda x: x[1])
            if d <= umbral:
                return {lado: 0}
        else:
            mejor = None
            for orden in (("izq", "der"), ("der", "izq")):
                costo, ok = 0.0, True
                for k, lado in enumerate(orden):
                    if lado in munecas:
                        d = float(np.linalg.norm(pos[k] - munecas[lado]))
                        costo += d
                        ok &= d <= umbral
                    else:
                        costo += float(ancho)
                if mejor is None or costo < mejor[0]:
                    mejor = (costo, orden, ok)
            if mejor[2]:
                return {mejor[1][0]: 0, mejor[1][1]: 1}
    out = {}
    for k in range(n):
        lado = "der" if res_manos.handedness[k][0].category_name == "Left" else "izq"
        out.setdefault(lado, k)
    return out


def tiempos_finales(t_pos, fps):
    """Marca de tiempo (ms, enteros estrictamente crecientes) de cada frame.
    Se usan las marcas REALES del contenedor (CAP_PROP_POS_MSEC) si son coherentes (todas validas, crecientes y con
    una duracion parecida a la nominal); si no, frame / fps. Asi los videos de frecuencia variable (celulares) no
    distorsionan velocidades ni aceleraciones."""
    t_pos = np.asarray(t_pos, float)
    n = len(t_pos)
    nominal = np.arange(n) * 1000.0 / fps
    t, fuente = nominal, "nominal (frame / fps)"
    if n >= 2 and np.isfinite(t_pos).all():
        rel = t_pos - t_pos[0]
        if (np.diff(rel) > 0).all() and nominal[-1] > 0 and 0.5 <= rel[-1] / nominal[-1] <= 2.0:
            t, fuente = rel, "marcas de tiempo del video"
    ms = np.round(t).astype(np.int64)
    for k in range(1, n):
        ms[k] = max(ms[k], ms[k - 1] + 1)
    return ms, fuente


_CLAHE = None


def preprocesar_luz(frame):
    """Ecualizacion adaptativa de la luminancia (CLAHE en el canal L de Lab). Solo si PREPROCESO_LUZ = True."""
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[..., 0] = _CLAHE.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def extraer(video, detector=None):
    """Video -> (landmarks DataFrame, angulos DataFrame, fps nominal, info dict). Recorre TODOS los frames."""
    mp = importar_mediapipe()
    tipo = elegir_detector(mp, detector)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ErrorAnalisis(f"No se pudo abrir el video: {video}")
    fps = fps_valido(cap.get(cv2.CAP_PROP_FPS))
    ancho, alto = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ESCALA[:] = [ancho, alto, ancho]
    print(f"MediaPipe {mp.__version__} | detector: {tipo} | {ancho}x{alto} | {fps:.2f} fps (nominal)")

    filas_lm, filas_ang, t_pos = [], [], []
    ultimo_mp, n = -1, 0
    clase = _DetectorHolistic if tipo == "holistic" else _DetectorPoseManos
    try:
        with clase(mp) as det:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                n += 1
                tp = cap.get(cv2.CAP_PROP_POS_MSEC)
                t_pos.append(tp if tp is not None and np.isfinite(tp) else np.nan)
                t_nom = (n - 1) * 1000.0 / fps
                t_mp = int(round(tp)) if np.isfinite(t_pos[-1]) and tp > 0 else int(round(t_nom))
                t_mp = max(t_mp, ultimo_mp + 1)            # MediaPipe exige marcas estrictamente crecientes
                ultimo_mp = t_mp
                rgb = cv2.cvtColor(preprocesar_luz(frame) if PREPROCESO_LUZ else frame, cv2.COLOR_BGR2RGB)
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                pose, pose_w, izq, der, izq_w, der_w, c_izq, c_der = det.detectar(img, t_mp, ancho, alto)
                if INVERTIR_LADOS:
                    izq, der, izq_w, der_w, c_izq, c_der = der, izq, der_w, izq_w, c_der, c_izq
                vec = np.concatenate([bloque_landmarks(pose, 33), bloque_landmarks(izq, 21),
                                      bloque_landmarks(der, 21), bloque_landmarks(pose_w, 33),
                                      bloque_landmarks(izq_w, 21), bloque_landmarks(der_w, 21)])
                filas_lm.append([n, 0] + list(vec) + bloque_calidad(pose, izq, der, c_izq, c_der))
                ab = calcular_angulos_brazo(pose, pose_w)
                ai = calcular_angulos_mano(izq, "manoI")
                ad = calcular_angulos_mano(der, "manoD")
                ac = calcular_angulos_cabeza(pose)
                filas_ang.append([n, 0] + [ab[k] for k in CLAVES_BRAZO] + [ai[k] for k in CLAVES_MANO_I]
                                 + [ad[k] for k in CLAVES_MANO_D] + [ac[k] for k in CLAVES_CABEZA])
                if n % 50 == 0:
                    print(f"  frames procesados: {n}")
    finally:
        cap.release()
    if n == 0:
        raise ErrorAnalisis("El video no tiene frames.")
    t_ms, fuente = tiempos_finales(t_pos, fps)
    lm = pd.DataFrame(filas_lm, columns=encabezados_landmarks())
    an = pd.DataFrame(filas_ang, columns=encabezados_angulos())
    lm["tiempo_ms"] = t_ms
    an["tiempo_ms"] = t_ms
    fps_real = (n - 1) / (t_ms[-1] / 1000.0) if n > 1 and t_ms[-1] > 0 else fps
    print(f"Total de frames: {n} | tiempo: {fuente} | fps efectivo {fps_real:.2f}")
    info = {"detector": tipo, "mediapipe": mp.__version__, "fuente_tiempo": fuente, "fps_nominal": fps,
            "fps_efectivo": fps_real, "resolucion": f"{ancho}x{alto}"}
    return lm, an, fps, info


# ======================================================================================================================
# 5. CALIBRACION DE ANGULOS (por persona / camara)
# ======================================================================================================================
# La calibracion reescala cada articulacion de la mano entre "mano abierta" (0 de flexion) y "puno" (ROM de esa
# articulacion) y aplica un desfase constante a codo, hombro y cabeza medido con los brazos extendidos.
MIN_FRAMES_CAL = 15        # frames minimos de una pose para fiarse de ella (calibracion automatica)
SEPARACION_MIN = 40.0      # grados de flexion media entre puno y mano abierta para distinguirlos
LADO_MANO = {"manoI": "izquierda", "manoD": "derecha"}


def calcular_calibracion(abierta, puno, brazos, min_frames=1):
    """abierta / puno: dict prefijo -> DataFrame de angulos con esa pose; brazos: DataFrame con brazos extendidos.
    Devuelve el dict de calibracion {"mano": {col: [abierta, puno]}, "offset": {col: grados}}."""
    cal = {"mano": {}, "offset": {}}
    for pref in ("manoI", "manoD"):
        ab, pu = abierta.get(pref, pd.DataFrame()), puno.get(pref, pd.DataFrame())
        if len(ab) < min_frames or len(pu) < min_frames:
            print(f"  mano {LADO_MANO[pref]}: faltan frames (abierta {len(ab)}, puno {len(pu)}); NO se calibra")
            continue
        for c in claves_mano(pref):
            col = "ang_" + c
            m_ab, m_pu = ab[col].median(), pu[col].median()
            if np.isfinite(m_ab) and np.isfinite(m_pu) and m_ab - m_pu > 20:
                cal["mano"][col] = [float(m_ab), float(m_pu)]
            else:
                print(f"  aviso: {col} no se pudo calibrar (mano no detectada o sin diferencia abierta/puno)")
    if len(brazos) >= min_frames:
        for k in ("codo_izq", "codo_der"):
            m = brazos["ang_" + k].median()
            if np.isfinite(m):
                cal["offset"]["ang_" + k] = float(180 - m)
        for k in ("hombro_izq", "hombro_der"):
            m = brazos["ang_" + k].median()
            if np.isfinite(m):
                cal["offset"]["ang_" + k] = float(HOMBRO_REPOSO - m)
        for k in CLAVES_CABEZA:
            m = brazos["ang_" + k].median()
            if np.isfinite(m):
                cal["offset"]["ang_" + k] = float(-m)
    else:
        print("  brazos/cabeza: no hay suficientes frames con los brazos extendidos; no se calibran")
    return cal


def guardar_calibracion(cal, ruta):
    with open(ruta, "w") as f:
        json.dump(cal, f, indent=1)
    print(f"\nCalibracion guardada: {ruta}")
    print(f"  articulaciones de mano calibradas: {len(cal['mano'])} de 30")
    print(f"  offsets de brazo/cabeza: {len(cal['offset'])} de 7")
    revisar_calibracion(ruta)


def ruta_calibracion_salida(persona=None, salida=None):
    if salida:
        return os.path.abspath(salida)
    if persona:
        return os.path.join(BASE, f"calibracion_{persona.strip().lower()}.json")
    return CALIB_JSON


def calibrar(v_abierta, v_puno, v_brazos, ruta=CALIB_JSON, detector=None):
    """Calibracion clasica con 3 videos cortos (3 a 5 s): mano abierta, puno, brazos extendidos."""
    for v in (v_abierta, v_puno, v_brazos):
        if not os.path.exists(v):
            raise SystemExit(f"No existe el video de calibracion: {v}")
    print("Calibracion 1/3: mano abierta")
    an_a = extraer(v_abierta, detector)[1]
    print("Calibracion 2/3: puno cerrado")
    an_p = extraer(v_puno, detector)[1]
    print("Calibracion 3/3: brazos extendidos y cabeza al frente")
    an_b = extraer(v_brazos, detector)[1]
    cal = calcular_calibracion({"manoI": an_a, "manoD": an_a}, {"manoI": an_p, "manoD": an_p}, an_b)
    guardar_calibracion(cal, ruta)


def resolver_calibracion(persona, calib=None):
    """Archivo de calibracion a usar: el indicado, el propio de la persona o el general."""
    if calib:
        return os.path.abspath(calib)
    propia = os.path.join(BASE, f"calibracion_{persona.lower()}.json")
    if os.path.exists(propia):
        print(f"Usando la calibracion de {persona}: {os.path.basename(propia)}")
        return propia
    return CALIB_JSON


def aplicar_calibracion(an, ruta=CALIB_JSON):
    """Devuelve (angulos calibrados y filtrados, calibrado: bool). Los angulos originales se conservan en el
    DataFrame de entrada (se trabaja sobre una copia)."""
    an = an.copy()
    calibrado = os.path.exists(ruta)
    if calibrado:
        with open(ruta) as f:
            cal = json.load(f)
        for col, (ab, pu) in cal["mano"].items():
            if col not in an.columns:
                continue
            rom = ROM[col.split("_")[-1]]
            flex = (ab - an[col]) / (ab - pu) * rom
            an[col] = 180 - flex.clip(-10, rom * 1.1)
        for col, off in cal["offset"].items():
            if col in an.columns:
                an[col] = an[col] + off
        for col in ("ang_codo_izq", "ang_codo_der"):
            an[col] = an[col].clip(0, 180)
    else:
        print(f"AVISO: no existe {os.path.basename(ruta)}; los angulos van SIN calibrar "
              "(usa el subcomando 'calibrar' o 'calibrar-auto').")
    cols = [c for c in an.columns if c.startswith("ang_")]
    # filtro de mediana: quita picos aislados sin desplazar el gesto en el tiempo (ventana en frames)
    an[cols] = an[cols].rolling(FILTRO_VENTANA, center=True, min_periods=1).median()
    return an, calibrado


def revisar_calibracion(ruta, breve=False):
    """Si la diferencia medida entre mano abierta y puno es pequena, el reescalado multiplica el ruido: se avisa."""
    if not os.path.exists(ruta):
        print(f"No existe {ruta}")
        return
    with open(ruta) as f:
        cal = json.load(f)
    malas, filas = [], []
    for col, (ab, pu) in cal["mano"].items():
        rom = ROM.get(col.split("_")[-1], 90)
        span = ab - pu
        ganancia = rom / span if span > 0 else np.inf
        estado = "OK" if ganancia <= 1.4 else ("ganancia alta" if ganancia <= 2.0 else "POCO FIABLE")
        filas.append((col, ab, pu, span, ganancia, estado))
        if estado != "OK":
            malas.append(col)
    n = len(cal["mano"])
    if breve:
        msg = (f"Calibracion {os.path.basename(ruta)}: {n}/30 articulaciones de mano, "
               f"{len(malas)} con ganancia alta o poco fiables")
        print(msg + (f" (revisa con: py -3.12 sistema_lsc.py revisar-calib {os.path.basename(ruta)})"
                     if malas or n < 28 else ""))
        return
    print(f"\nCalibracion: {ruta}\nArticulaciones de mano calibradas: {n} de 30")
    print(f"{'articulacion':32s} {'abierta':>8s} {'puno':>8s} {'dif.':>7s} {'ganancia':>9s}  estado")
    for col, ab, pu, span, g, est in sorted(filas, key=lambda r: -r[4]):
        print(f"{col:32s} {ab:8.1f} {pu:8.1f} {span:7.1f} {g:9.2f}  {est}")
    print("\nOffsets de brazo/cabeza:")
    for col, off in cal["offset"].items():
        print(f"  {col:20s} {off:+7.1f} grados")
    print("\nGanancia = rango anatomico esperado / diferencia medida entre mano abierta y puno.\n"
          "  ~1 -> la camara ve bien ese movimiento; >2 -> la camara lo ve 'aplastado' (dedos que se doblan hacia\n"
          "  la camara en vista frontal) y el reescalado amplifica el ruido. Se arregla con una vista sagital o\n"
          "  repitiendo la calibracion con la palma bien de frente, buena luz y 60 fps.")


# ---------------- calibracion automatica (videos con toda la secuencia) ----------------
def otsu(x, bins=64):
    """Umbral de Otsu sobre un histograma 1D: separa dos grupos (mano abierta / puno)."""
    h, e = np.histogram(x, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    s = np.cumsum(h * c)
    m0 = s / np.maximum(w0, 1)
    m1 = (s[-1] - s) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def flexion_media(an, pref):
    """Flexion media (indice a menique, MCP y PIP) por frame. 0 = dedos rectos."""
    cols = [f"ang_{pref}_{d}_{a}" for d in DEDOS_FLEX for a in ("mcp", "pip")]
    S = np.column_stack([pd.to_numeric(an[c], errors="coerce").to_numpy(float) for c in cols])
    valido = np.isfinite(S).sum(axis=1) >= 6
    return np.nanmean(180.0 - S, axis=1), valido


def frames_estables(fx, t, ok):
    fs = (pd.Series(fx).interpolate(limit=5, limit_direction="both")
          .rolling(5, center=True, min_periods=1).median().to_numpy())
    v = np.abs(np.gradient(fs, t))
    lim = np.nanpercentile(v[ok], 50) if ok.sum() else 0
    return fs, (v <= lim)


def seleccionar_mano(an, pref, t, solo_abierta=False):
    fx, valido = flexion_media(an, pref)
    ok = valido & np.isfinite(fx)
    if ok.sum() < MIN_FRAMES_CAL:
        return None
    fs, est = frames_estables(fx, t, ok)
    if solo_abierta:
        abierta = ok & est & (fs <= np.nanmedian(fs[ok]))
        return {"fx": fs, "abierta": abierta, "puno": np.zeros(len(fx), bool), "sep": np.nan, "umbral": np.nan}
    umbral = otsu(fs[ok])
    abierta = ok & est & (fs < umbral)
    puno = ok & est & (fs >= umbral)
    if abierta.sum() >= MIN_FRAMES_CAL:                     # el 60 % mas extremo de cada grupo
        abierta &= fs <= np.percentile(fs[abierta], 60)
    if puno.sum() >= MIN_FRAMES_CAL:
        puno &= fs >= np.percentile(fs[puno], 40)
    sep = (np.nanmedian(fs[puno]) - np.nanmedian(fs[abierta])) if abierta.any() and puno.any() else np.nan
    return {"fx": fs, "abierta": abierta, "puno": puno, "sep": sep, "umbral": umbral}


def seleccionar_brazos(an):
    def col(n):
        return pd.to_numeric(an[n], errors="coerce").to_numpy(float)

    ce, cd, he, hd = col("ang_codo_izq"), col("ang_codo_der"), col("ang_hombro_izq"), col("ang_hombro_der")
    for cmin, hmax, nota in ((160, 30, ""), (150, 45, " (criterio relajado: brazos no del todo rectos o pegados)")):
        m = (ce >= cmin) & (cd >= cmin) & (he <= hmax) & (hd <= hmax)
        if m.sum() >= MIN_FRAMES_CAL:
            return m, nota
    return np.zeros(len(an), bool), " (no se encontro la pose de brazos extendidos)"


def tramo_segundos(texto_tramo):
    a, b = texto_tramo.split("-")
    return float(a), float(b)


def graficar_seleccion(ruta, t, sel, brazos, an, titulo):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    col = {"manoI": "#1f77b4", "manoD": "#d62728"}
    for pref, s in sel.items():
        if s is None:
            continue
        ax1.plot(t, s["fx"], color=col[pref], lw=1.3, label=f"flexion media mano {LADO_MANO[pref]}")
        ax1.plot(t[s["abierta"]], s["fx"][s["abierta"]], ".", color="green", ms=6)
        ax1.plot(t[s["puno"]], s["fx"][s["puno"]], ".", color="black", ms=6)
    ax1.plot([], [], ".", color="green", label="elegidos como MANO ABIERTA")
    ax1.plot([], [], ".", color="black", label="elegidos como PUNO")
    ax1.set_ylabel("Flexion media de dedos (°)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)
    for n_, c in (("ang_codo_izq", "#1f77b4"), ("ang_codo_der", "#d62728")):
        ax2.plot(t, pd.to_numeric(an[n_], errors="coerce"), color=c, lw=1.2, label=n_.replace("ang_", ""))
    for n_, c in (("ang_hombro_izq", "#1f77b4"), ("ang_hombro_der", "#d62728")):
        ax2.plot(t, pd.to_numeric(an[n_], errors="coerce"), color=c, lw=1.2, ls="--", label=n_.replace("ang_", ""))
    ax2.plot(t[brazos], np.full(brazos.sum(), -5.0), "|", color="#2ca02c", ms=10,
             label="elegidos como BRAZOS EXTENDIDOS")
    ax2.set_ylabel("Angulo (°)")
    ax2.set_xlabel("Tiempo (s)")
    ax2.legend(fontsize=8, ncol=3, loc="center right")
    ax2.grid(alpha=0.3)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def calibrar_auto(videos, persona, solo_abierta=(), salida=None, abierta=None, puno=None, brazos_tr=None,
                  detector=None):
    """Calibra a UNA persona con videos que traen toda la secuencia (mano abierta, puno y brazos extendidos).
    Detecta cada pose sola y guarda una grafica por video con los frames elegidos: REVISALA."""
    manual = any([abierta, puno, brazos_tr])
    if manual and len(videos) != 1:
        raise SystemExit("Los tramos manuales (--abierta/--puno/--brazos) solo valen con UN video.")
    for v in list(videos) + list(solo_abierta):
        if not os.path.exists(v):
            raise SystemExit(f"No existe el video: {v}")
    persona = persona.strip().title()
    salida = ruta_calibracion_salida(persona, salida)
    pool = {"abierta": {"manoI": [], "manoD": []}, "puno": {"manoI": [], "manoD": []}, "brazos": []}

    for ruta, solo_ab in [(v, False) for v in videos] + [(v, True) for v in solo_abierta]:
        print(f"\n=== {os.path.basename(ruta)} {'(solo palmas abiertas)' if solo_ab else ''}")
        _, an, _, _ = extraer(ruta, detector)
        t = pd.to_numeric(an["tiempo_ms"], errors="coerce").to_numpy(float) / 1000.0
        sel = {pref: seleccionar_mano(an, pref, t, solo_abierta=solo_ab) for pref in ("manoI", "manoD")}
        brazos, nota = (np.zeros(len(an), bool), "") if solo_ab else seleccionar_brazos(an)

        if manual:   # tramos dados a mano (segundos)
            def en(tr):
                return (t >= tr[0]) & (t <= tr[1]) if tr else np.zeros(len(t), bool)
            for pref, s in sel.items():
                if s is not None:
                    fx, valido = flexion_media(an, pref)
                    ok = valido & np.isfinite(fx)
                    s["abierta"], s["puno"] = ok & en(abierta), ok & en(puno)
            if brazos_tr:
                brazos = en(brazos_tr)

        for pref, s in sel.items():
            lado = LADO_MANO[pref]
            if s is None:
                print(f"  mano {lado}: casi no se detecto en este video")
                continue
            parecidas = (not solo_ab and not manual and np.isfinite(s["sep"]) and s["sep"] < SEPARACION_MIN)
            print(f"  mano {lado}: {s['abierta'].sum()} frames de mano abierta, {s['puno'].sum()} de puno"
                  + ("" if solo_ab or np.isnan(s["sep"]) else f", separacion {s['sep']:.0f} grados de flexion media")
                  + ("   <-- ABIERTA Y PUNO SE PARECEN: revisa la grafica" if parecidas else ""))
            if solo_ab or manual or (np.isfinite(s["sep"]) and s["sep"] >= SEPARACION_MIN):
                pool["abierta"][pref].append(an[s["abierta"]])
                pool["puno"][pref].append(an[s["puno"]])
        if not solo_ab:
            print(f"  brazos extendidos: {brazos.sum()} frames{nota}")
            pool["brazos"].append(an[brazos])

        png = os.path.join(BASE, f"calibracion_{persona.lower()}_"
                                 f"{os.path.splitext(os.path.basename(ruta))[0]}_seleccion.png")
        graficar_seleccion(png, t, sel, brazos, an, f"Seleccion de poses: {os.path.basename(ruta)}")
        print(f"  grafica de verificacion: {os.path.basename(png)}")

    print("\n=== Resultado")
    unir = lambda lista: pd.concat(lista) if lista else pd.DataFrame()   # noqa: E731
    cal = calcular_calibracion({p: unir(pool["abierta"][p]) for p in ("manoI", "manoD")},
                               {p: unir(pool["puno"][p]) for p in ("manoI", "manoD")},
                               unir(pool["brazos"]), min_frames=MIN_FRAMES_CAL)
    guardar_calibracion(cal, salida)
    if os.path.abspath(salida) == os.path.abspath(CALIB_JSON):
        propias = sorted(os.path.basename(f) for f in glob.glob(os.path.join(BASE, "calibracion_*.json"))
                         if not os.path.basename(f).startswith("calibracion_respaldo_"))
        print(f"\nCalibracion GENERAL guardada como {os.path.basename(salida)}: se usara para todas las personas que no "
              "tengan calibracion propia.")
        if propias:
            print("  OJO: estas calibraciones por persona tienen prioridad sobre la general (borralas o muevelas si "
                  "no quieres usarlas): " + ", ".join(propias))
    else:
        print(f"\nAhora analiza los videos de {persona} normalmente (escribe '{persona}' como persona); se usara "
              f"{os.path.basename(salida)} solo.")


# ======================================================================================================================
# 6. EXCEL DE LANDMARKS Y ANGULOS (con hoja de leyenda)
# ======================================================================================================================
def leyenda_landmarks():
    return [("Columna", "Significado"),
            ("frame", "Numero de frame del video (empieza en 1)"),
            ("tiempo_ms", "Tiempo desde el inicio del video, en milisegundos (marcas reales del video si son coherentes)"),
            ("pose_0 ... pose_32", "33 puntos del cuerpo (x, y, z). Ej: 11/12 hombros, 13/14 codos, 15/16 munecas"),
            ("manoI_0 ... manoI_20", "21 puntos de la mano izquierda (0 = muneca, 4/8/12/16/20 = puntas)"),
            ("manoD_0 ... manoD_20", "21 puntos de la mano derecha"),
            ("x, y", "Posicion normalizada 0-1 (x: izquierda a derecha, y: arriba a abajo)"),
            ("z", "Profundidad relativa (negativo = mas cerca de la camara)"),
            ("mundo_*", "Coordenadas 3D estimadas en metros (pose: origen en la cadera; mano: centro de la mano)"),
            ("pose_i_vis", "Visibilidad 0-1 del punto i (probabilidad de que este visible y no ocluido)"),
            ("pose_i_pres", "Presencia 0-1 del punto i (probabilidad de que este dentro de la imagen)"),
            ("manoI/manoD_presente", "1 = el modelo detecto esa mano en el frame; 0 = no"),
            ("manoI/manoD_conf", "Confianza 0-1 de la mano (solo detector Pose+Manos; con Holistic queda vacia)"),
            ("img_ancho_px / img_alto_px", "Resolucion del video (pixeles)"),
            ("Celda vacia", "No se detecto esa parte en ese frame"),
            ("Lados", f"INVERTIR_LADOS = {INVERTIR_LADOS}")]


def leyenda_angulos(calibrado=False):
    return [("Columna", "Significado"),
            ("ang_codo_*", "Angulo hombro-codo-muneca. ~180 = estirado; bajo = doblado"),
            ("ang_hombro_*", "Angulo cadera-hombro-codo. ~0-20 = brazo pegado; ~90 = horizontal; ~180 = arriba"),
            ("ang_mano*_dedo_mcp/pip/dip", "Angulo de cada articulacion del dedo. ~180 = recto; ~90 = doblado"),
            ("Pulgar", "cmc = base, mcp = nudillo de la base, ip = nudillo de la punta"),
            ("ang_cabeza_roll", "Inclinacion lateral de la cabeza (0 = neutra), estimada con las orejas"),
            ("ang_cabeza_pitch", "Cabeceo (0 = neutra; positivo = mirando hacia abajo)"),
            ("ang_cabeza_yaw", "Giro de la cabeza (0 = neutra), estimado con la profundidad de las orejas"),
            ("Celda vacia", "No se detecto la mano (o baja visibilidad) en ese frame"),
            ("Calibracion", f"Calibrados por persona y filtrados con mediana de {FILTRO_VENTANA} frames"
             if calibrado else f"SIN calibrar (solo filtro de mediana de {FILTRO_VENTANA} frames)"),
            ("Nota", "Angulos estimados por vision artificial, no mediciones clinicas"),
            ("Lados", f"INVERTIR_LADOS = {INVERTIR_LADOS}")]


def formato_datos(ws, filas, cols, color_encabezado="1F4E78"):
    par = PatternFill(start_color="DCE6F1", end_color="DCE6F1", fill_type="solid")
    for c in range(1, cols + 1):
        cel = ws.cell(row=1, column=c)
        cel.font = Font(bold=True, color="FFFFFF", size=10)
        cel.fill = PatternFill(start_color=color_encabezado, end_color=color_encabezado, fill_type="solid")
        cel.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(c)].width = 14
    ws.freeze_panes = "A2"
    for fila in range(2, filas + 2, 2):
        for c in range(1, cols + 1):
            ws.cell(row=fila, column=c).fill = par


def formato_leyenda(ws):
    for c in (1, 2):
        cel = ws.cell(row=1, column=c)
        cel.font = Font(bold=True, color="FFFFFF", size=11)
        cel.fill = PatternFill(start_color="548235", end_color="548235", fill_type="solid")
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 95
    for fila in ws.iter_rows(min_row=2):
        for cel in fila:
            cel.alignment = Alignment(vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"


def guardar_datos(df, hoja, leyenda, ruta, escala_color=False):
    with pd.ExcelWriter(ruta, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=hoja, index=False)
        pd.DataFrame(leyenda).to_excel(w, sheet_name="Leyenda", index=False, header=False)
    wb = load_workbook(ruta)
    formato_datos(wb[hoja], len(df), len(df.columns))
    formato_leyenda(wb["Leyenda"])
    if escala_color:
        regla = ColorScaleRule(start_type="min", start_color="F8696B", mid_type="percentile",
                               mid_value=50, mid_color="FFEB84", end_type="max", end_color="63BE7B")
        for c in range(3, len(df.columns) + 1):
            letra = get_column_letter(c)
            wb[hoja].conditional_formatting.add(f"{letra}2:{letra}{len(df) + 1}", regla)
    wb.save(ruta)


# ======================================================================================================================
# 7. ANALISIS DEL GESTO: brazo dominante, tramo activo y fases
# ======================================================================================================================
SENALES = {
    "v": ("Velocidad muneca (rel.)", 2),
    "d": ("Distancia a objetivo (anchos de hombro)", 2),
    "codo": ("Codo (grados)", 0),
    "hombro": ("Hombro (grados)", 0),
    "flex": ("Flexion dedos (grados)", 0),
}

LUGAR_SENA = {
    # sena: textos del lugar donde se hace y del verbo de llegada, en ingles y espanol. Agrega aqui senas nuevas
    # (y su punto en objetivo()) para que las fases digan "hacia la barbilla", etc.
    "AGUA": {"en": ("the chin", "rises"), "es": ("la barbilla", "sube")},
    "PENSAR": {"en": ("the temple", "rises"), "es": ("la sien", "sube")},
    "MIO": {"en": ("the chest", "moves"), "es": ("el pecho", "se mueve")},
    "AYUDA": {"en": ("the other hand", "moves"), "es": ("la otra mano", "se mueve")},
}
LUGAR_DEFECTO = {"en": ("the sign position", "moves"), "es": ("el punto de la seña", "se mueve")}

# codigo interno -> {idioma: (nombre corto para la figura, descripcion)}.  {l} = lugar, {v} = verbo
FASES_TXT = {
    "Preparacion (reposo inicial)": {
        "en": ("Start: arm at rest", "The arm is still at rest; the sign has not started yet."),
        "es": ("Inicio: brazo en reposo", "El brazo aún está en reposo; la seña no ha comenzado.")},
    "Transporte al objetivo": {
        "en": ("Hand {v} to {l}", "The hand travels toward {l}."),
        "es": ("La mano {v} hacia {l}", "La mano se desplaza hacia {l}.")},
    "Contacto con el objetivo": {
        "en": ("Hand holds at {l}", "The hand stays at {l} with almost no movement."),
        "es": ("La mano se mantiene en {l}", "La mano permanece en {l} casi sin moverse.")},
    "Cierre de mano": {
        "en": ("Hand closes", "The fingers of the dominant hand curl and the hand closes."),
        "es": ("La mano se cierra", "Los dedos de la mano dominante se flexionan y la mano se cierra.")},
    "Apertura de mano": {
        "en": ("Hand opens", "The fingers of the dominant hand extend and the hand opens."),
        "es": ("La mano se abre", "Los dedos de la mano dominante se extienden y la mano se abre.")},
    "Retorno / alejamiento": {
        "en": ("Hand moves away from {l}", "The hand moves away from {l}."),
        "es": ("La mano se aleja de {l}", "La mano se aleja de {l}.")},
    "Manos se juntan": {
        "en": ("Hands come together", "The hands move toward each other."),
        "es": ("Las manos se juntan", "Las manos se acercan entre sí.")},
    "Manos se separan": {
        "en": ("Hands move apart", "The hands move away from each other."),
        "es": ("Las manos se separan", "Las manos se alejan entre sí.")},
    "Mano sube": {
        "en": ("Hand rises", "The dominant hand rises."),
        "es": ("La mano sube", "La mano dominante sube.")},
    "Mano baja": {
        "en": ("Hand lowers", "The dominant hand lowers."),
        "es": ("La mano baja", "La mano dominante baja.")},
    "Codo se flexiona": {
        "en": ("Elbow flexes", "The elbow of the dominant arm bends."),
        "es": ("El codo se flexiona", "El codo del brazo dominante se dobla.")},
    "Codo se extiende": {
        "en": ("Elbow extends", "The elbow of the dominant arm straightens."),
        "es": ("El codo se extiende", "El codo del brazo dominante se estira.")},
    "Otra mano se cierra": {
        "en": ("Other hand closes", "The fingers of the non-dominant hand curl."),
        "es": ("La otra mano se cierra", "Los dedos de la mano no dominante se flexionan.")},
    "Otra mano se abre": {
        "en": ("Other hand opens", "The fingers of the non-dominant hand extend."),
        "es": ("La otra mano se abre", "Los dedos de la mano no dominante se extienden.")},
    "Movimiento (ajuste / repeticion)": {
        "en": ("Small adjustment or repeat",
               "The wrist moves, but no clear change in height, hand distance, elbow or hand shape is detected."),
        "es": ("Ajuste o repetición",
               "La muñeca se mueve, pero no se detecta un cambio claro de altura, distancia entre manos, codo ni forma "
               "de la mano.")},
    "Manos juntas": {
        "en": ("Hands held together", "The hands stay close together with almost no movement."),
        "es": ("Manos juntas", "Las manos se mantienen juntas casi sin moverse.")},
    "Sostenimiento final / reposo": {
        "en": ("End: arm back at rest", "The sign is finished and the arm settles back at rest."),
        "es": ("Final: brazo en reposo", "La seña terminó y el brazo vuelve al reposo.")},
    "Sostenimiento": {
        "en": ("Pose held", "The pose is held with almost no movement."),
        "es": ("Postura sostenida", "La postura se mantiene casi sin movimiento.")},
}


def serie(df, nombre):
    """Columna numerica con huecos cortos (<= 15 frames) interpolados. Un 0 exacto se trata como 'sin dato'
    (formato de archivos antiguos)."""
    s = pd.to_numeric(df[nombre], errors="coerce").replace(0, np.nan)
    return s.interpolate(limit=15, limit_direction="both").to_numpy(dtype=float)


def escala_px(lm):
    """(ancho, alto) en pixeles del video de esta repeticion; (1, 1) en archivos viejos sin esas columnas."""
    if "img_ancho_px" in lm.columns:
        w, h = pd.to_numeric(lm["img_ancho_px"], errors="coerce").median(), \
            pd.to_numeric(lm["img_alto_px"], errors="coerce").median()
        if np.isfinite(w) and np.isfinite(h) and w > 0 and h > 0:
            return float(w), float(h)
    return 1.0, 1.0


def xy(lm, i):
    """Posicion 2D del punto i de la pose en PIXELES (si se conoce la resolucion). Asi una distancia vertical y una
    horizontal valen lo mismo aunque el video sea vertical (1080x1920); todo se divide luego por el ancho de hombros,
    asi que las unidades finales son "anchos de hombro" en cualquier caso."""
    w, h = escala_px(lm)
    return np.column_stack([serie(lm, f"pose_{i}_x") * w, serie(lm, f"pose_{i}_y") * h])


def idx_muneca(lado):
    base = {"izq": 16, "der": 15} if INVERTIR_LADOS else {"izq": 15, "der": 16}
    return base[lado]


def objetivo(sena, lm, idx_m):
    """Punto del cuerpo donde se realiza la sena (para medir la distancia de la mano). None si no esta definido."""
    s = sena.upper().replace("Í", "I")
    if s == "AGUA":
        return (xy(lm, 9) + xy(lm, 10)) / 2
    if s == "PENSAR":
        return xy(lm, 7 if idx_m == 15 else 8)
    if s == "MIO":
        h = (xy(lm, 11) + xy(lm, 12)) / 2
        c = (xy(lm, 23) + xy(lm, 24)) / 2
        return h + 0.3 * (c - h)
    if s == "AYUDA":
        return xy(lm, 15 if idx_m == 16 else 16)
    return None


def remuestrear(y, tt):
    """Serie del tramo activo -> N muestras en tiempo normalizado 0-1 (usa los tiempos reales tt)."""
    ok = ~np.isnan(y)
    if ok.sum() < 5:
        return np.full(N, np.nan)
    tn = (tt - tt[0]) / (tt[-1] - tt[0])
    return np.interp(np.linspace(0, 1, N), tn[ok], y[ok])


def remuestrear_n(y, n):
    """Lleva una serie de cualquier largo a n puntos (para el dataset de secuencias)."""
    y = np.asarray(y, dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return np.full(n, np.nan)
    pos = np.linspace(0, 1, len(y))
    return np.interp(np.linspace(0, 1, n), pos[ok], y[ok])


def analizar(lm, an, sena):
    """Brazo dominante (mayor recorrido de muneca), tramo activo (ver tramo_activo, +- PAD_S) y fases (ver
    segmentar_gesto). Lanza ErrorAnalisis si el video no lo permite."""
    if len(lm) < 20:
        raise ErrorAnalisis("Video demasiado corto.")
    t = pd.to_numeric(lm["tiempo_ms"], errors="coerce").to_numpy(dtype=float) / 1000.0
    ancho = np.nanmedian(np.linalg.norm(xy(lm, 11) - xy(lm, 12), axis=1))
    if not ancho > 0:
        raise ErrorAnalisis("No se detecto el cuerpo (hombros). Revisa que se vea de la cintura para arriba.")

    pos = {l: xy(lm, idx_muneca(l)) / ancho for l in ("izq", "der")}
    recorrido = {l: np.nansum(np.linalg.norm(np.diff(p, axis=0), axis=1)) for l, p in pos.items()}
    lado = max(recorrido, key=recorrido.get)
    otro = "izq" if lado == "der" else "der"

    dt = np.maximum(np.gradient(t), 1e-3)

    def vel(p):
        v = np.linalg.norm(np.gradient(p, axis=0), axis=1) / dt
        return pd.Series(v).rolling(5, center=True, min_periods=1).mean().to_numpy()

    v_dom, v_otro = vel(pos[lado]), vel(pos[otro])
    vref = np.nanpercentile(v_dom, 95)
    if not vref > 0:
        raise ErrorAnalisis("No se detecto movimiento de la muneca.")

    # tramo activo: se mueve CUALQUIERA de las dos manos (senas a dos manos en que la no dominante tambien trabaja)
    v_ambas = np.fmax(v_dom, v_otro)
    tramo = tramo_activo(t, v_ambas, UMBRAL_MOV * vref)
    if tramo is None or tramo[1] - tramo[0] + 1 < 10:
        raise ErrorAnalisis("Tramo activo demasiado corto.")
    i0 = max(0, int(np.searchsorted(t, t[tramo[0]] - PAD_S)))
    i1 = min(len(t) - 1, int(np.searchsorted(t, t[tramo[1]] + PAD_S)))
    sl = slice(i0, i1 + 1)
    tt = t[sl]

    obj = objetivo(sena, lm, idx_muneca(lado))
    d = np.full(len(t), np.nan) if obj is None else np.linalg.norm(pos[lado] - obj / ancho, axis=1)
    def flex_media(l):
        pref = "manoD" if l == "der" else "manoI"
        cols = [f"ang_{pref}_{dd}_{a}" for dd in DEDOS_FLEX for a in ("mcp", "pip")]
        return np.nanmean(np.column_stack([serie(an, c) for c in cols]), axis=1)

    hombros_y = (xy(lm, 11)[:, 1] + xy(lm, 12)[:, 1]) / 2 / ancho
    perfil = {
        "v": remuestrear(np.clip(v_dom[sl] / vref, 0, 3), tt),
        "d": remuestrear(d[sl], tt),
        "codo": remuestrear(serie(an, f"ang_codo_{lado}")[sl], tt),
        "hombro": remuestrear(serie(an, f"ang_hombro_{lado}")[sl], tt),
        "flex": remuestrear(flex_media(lado)[sl], tt),
        # extras para describir fases de senas sin punto objetivo (punto 8)
        "alto": remuestrear((hombros_y - pos[lado][:, 1])[sl], tt),           # + = muneca sobre la linea de hombros
        "alto_otro": remuestrear((hombros_y - pos[otro][:, 1])[sl], tt),
        "dmanos": remuestrear(np.linalg.norm(pos["izq"] - pos["der"], axis=1)[sl], tt),
        "flex_otro": remuestrear(flex_media(otro)[sl], tt),
    }
    if np.isnan(perfil["v"]).all():
        raise ErrorAnalisis("La muneca casi no se detecto en el tramo activo.")
    v_otro_n = remuestrear(np.clip(v_otro[sl] / vref, 0, 3), tt)

    # las fases se segmentan con la mano que mas se mueva en cada instante (dominante u otra). Se normaliza con el
    # percentil 95 del tramo (no con el maximo): un solo pico muy rapido ya no cambia la escala de las demas fases.
    v_ref = pd.Series(np.fmax(perfil["v"], np.nan_to_num(v_otro_n))).rolling(5, center=True, min_periods=1).mean().to_numpy()
    escala = np.nanpercentile(v_ref, 95)
    v_ref = v_ref / (escala if escala > 0 else np.nanmax(v_ref))
    dur = float(tt[-1] - tt[0])
    runs = segmentar_gesto(v_ref, dur)
    avisos = []
    if runs[0][2]:
        avisos.append("el gesto empieza en movimiento (el video pudo empezar tarde)")
    if runs[-1][2]:
        avisos.append("el gesto termina en movimiento (el video pudo cortarse antes del reposo)")
    if i0 == 0 and t[tramo[0]] - t[0] < PAD_S:
        avisos.append("el tramo activo empieza en el primer frame del video")
    if i1 == len(t) - 1 and t[-1] - t[tramo[1]] < PAD_S:
        avisos.append("el tramo activo termina en el ultimo frame del video")
    return {"sena": sena, "t": t, "i0": i0, "i1": i1, "dur": dur, "lado": lado, "otro": otro,
            "perfil": perfil, "v_otro": v_otro_n, "runs": runs, "avisos": avisos,
            "valido": float(np.mean(~np.isnan(pos[lado][sl, 0])) * 100)}


def _corridas(mask):
    """[(inicio, fin_exclusivo, valor), ...] de las corridas de un arreglo booleano."""
    m = np.asarray(mask, bool)
    out, ini = [], 0
    for k in range(1, len(m) + 1):
        if k == len(m) or m[k] != m[ini]:
            out.append((ini, k, bool(m[ini])))
            ini = k
    return out


def tramo_activo(t, v, umbral):
    """(primer, ultimo) frame del gesto, o None. Frames con v > umbral; se descartan las rafagas mas cortas que
    MIN_ACTIVO_S (p. ej. el salto de los landmarks en los primeros frames, cuando MediaPipe aun no se estabiliza) y se
    toma el bloque que contiene el pico de velocidad, uniendo los bloques vecinos separados por pausas menores que
    MAX_PAUSA_ACTIVO_S. Asi un movimiento suelto lejos de la sena (acomodarse, bajar la mano) no alarga el gesto."""
    activo = np.nan_to_num(np.asarray(v, float), nan=0.0) > umbral
    for a, b, val in _corridas(activo):
        if val and t[b - 1] - t[a] < MIN_ACTIVO_S:
            activo[a:b] = False
    bloques = [(a, b) for a, b, val in _corridas(activo) if val]
    if not bloques:
        return None
    k_pico = int(np.argmax(np.where(activo, np.nan_to_num(v, nan=0.0), -np.inf)))
    j = next(i for i, (a, b) in enumerate(bloques) if a <= k_pico < b)
    ini, fin = j, j
    while ini > 0 and t[bloques[ini][0]] - t[bloques[ini - 1][1] - 1] <= MAX_PAUSA_ACTIVO_S:
        ini -= 1
    while fin < len(bloques) - 1 and t[bloques[fin + 1][0]] - t[bloques[fin][1] - 1] <= MAX_PAUSA_ACTIVO_S:
        fin += 1
    return bloques[ini][0], bloques[fin][1] - 1


def segmentar_gesto(v, dur):
    """Fases del gesto sobre la velocidad normalizada (N muestras del tramo activo, dur en segundos).
      1. Histeresis: el movimiento empieza al superar UMBRAL_MOV_ENTRA y termina al bajar de UMBRAL_MOV_SALE. Una
         velocidad que oscila alrededor de un unico umbral ya no crea fases extra.
      2. Movimientos mas cortos que MIN_MOV_S -> sostenimiento (ruido).
      3. Pausas ENTRE dos movimientos mas cortas que MIN_SOST_S -> movimiento (una frenada breve no es una fase).
      4. Las pausas del borde (reposo inicial / final) se conservan si duran al menos 2 muestras.
    Las duraciones minimas estan en SEGUNDOS: el criterio no depende de lo que dure cada repeticion.
    Devuelve [[inicio, fin, es_movimiento], ...] como segmentar()."""
    v = np.nan_to_num(np.asarray(v, float), nan=0.0)
    n = len(v)

    def muestras(seg):
        return max(2, int(np.ceil(seg / max(dur, 1e-6) * (n - 1))))

    mov, estado = np.zeros(n, bool), False
    for k in range(n):
        if not estado and v[k] > UMBRAL_MOV_ENTRA:
            estado = True
        elif estado and v[k] < UMBRAL_MOV_SALE:
            estado = False
        mov[k] = estado
    for a, b, val in _corridas(mov):
        if val and b - a < muestras(MIN_MOV_S):
            mov[a:b] = False
    for a, b, val in _corridas(mov):
        if not val and a > 0 and b < n and b - a < muestras(MIN_SOST_S):
            mov[a:b] = True
    tr = _corridas(mov)
    if len(tr) > 1:
        for a, b, val in (tr[0], tr[-1]):
            if not val and b - a < 2:
                mov[a:b] = True
    return [[a, b, val] for a, b, val in _corridas(mov)]


def segmentar(v):
    """Divide una senal normalizada (N muestras) en tramos de movimiento / sostenimiento; los tramos mas cortos que
    MIN_FASE se unen al vecino. Devuelve [[inicio, fin, es_movimiento], ...]."""
    mov = v > UMBRAL_MOV
    runs, ini = [], 0
    for k in range(1, N + 1):
        if k == N or mov[k] != mov[ini]:
            runs.append([ini, k, bool(mov[ini])])
            ini = k
    while True:
        cortos = [j for j, r in enumerate(runs) if r[1] - r[0] < MIN_FASE]
        if not cortos or len(runs) == 1:
            break
        j = cortos[0]
        if j > 0:
            runs[j - 1][1] = runs[j][1]
        else:
            runs[j + 1][0] = runs[j][0]
        del runs[j]
        fus = [runs[0]]
        for r in runs[1:]:
            if r[2] == fus[-1][2]:
                fus[-1][1] = r[1]
            else:
                fus.append(r)
        runs = fus
    return runs


def _delta(perfil, k, a, b):
    s = perfil.get(k)
    if s is None or not (np.isfinite(s[a]) and np.isfinite(s[b - 1])):
        return np.nan
    return s[b - 1] - s[a]


def codigos_fase(j, total, a, b, mov, perfil):
    """Lista de codigos que describen la fase (el primero es el principal). En las fases de movimiento se revisan,
    en orden: forma de la mano dominante, distancia al punto objetivo (si la sena lo tiene), distancia entre manos,
    altura de la mano, codo y forma de la otra mano."""
    if mov:
        cods = []
        dflex, dd = _delta(perfil, "flex", a, b), _delta(perfil, "d", a, b)
        if np.isfinite(dflex) and abs(dflex) > 30:           # angulo interno: baja = los dedos se cierran
            cods.append("Cierre de mano" if dflex < 0 else "Apertura de mano")
        if np.isfinite(dd) and dd < -0.15:
            cods.append("Transporte al objetivo")
        elif np.isfinite(dd) and dd > 0.15:
            cods.append("Retorno / alejamiento")
        dm = _delta(perfil, "dmanos", a, b)
        if np.isfinite(dm) and dm < -UMBRAL_DIST_MANOS_SW:
            cods.append("Manos se juntan")
        elif np.isfinite(dm) and dm > UMBRAL_DIST_MANOS_SW:
            cods.append("Manos se separan")
        da = _delta(perfil, "alto", a, b)
        if np.isfinite(da) and da > UMBRAL_ALTURA_SW:
            cods.append("Mano sube")
        elif np.isfinite(da) and da < -UMBRAL_ALTURA_SW:
            cods.append("Mano baja")
        dc = -_delta(perfil, "codo", a, b)                   # flexion = 180 - angulo interno
        if np.isfinite(dc) and dc > UMBRAL_CODO_DEG:
            cods.append("Codo se flexiona")
        elif np.isfinite(dc) and dc < -UMBRAL_CODO_DEG:
            cods.append("Codo se extiende")
        dfo = _delta(perfil, "flex_otro", a, b)
        if np.isfinite(dfo) and abs(dfo) > 30:
            cods.append("Otra mano se cierra" if dfo < 0 else "Otra mano se abre")
        return cods or ["Movimiento (ajuste / repeticion)"]
    if j == 0:
        return ["Preparacion (reposo inicial)"]
    if j == total - 1:
        return ["Sostenimiento final / reposo"]
    dmed = np.nanmean(perfil["d"][a:b]) if np.isfinite(perfil["d"][a:b]).any() else np.nan
    if np.isfinite(dmed) and dmed < 0.5:
        return ["Contacto con el objetivo"]
    dmm = np.nanmean(perfil["dmanos"][a:b]) if "dmanos" in perfil and np.isfinite(perfil["dmanos"][a:b]).any() \
        else np.nan
    if np.isfinite(dmm) and dmm < 0.6:
        return ["Manos juntas"]
    return ["Sostenimiento"]


def datos_fase(perfil, a, b, lugar="the sign position", lang="en"):
    nombres = {"en": ("Elbow flexion", "Shoulder angle", "Finger flexion", f"Distance to {lugar}", "shoulder widths"),
               "es": ("Flexión del codo", "Ángulo del hombro", "Flexión de dedos", f"Distancia a {lugar}",
                      "anchos de hombro")}[lang]
    partes = []
    for k, nombre, convertir in (("codo", nombres[0], True), ("hombro", nombres[1], False),
                                 ("flex", nombres[2], True), ("d", nombres[3], False)):
        s = perfil[k][a:b]
        s = s[np.isfinite(s)]
        if len(s):
            ini, fin = (180 - s[0], 180 - s[-1]) if convertir else (s[0], s[-1])
            if k == "d":
                partes.append(f"{nombre}: {ini:.2f} → {fin:.2f} {nombres[4]}")
            else:
                partes.append(f"{nombre}: {ini:.0f}° → {fin:.0f}°")
    return "; ".join(partes)


def describir_fases(res, lang="en"):
    """[(nombre, que pasa, datos, t0, t1), ...] por fase, en ingles ('en', figuras del articulo) o espanol ('es')."""
    runs, perfil, dur = res["runs"], res["perfil"], res["dur"]
    lugar, verbo = LUGAR_SENA.get(str(res.get("sena", "")).upper(), LUGAR_DEFECTO)[lang]
    out = []
    for j, (a, b, mov) in enumerate(runs):
        cods = codigos_fase(j, len(runs), a, b, mov, perfil)
        txt = [FASES_TXT[c][lang] for c in cods]
        nombre = txt[0][0].format(l=lugar, v=verbo)
        que = " ".join(t[1].format(l=lugar, v=verbo) for t in txt)
        out.append((nombre, que, datos_fase(perfil, a, b, lugar, lang),
                    a / (N - 1) * dur, min(b, N - 1) / (N - 1) * dur))
    return out


# ======================================================================================================================
# 8. FOTOGRAMAS: cara difuminada, esqueleto, fotograma mas nitido y realce
# ======================================================================================================================
VENTANA_NITIDEZ_S = 0.20     # se busca el fotograma mas nitido a +-0.2 s del centro de cada fase
MAX_FRAMES_NITIDEZ = 13      # maximo de fotogramas evaluados por fase
LADO_CROP_PX = 720           # lado (px) de los recortes de mano ya ampliados
LADO_CUERPO_PX = 1280        # lado mayor (px) de los fotogramas de cuerpo completo


def desenfocar_cara(frame, fila):
    """Pixela y difumina la cara (puntos 0-10 de la pose; si no estan, detector Haar de OpenCV)."""
    h, w = frame.shape[:2]
    pts = []
    for i in range(11):
        x, y = fila.get(f"pose_{i}_x"), fila.get(f"pose_{i}_y")
        if x is not None and y is not None and not (np.isnan(x) or np.isnan(y)):
            pts.append((x * w, y * h))
    mask = np.zeros((h, w), np.uint8)
    if len(pts) >= 3:
        pts = np.array(pts)
        cx, cy = pts.mean(axis=0)
        ancho = max(pts[:, 0].max() - pts[:, 0].min(), 0.05 * w)
        cv2.ellipse(mask, (int(cx), int(cy - 0.1 * ancho)),
                    (int(0.9 * ancho), int(1.3 * ancho)), 0, 0, 360, 255, -1)
    else:
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        caras = det.detectMultiScale(gris, 1.1, 5)
        if len(caras) == 0:
            print("AVISO: no se encontro la cara en este fotograma; revisalo a mano antes de publicar.")
            return frame
        for (x, y, cw, ch) in caras:
            cv2.ellipse(mask, (x + cw // 2, y + ch // 2), (int(cw * 0.8), int(ch * 0.9)), 0, 0, 360, 255, -1)
        ancho = max(c[2] for c in caras)
    bloque = max(8, int(ancho / 5))
    pix = cv2.resize(cv2.resize(frame, (max(1, w // bloque), max(1, h // bloque))),
                     (w, h), interpolation=cv2.INTER_NEAREST)
    pix = cv2.GaussianBlur(pix, (0, 0), max(3, ancho * 0.12))
    m = cv2.GaussianBlur(mask, (0, 0), max(2, ancho * 0.05))[..., None] / 255.0
    return (frame * (1 - m) + pix * m).astype(np.uint8)


def dibujar_esqueleto(frame, fila):
    h, w = frame.shape[:2]

    def p(pref, i):
        x, y = fila.get(f"{pref}_{i}_x"), fila.get(f"{pref}_{i}_y")
        if x is None or y is None or np.isnan(x) or np.isnan(y):
            return None
        return int(x * w), int(y * h)

    for i, j in CONEXIONES:
        a, b = p("pose", i), p("pose", j)
        if a and b:
            cv2.line(frame, a, b, (0, 255, 0), 3)
    nariz = p("pose", 0)
    if nariz:
        cv2.circle(frame, nariz, 8, (0, 255, 0), -1)
    for pref, col in (("manoI", (255, 0, 0)), ("manoD", (0, 0, 255))):
        for i in range(21):
            q = p(pref, i)
            if q:
                cv2.circle(frame, q, 4, col, -1)
    return frame


def realzar(img, lado_px):
    """Amplia (Lanczos) o reduce (area), quita ruido de compresion y realza contraste y detalle.
    Solo para las FIGURAS: los landmarks se calculan siempre sobre el frame original. Devuelve (imagen, escala)."""
    h, w = img.shape[:2]
    esc = lado_px / max(h, w)
    interp = cv2.INTER_LANCZOS4 if esc > 1 else cv2.INTER_AREA
    img = cv2.resize(img, None, fx=esc, fy=esc, interpolation=interp)
    img = cv2.bilateralFilter(img, 5, 30, 5)                       # ruido sin borrar bordes
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_, a_, b_ = cv2.split(lab)
    l_ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l_)   # contraste local
    img = cv2.cvtColor(cv2.merge((l_, a_, b_)), cv2.COLOR_LAB2BGR)
    suave = cv2.GaussianBlur(img, (0, 0), 1.6)
    return cv2.addWeighted(img, 1.7, suave, -0.7, 0), esc            # mascara de enfoque


def nitidez(img_bgr):
    """Varianza del Laplaciano (mas alta = menos borroso por movimiento)."""
    g = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (200, 200), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _centro_ventana(res, a, b):
    t, i0, i1, dur = res["t"], res["i0"], res["i1"], res["dur"]
    tc = t[i0] + ((a + b - 1) / 2) / (N - 1) * dur
    ic = int(np.argmin(np.abs(t - tc)))
    ia = max(i0, int(np.argmin(np.abs(t - (tc - VENTANA_NITIDEZ_S)))))
    ib = min(i1, int(np.argmin(np.abs(t - (tc + VENTANA_NITIDEZ_S)))))
    return ic, ia, max(ia, ib)


def _mejor_frame(cap, ia, ib, evaluar):
    """Lee los frames ia..ib (de forma secuencial: un solo salto) y devuelve el de mayor puntaje segun evaluar(k, fr),
    que retorna (puntaje, extra) o None. Resultado: (puntaje, k, frame, extra) o None."""
    paso = max(1, int(np.ceil((ib - ia + 1) / MAX_FRAMES_NITIDEZ)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ia)
    mejor = None
    for k in range(ia, ib + 1):
        ok, fr = cap.read()
        if not ok:
            break
        if (k - ia) % paso:
            continue
        r = evaluar(k, fr)
        if r is not None and (mejor is None or r[0] > mejor[0]):
            mejor = (r[0], k, fr.copy(), r[1])
    return mejor


def fotogramas_cuerpo_hd(video, lm, res, blur=True):
    """Fotograma mas nitido de cada fase, con la cara difuminada, realzado y con el esqueleto.
    Devuelve (lista de imagenes RGB o None, lista de tiempos en s)."""
    cap = cv2.VideoCapture(video)
    imgs, tiempos = [], []
    try:
        for a, b, _ in res["runs"]:
            ic, ia, ib = _centro_ventana(res, a, b)
            mejor = _mejor_frame(cap, ia, ib, lambda k, fr: (nitidez(fr), None))
            if mejor is None:
                imgs.append(None)
                tiempos.append(float(res["t"][ic]))
                continue
            _, k, fr, _ = mejor
            fila = lm.iloc[k]
            if blur:
                fr = desenfocar_cara(fr, fila)
            fr, _ = realzar(fr, LADO_CUERPO_PX)
            fr = dibujar_esqueleto(fr, fila)
            imgs.append(_rgb(fr))
            tiempos.append(float(res["t"][k]))
    finally:
        cap.release()
    return imgs, tiempos


# ======================================================================================================================
# 9. FIGURA DE FASES DEL GESTO
# ======================================================================================================================
def graficar(res, fases, imgs, sena, persona, rep, ruta_png, calibrado=False):
    runs, perfil, dur = res["runs"], res["perfil"], res["dur"]
    n = len(runs)
    x = np.linspace(0, dur, N)
    lado = {"izq": "left", "der": "right"}
    fx = lambda i: i / (N - 1) * dur    # noqa: E731

    fig = plt.figure(figsize=(14, 11 + 0.5 * n))
    outer = fig.add_gridspec(4, 1, height_ratios=[2.2, 4, 2.6, 1.0 + 0.5 * n], hspace=0.45, top=0.95)
    sub = outer[0].subgridspec(1, n, wspace=0.06)
    ax = fig.add_subplot(outer[1])
    ax2 = fig.add_subplot(outer[2], sharex=ax)
    axp = fig.add_subplot(outer[3])
    axp.axis("off")

    ax.plot(x, perfil["v"], color="#2ca02c", lw=2.2, label=f"{lado[res['lado']].capitalize()} wrist (dominant)")
    ax.plot(x, res["v_otro"], color="#b8860b", lw=2.2, label=f"{lado[res['otro']].capitalize()} wrist")
    ymax = float(np.nanmax([np.nanmax(perfil["v"]), np.nanmax(res["v_otro"])])) * 1.45
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, dur)
    ax.set_ylabel("Wrist speed\n(relative to peak)")
    ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), visible=False)

    bordes = sorted({r[0] for r in runs} | {N - 1})
    for a, b, mov in runs:
        for eje in (ax, ax2):
            eje.axvspan(fx(a), fx(min(b, N - 1)), color="#ffb347" if mov else "#9ecae1", alpha=0.16)
    for bi in bordes:
        for eje in (ax, ax2):
            eje.axvline(fx(bi), color="red", ls=":", lw=1)
    ax.plot([fx(b) for b in bordes], [0] * len(bordes), "o", color="orange", mec="red", ms=9, clip_on=False, zorder=6)

    for j, (a, b, mov) in enumerate(runs):
        xc = (fx(a) + fx(min(b, N - 1))) / 2
        etiqueta = f"F{j + 1}\n" + textwrap.fill(fases[j][0], 16)
        ax.text(xc, ymax * (0.96 if j % 2 == 0 else 0.70), etiqueta, ha="center", va="top",
                fontsize=8, bbox=dict(boxstyle="square", fc="white", ec="black", lw=1.2), zorder=7)
    ax.annotate("Gesture start", xy=(0, 0), xytext=(dur * 0.01, ymax * 0.3), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("Gesture end", xy=(dur, 0), xytext=(dur * 0.99, ymax * 0.3), fontsize=9,
                ha="right", arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    for j, (a, b, mov) in enumerate(runs):
        axt = fig.add_subplot(sub[0, j])
        if imgs[j] is not None:
            axt.imshow(imgs[j])
        axt.set_xticks([])
        axt.set_yticks([])
        axt.set_title(f"F{j + 1}", fontsize=10, fontweight="bold")
        xc = (fx(a) + fx(min(b, N - 1))) / 2
        fig.add_artist(ConnectionPatch(xyA=(0.5, 0), coordsA=axt.transAxes, axesA=axt,
                                       xyB=(xc, ymax), coordsB=ax.transData, axesB=ax,
                                       color="0.45", ls=":", lw=1))

    ax2.plot(x, 180 - perfil["codo"], label="Elbow flexion", lw=1.8)
    ax2.plot(x, perfil["hombro"], label="Shoulder angle", lw=1.8)
    ax2.plot(x, 180 - perfil["flex"], label="Finger flexion", lw=1.8)
    ax2.set_ylabel("Angle (degrees)")
    ax2.set_xlabel("Time within the active segment (s)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, ncol=3, frameon=False)

    if PIE_EN_FIGURA:
        axp.text(0, 1, _envolver(pie_gesto(res, fases, sena, persona, rep, calibrado, "en"), 190), va="top",
                 ha="left", fontsize=9, family="DejaVu Sans")

    fig.suptitle(TITULOS["gesto"]["en"].format(s=sena), fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(ruta_png, dpi=DPI_FIG, bbox_inches="tight")
    plt.close(fig)


# ======================================================================================================================
# 10. DEDOS Y CINEMATICA (angulo -> velocidad -> aceleracion; fases de los dedos; figuras de cuerpo y dedos)
# ======================================================================================================================
PLANO = "frontal"            # por ahora solo vista frontal (columna "plano" lista para unir una vista sagital)
MIN_DATOS_MANO = 0.30        # fraccion minima de frames del tramo con todos los dedos detectados
MOV_MUNECA_REL = 0.30        # una mano "se mueve" si su muneca recorre >= 30 % de la dominante...
RANGO_DEDOS_MIN = 40.0       # ... o si la flexion media indice-menique cambia >= 40 grados...
RANGO_DEDO_MIN = 60.0        # ... o si algun dedo cambia >= 60 grados
FORZAR_MANOS = None          # None = automatico; o ("izq",), ("der",), ("izq", "der")
ACT_MIN_ROM = 0.15           # actividad minima de dedos (fraccion del rango por segundo) para hablar de movimiento
FRAC_EXT, FRAC_FLEX = 0.30, 0.60   # dedo extendido < 30 % de su rango; flexionado > 60 %
DELTA_MOV = 0.15             # un dedo "cierra/abre" en una fase si cambia > 15 % de su rango

NOMBRE_EN = {"pulgar": "Thumb", "indice": "Index", "medio": "Middle", "anular": "Ring", "menique": "Little"}
COLOR_DEDO = {"pulgar": "#d62728", "indice": "#2ca02c", "medio": "#1f77b4", "anular": "#ff7f0e", "menique": "#9467bd"}
LADO_TXT = {"izq": "left", "der": "right"}


def _pref(lado):
    return "manoD" if lado == "der" else "manoI"


def _bgr(hexcolor):
    return int(hexcolor[5:7], 16), int(hexcolor[3:5], 16), int(hexcolor[1:3], 16)


# ---------------- filtrado de senales (punto 6) ----------------
# Problema que resuelve: los landmarks "tiemblan" de un frame a otro (ruido de alta frecuencia) y al derivar ese ruido
# se multiplica (velocidad) y se vuelve a multiplicar (aceleracion). Un pasa-bajos quita lo que cambia mas rapido que
# la frecuencia de corte. Butterworth = respuesta plana en la banda que pasa (no deforma la amplitud del movimiento);
# aplicado ida y vuelta (filtfilt) no retrasa la senal. Lo que puede modificar: atenua picos muy breves (< 1/fc s) y
# suaviza los cambios bruscos de direccion. Los datos crudos se conservan en landmarks.csv y angulos.csv.
def _fs(t):
    """Frecuencia de muestreo efectiva (Hz) a partir de los tiempos reales (mediana del intervalo entre frames)."""
    dt = np.diff(np.asarray(t, float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return 1.0 / float(np.median(dt)) if len(dt) else 30.0


def _butter2(fc, fs):
    """Coeficientes (b, a) de un Butterworth pasa-bajos de 2o orden (transformacion bilineal con pre-distorsion)."""
    fc = min(fc, 0.45 * fs)
    k = math.tan(math.pi * fc / fs)
    q = math.sqrt(2.0)
    n = 1.0 / (1.0 + q * k + k * k)
    b = np.array([k * k * n, 2 * k * k * n, k * k * n])
    a = np.array([1.0, 2 * (k * k - 1) * n, (1 - q * k + k * k) * n])
    return b, a


def _lfilter2(b, a, x):
    """Filtro IIR de 2o orden (forma directa II transpuesta) con estado inicial en regimen para x[0] (sin transitorio
    de arranque). x: (T,) o (T, C)."""
    y = np.empty_like(x)
    z1 = (b[1] - a[1] + b[2] - a[2]) * x[0]
    z2 = (b[2] - a[2]) * x[0]
    for i in range(len(x)):
        xi = x[i]
        yi = b[0] * xi + z1
        z1 = b[1] * xi - a[1] * yi + z2
        z2 = b[2] * xi - a[2] * yi
        y[i] = yi
    return y


def _filtfilt(x, fc, fs):
    """Fase cero: ida y vuelta, con extension impar en los bordes para no deformar el inicio y el final."""
    b, a = _butter2(fc, fs)
    pad = min(len(x) - 1, max(6, int(round(3 * fs / fc))))
    ini = 2 * x[0] - x[pad:0:-1]
    fin = 2 * x[-1] - x[-2:-pad - 2:-1]
    ext = np.concatenate([ini, x, fin])
    y = _lfilter2(b, a, ext)
    y = _lfilter2(b, a, y[::-1])[::-1]
    return y[pad:pad + len(x)]


def rellenar_huecos(y, t, max_s=None):
    """Interpola linealmente los huecos (NaN) mas cortos que max_s segundos. Devuelve (serie, mascara_interpolada)."""
    max_s = GAP_MAX_S if max_s is None else max_s
    y = np.asarray(y, float).copy()
    t = np.asarray(t, float)
    nan = ~np.isfinite(y)
    interp = np.zeros(len(y), bool)
    if nan.all() or not nan.any():
        return y, interp
    ok = np.where(~nan)[0]
    i = 0
    while i < len(y):
        if nan[i]:
            j = i
            while j < len(y) and nan[j]:
                j += 1
            if i > 0 and j < len(y) and t[j] - t[i - 1] <= max_s + 1e-9:     # hueco interior y corto
                y[i:j] = np.interp(t[i:j], [t[i - 1], t[j]], [y[i - 1], y[j]])
                interp[i:j] = True
            i = j
        else:
            i += 1
    del ok
    return y, interp


def filtrar(y, t, fc):
    """Serie filtrada (Butterworth fase cero). Los huecos cortos se interpolan antes; los largos quedan NaN y cada tramo
    continuo se filtra por separado. Tramos muy cortos (< 6 muestras) se dejan sin filtrar."""
    y, _ = rellenar_huecos(y, t)
    out = np.full(len(y), np.nan)
    fs = _fs(t)
    ok = np.isfinite(y)
    i = 0
    while i < len(y):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < len(y) and ok[j]:
            j += 1
        out[i:j] = _filtfilt(y[i:j], fc, fs) if j - i >= 6 else y[i:j]
        i = j
    return out


def derivadas(y, t, fc=None):
    """(y filtrada, velocidad, aceleracion). Derivada numerica centrada sobre los tiempos reales t (np.gradient acepta
    pasos desiguales). La aceleracion sigue siendo la variable mas sensible al ruido: interpretala con cautela."""
    fc = FC_DEDOS_HZ if fc is None else fc
    t = np.asarray(t, float)
    ys = filtrar(y, t, fc)
    v = np.gradient(ys, t) if len(ys) > 1 else np.zeros(len(ys))
    a = np.gradient(v, t) if len(ys) > 1 else np.zeros(len(ys))
    return ys, v, a


def _col(an, nombre):
    return serie(an, nombre) if nombre in an.columns else np.full(len(an), np.nan)


def flexion_dedo(an, pref, dedo):
    """Flexion total del dedo = suma de (180 - angulo) de sus articulaciones. 0 = recto. NaN si falta alguna."""
    cols = [f"ang_{pref}_{dedo}_{a}" for a in ARTICULACIONES[dedo]]
    S = np.column_stack([_col(an, c) for c in cols])
    art = 180.0 - S
    total = art.sum(axis=1)
    total[np.isnan(S).any(axis=1)] = np.nan
    return total, art


def cinematica_mano(an, t, lado, qc=None):
    """Flexion, velocidad y aceleracion de cada dedo. Con qc, los frames en que ese dedo no es valido quedan NaN
    (los huecos cortos se interpolan al filtrar)."""
    pref, kin, arts = _pref(lado), {}, {}
    for d in NOMBRES_DEDOS:
        cruda, art = flexion_dedo(an, pref, d)
        col = f"qc_dedo_{lado}_{d}_valido"
        if qc is not None and col in qc.columns:
            cruda = np.where(qc[col].to_numpy(bool), cruda, np.nan)
        f, v, a = derivadas(cruda, t, FC_DEDOS_HZ)
        kin[d] = {"flex": f, "vel": v, "acc": a}
        arts[d] = art
    return kin, arts


def cinematica_brazo(an, t, lado):
    out = {}
    for nom in ("codo", "hombro"):
        y = _col(an, f"ang_{nom}_{lado}")
        if nom == "codo":
            y = 180.0 - y          # flexion del codo: 0 = extendido (convencion fisiologica)
        ys, v, a = derivadas(y, t, FC_CUERPO_HZ)
        out[nom] = {"ang": ys, "vel": v, "acc": a}
    return out


def cinematica_muneca(lm, t, lado, ancho):
    """Rapidez y aceleracion lineales de la muneca (2D, imagen), en anchos de hombro por s y por s^2."""
    p = xy(lm, idx_muneca(lado)) / ancho
    ps = np.column_stack([filtrar(p[:, k], t, FC_POSICION_HZ) for k in range(2)])
    v = np.gradient(ps, t, axis=0)
    a = np.gradient(v, t, axis=0)
    return np.linalg.norm(v, axis=1), np.linalg.norm(a, axis=1)


def ancho_hombros(lm):
    return float(np.nanmedian(np.linalg.norm(xy(lm, 11) - xy(lm, 12), axis=1)))


def _rango(y):
    f = np.asarray(y, float)
    f = f[np.isfinite(f)]
    return float(f.max() - f.min()) if len(f) >= 2 else 0.0


def manos_activas(lm, an, t, res, ancho):
    """Decide que mano(s) participan en la sena (recorrido de la muneca o cambio de flexion de los dedos)."""
    i0, i1 = res["i0"], res["i1"]
    info = {}
    for lado in ("izq", "der"):
        p = xy(lm, idx_muneca(lado))[i0:i1 + 1] / ancho
        recorrido = float(np.nansum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        kin, _ = cinematica_mano(an, t, lado)
        M = np.column_stack([kin[d]["flex"][i0:i1 + 1] for d in NOMBRES_DEDOS])
        datos = float(np.mean(~np.isnan(M).any(axis=1)))
        media4 = np.nanmean(M[:, 1:], axis=1) if datos > 0 else np.full(len(M), np.nan)
        info[lado] = {"recorrido": recorrido, "datos": datos, "rango_medio": _rango(media4),
                      "rango_dedo": max(_rango(M[:, k]) for k in range(5))}
    rmax = max(v["recorrido"] for v in info.values())
    for lado, v in info.items():
        v["rel"] = v["recorrido"] / rmax if rmax > 0 else 0.0
        v["activa"] = bool(v["datos"] >= MIN_DATOS_MANO and (
            v["rel"] >= MOV_MUNECA_REL or v["rango_medio"] >= RANGO_DEDOS_MIN or v["rango_dedo"] >= RANGO_DEDO_MIN))
        if FORZAR_MANOS is not None:
            v["activa"] = lado in FORZAR_MANOS and v["datos"] > 0
        print(f"  mano {LADO_TXT[lado]:9s}: muneca {v['rel'] * 100:3.0f}% de la dominante | "
              f"dedos detectados {v['datos'] * 100:3.0f}% | rango flexion media {v['rango_medio']:3.0f} grados, "
              f"mayor rango de un dedo {v['rango_dedo']:3.0f} grados -> "
              f"{'SE MUEVE' if v['activa'] else 'sin movimiento relevante'}")
    return info


def _frac(d, grados_):
    return float(np.clip(grados_ / ROM_TOTAL[d], 0, 1)) if np.isfinite(grados_) else np.nan


NOMBRE_DEDO = {"en": NOMBRE_EN,
               "es": {"pulgar": "Pulgar", "indice": "Índice", "medio": "Medio", "anular": "Anular",
                      "menique": "Meñique"}}
TXT_DEDOS = {
    "en": {"sin": ("No finger data", "The fingers were not detected in this phase."),
           "cierran": "Fingers close", "abren": "Fingers open", "mixto": "Fingers move differently",
           "leve": ("Small finger adjustment", "The finger posture barely changes."),
           "flex": "Flexing", "ext": "Extending", "Extended": "Extended", "flexed": "flexed",
           "semi": "half-flexed", "ninguno": "none",
           "pos": {"inicio": "Start", "final": "End", "sost": "Hold"},
           "forma": {"abierta": "open hand", "puno": "closed fist", "puno_pulgar": "closed fist with thumb extended",
                     "otra": "hand shape"}},
    "es": {"sin": ("Sin datos de dedos", "No se detectaron los dedos en esta fase."),
           "cierran": "Los dedos se cierran", "abren": "Los dedos se abren", "mixto": "Movimiento mixto de dedos",
           "leve": ("Ajuste leve de dedos", "La postura de los dedos casi no cambia."),
           "flex": "Se flexionan", "ext": "Se extienden", "Extended": "Extendidos", "flexed": "flexionados",
           "semi": "semiflexionados", "ninguno": "ninguno",
           "pos": {"inicio": "Inicio", "final": "Final", "sost": "Sostenimiento"},
           "forma": {"abierta": "mano abierta", "puno": "puño cerrado", "puno_pulgar": "puño con el pulgar extendido",
                     "otra": "configuración de mano"}},
}


def _nombres(lista, lang="en"):
    if not lista:
        return TXT_DEDOS[lang]["ninguno"]
    return ", ".join(NOMBRE_DEDO[lang][d].lower() for d in lista)


def texto_fase_dedos(f, lang="en"):
    """(tipo, que pasa, datos) de una fase de dedos en el idioma pedido."""
    T = TXT_DEDOS[lang]
    c = f["codigo"]
    if c in ("sin", "leve"):
        tipo, que = T[c]
    elif c in ("cierran", "abren", "mixto"):
        tipo = T[c]
        partes = []
        if f["cierran"]:
            partes.append(f"{T['flex']}: {_nombres(f['cierran'], lang)}")
        if f["abren"]:
            partes.append(f"{T['ext']}: {_nombres(f['abren'], lang)}")
        que = "; ".join(partes) + "."
    else:                                                     # sostenimiento con forma de mano
        tipo = f"{T['pos'][f['pos']]}: {T['forma'][f['forma']]}"
        que = (f"{T['Extended']}: {_nombres(f['ext'], lang)}; {T['flexed']}: {_nombres(f['flx'], lang)}; "
               f"{T['semi']}: {_nombres(f['sem'], lang)}.")
    datos = "; ".join(f"{NOMBRE_DEDO[lang][d]} {f['ini'][d]:.0f}→{f['fin'][d]:.0f}°" for d in NOMBRES_DEDOS
                      if np.isfinite(f["ini"][d]) and np.isfinite(f["fin"][d]))
    if not datos:
        datos = "sin datos fiables" if lang == "es" else "no reliable data"
    return tipo, que, datos


def fases_dedos(kin, res):
    """Divide el tramo activo segun la actividad de los dedos y clasifica cada fase (forma de la mano o cambio).
    Cada fase guarda los datos; los textos (en / es) salen de texto_fase_dedos."""
    i0, i1, t, dur = res["i0"], res["i1"], res["t"], res["dur"]
    tt = t[i0:i1 + 1]

    def hay_dato(y):
        """Fraccion (0-1) de frames con dato alrededor de cada una de las N muestras."""
        return remuestrear(np.isfinite(y).astype(float), tt) if len(tt) > 1 else np.zeros(N)

    # los tramos sin datos fiables (frames rechazados por el control de calidad) NO se rellenan: quedan como
    # "sin datos de dedos" para no inventar movimientos ni posturas
    perfil = {}
    for d in NOMBRES_DEDOS:
        y = kin[d]["flex"][i0:i1 + 1]
        p = remuestrear(y, tt)
        p[hay_dato(y) < 0.5] = np.nan
        perfil[d] = p
    act = np.nanmean(np.column_stack([np.abs(kin[d]["vel"][i0:i1 + 1]) / ROM_TOTAL[d] for d in NOMBRES_DEDOS]),
                     axis=1)
    act_n = remuestrear(np.nan_to_num(act, nan=0.0), tt)
    act_n[hay_dato(act) < 0.5] = 0.0
    if np.isnan(act_n).all() or np.nanmax(act_n) < ACT_MIN_ROM:
        runs = [[0, N, False]]     # los dedos casi no cambian: una sola fase de sostenimiento
    else:
        v_ref = pd.Series(act_n).rolling(5, center=True, min_periods=1).mean().to_numpy()
        runs = segmentar(np.nan_to_num(v_ref / np.nanmax(v_ref)))

    # separa los tramos sin datos de dedos (ningun dedo valido) en fases propias "sin datos", para no mezclar
    # un tramo ocluido con uno medido; los trozos muy cortos (< 3 muestras) se unen al vecino
    sin_dato = ~np.any([np.isfinite(perfil[d]) for d in NOMBRES_DEDOS], axis=0)
    partes = []
    for a, b, mov in runs:
        k = a
        for m in range(a + 1, b + 1):
            if m == b or sin_dato[m] != sin_dato[k]:
                if partes and (m - k < 3 or partes[-1][1] - partes[-1][0] < 3) and partes[-1][1] == k:
                    partes[-1][1] = m                                  # une trozo corto con el anterior
                else:
                    partes.append([k, m, mov])
                k = m
    runs = partes

    def extremo(p, desde_inicio):
        """Primer (o ultimo) valor valido de la fase: mediana de las 3 primeras (o ultimas) muestras con dato."""
        v = p[np.isfinite(p)]
        if not len(v):
            return np.nan
        return float(np.median(v[:3] if desde_inicio else v[-3:]))

    fases = []
    for j, (a, b, mov) in enumerate(runs):
        ini = {d: extremo(perfil[d][a:b], True) for d in NOMBRES_DEDOS}
        fin = {d: extremo(perfil[d][a:b], False) for d in NOMBRES_DEDOS}
        cobertura = float(np.mean(~sin_dato[a:b])) if b > a else 0.0
        med = {d: float(np.nanmean(perfil[d][a:b])) if np.isfinite(perfil[d][a:b]).any() else np.nan
               for d in NOMBRES_DEDOS}
        validos = [d for d in NOMBRES_DEDOS if np.isfinite(med[d])]
        f = {"a": a, "b": b, "mov": mov, "ini": ini, "fin": fin, "cierran": [], "abren": [], "ext": [], "flx": [],
             "sem": [], "pos": "sost", "forma": "otra",
             "t0": a / (N - 1) * dur, "t1": min(b, N - 1) / (N - 1) * dur}
        if not validos or cobertura < 0.5:        # menos de la mitad de la fase con dedos fiables
            f["codigo"] = "sin"
        elif mov:
            delta = {d: _frac(d, fin[d]) - _frac(d, ini[d]) for d in validos}
            f["cierran"] = [d for d in validos if delta[d] > DELTA_MOV]
            f["abren"] = [d for d in validos if delta[d] < -DELTA_MOV]
            f["codigo"] = ("mixto" if f["cierran"] and f["abren"] else "cierran" if f["cierran"]
                           else "abren" if f["abren"] else "leve")
        else:
            fr = {d: _frac(d, med[d]) for d in validos}
            f["ext"] = [d for d in validos if fr[d] < FRAC_EXT]
            f["flx"] = [d for d in validos if fr[d] > FRAC_FLEX]
            f["sem"] = [d for d in validos if d not in f["ext"] and d not in f["flx"]]
            if len(f["ext"]) == len(NOMBRES_DEDOS):
                f["forma"] = "abierta"
            elif all(d in f["flx"] for d in NOMBRES_DEDOS[1:]):
                f["forma"] = "puno_pulgar" if "pulgar" in f["ext"] else "puno"
            f["pos"] = ("inicio" if (j == 0 and len(runs) > 1) else
                        "final" if (j == len(runs) - 1 and len(runs) > 1) else "sost")
            f["codigo"] = "forma"
        f["tipo"], f["que"], f["datos"] = texto_fase_dedos(f, "en")
        fases.append(f)
    return fases


def _idx_fase(a, b, i0, i1):
    M = i1 - i0 + 1
    ia = i0 + int(round(a / (N - 1) * (M - 1)))
    ib = i0 + int(round((b - 1) / (N - 1) * (M - 1)))
    return ia, max(ia, ib)


def _bbox_mano(fila, pref, w, h, margen=0.45):
    pts = []
    for i in range(21):
        x, y = fila.get(f"{pref}_{i}_x"), fila.get(f"{pref}_{i}_y")
        if x is not None and y is not None and not (np.isnan(x) or np.isnan(y)):
            pts.append((x * w, y * h))
    if len(pts) < 15:
        return None
    p = np.array(pts)
    (x0, y0), (x1, y1) = p.min(axis=0), p.max(axis=0)
    L = max(x1 - x0, y1 - y0) * (1 + 2 * margen)
    L = int(min(max(L, 0.12 * min(w, h)), w, h))
    xa = int(np.clip((x0 + x1) / 2 - L / 2, 0, w - L))
    ya = int(np.clip((y0 + y1) / 2 - L / 2, 0, h - L))
    return xa, ya, xa + L, ya + L


def _solapa_cara(fila, bb, shape):
    h, w = shape[:2]
    pts = [(fila.get(f"pose_{i}_x"), fila.get(f"pose_{i}_y")) for i in range(11)]
    pts = [(x * w, y * h) for x, y in pts if x is not None and y is not None and not (np.isnan(x) or np.isnan(y))]
    if len(pts) < 3:
        return False
    p = np.array(pts)
    (fx0, fy0), (fx1, fy1) = p.min(axis=0), p.max(axis=0)
    mx, my = (fx1 - fx0) * 0.75, (fy1 - fy0) * 1.0
    fx0, fx1, fy0, fy1 = fx0 - mx, fx1 + mx, fy0 - my, fy1 + my
    xa, ya, xb, yb = bb
    return not (xb < fx0 or xa > fx1 or yb < fy0 or ya > fy1)


def _dibujar_dedos(img, fila, pref, bb, esc, W, H):
    xa, ya = bb[0], bb[1]
    grosor = max(2, img.shape[1] // 260)

    def pt(i):
        x, y = fila.get(f"{pref}_{i}_x"), fila.get(f"{pref}_{i}_y")
        if x is None or y is None or np.isnan(x) or np.isnan(y):
            return None
        return int((x * W - xa) * esc), int((y * H - ya) * esc)

    for d in NOMBRES_DEDOS:
        cadena = [0] + DEDOS[d]
        col = _bgr(COLOR_DEDO[d])
        for i, j in zip(cadena[:-1], cadena[1:]):
            p, q = pt(i), pt(j)
            if p and q:
                cv2.line(img, p, q, col, grosor, cv2.LINE_AA)
        for i in cadena[1:]:
            p = pt(i)
            if p:
                cv2.circle(img, p, grosor + 2, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, p, grosor, col, -1, cv2.LINE_AA)
    p = pt(0)
    if p:
        cv2.circle(img, p, grosor + 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, grosor + 1, (60, 60, 60), -1, cv2.LINE_AA)


def fotogramas_mano(video, lm, res, lado, fases, blur=True):
    """Por fase: recorte de la mano en el fotograma mas nitido, ampliado y realzado (con y sin esqueleto)."""
    cap = cv2.VideoCapture(video)
    pref, out = _pref(lado), []
    try:
        for f in fases:
            _, ia, ib = _centro_ventana(res, f["a"], f["b"])

            def evaluar(k, fr):
                bb = _bbox_mano(lm.iloc[k], pref, fr.shape[1], fr.shape[0])
                return None if bb is None else (nitidez(fr[bb[1]:bb[3], bb[0]:bb[2]]), bb)

            mejor = _mejor_frame(cap, ia, ib, evaluar)
            if mejor is None:
                out.append(None)
                continue
            n_, k, fr, bb = mejor
            fila = lm.iloc[k]
            H, W = fr.shape[:2]
            if blur and _solapa_cara(fila, bb, fr.shape):
                fr = desenfocar_cara(fr, fila)
            limpio, esc = realzar(fr[bb[1]:bb[3], bb[0]:bb[2]], LADO_CROP_PX)
            con = limpio.copy()
            _dibujar_dedos(con, fila, pref, bb, esc, W, H)
            out.append({"con": _rgb(con), "limpio": _rgb(limpio), "idx": k, "t": float(res["t"][k]), "nitidez": n_})
    finally:
        cap.release()
    return out


def _limites(series, extra=0.0):
    v = np.concatenate([np.asarray(s, float)[np.isfinite(s)] for s in series] + [np.array([])])
    if len(v) == 0:
        return -1, 1
    lo, hi = np.percentile(v, [1, 99])
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad + extra * (hi - lo)


def _tramos_bool(m):
    """[(inicio, fin_exclusivo), ...] de los tramos True de un arreglo booleano."""
    m = np.asarray(m, bool)
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def figura_paneles(ruta, titulo, x, paneles, runs, dur, etiquetas, imgs, tit_thumbs, leyenda, pie, invalido=None):
    """Figura con miniaturas por fase y paneles de series. invalido: mascara (mismo largo que x) de frames rechazados
    por el control de calidad; se sombrean en gris en todos los paneles."""
    n, npan = len(runs), len(paneles)
    fx = lambda i: i / (N - 1) * dur    # noqa: E731
    r_pie = 0.6 + 0.19 * (pie.count("\n") + 1)
    ref = next((im for im in imgs if im is not None), None)
    aspecto = (ref.shape[1] / ref.shape[0]) if ref is not None else 1.0
    r_thumb = float(np.clip((11.5 / n) / aspecto + 0.5, 1.3, 3.0))   # alto justo para los fotogramas
    ratios = [r_thumb, 3.2] + [2.5] * (npan - 1) + [0.35, r_pie]
    fig = plt.figure(figsize=(14, sum(ratios) + 1.2))
    gs = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0.42, top=0.95)
    sub = gs[0].subgridspec(1, n, wspace=0.05)
    axes = []
    for k in range(npan):
        axes.append(fig.add_subplot(gs[1 + k], sharex=axes[0] if axes else None))
    axl = fig.add_subplot(gs[1 + npan])
    axl.axis("off")
    axp = fig.add_subplot(gs[2 + npan])
    axp.axis("off")
    tramos_inv = _tramos_bool(invalido) if invalido is not None else []

    bordes = sorted({r[0] for r in runs} | {N - 1})
    for k, (ax, p) in enumerate(zip(axes, paneles)):
        for (_, y, col, ls, lw) in p["series"]:
            ax.plot(x, y, color=col, ls=ls, lw=lw)
        lo, hi = _limites([s[1] for s in p["series"]], extra=0.45 if k == 0 else 0.0)
        ax.set_ylim(lo, hi)
        ax.set_xlim(0, dur)
        ax.set_ylabel(p["ylabel"], fontsize=9)
        ax.grid(alpha=0.3)
        for a, b, mov in runs:
            ax.axvspan(fx(a), fx(min(b, N - 1)), color="#ffb347" if mov else "#9ecae1", alpha=0.16)
        for i, j in tramos_inv:
            ax.axvspan(x[i], x[min(j, len(x) - 1)], color="0.55", alpha=0.35, lw=0, zorder=0.5)
        for bi in bordes:
            ax.axvline(fx(bi), color="red", ls=":", lw=1)
        if k < npan - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
    axes[-1].set_xlabel("Time within the active segment (s)")

    ax0 = axes[0]
    ax0.plot([fx(b) for b in bordes], [0] * len(bordes), "o", color="orange", mec="red", ms=9,
             clip_on=False, zorder=6, transform=ax0.get_xaxis_transform())
    for j, (a, b, mov) in enumerate(runs):
        xc = (fx(a) + fx(min(b, N - 1))) / 2
        ax0.text(xc, 0.97 if j % 2 == 0 else 0.76, f"F{j + 1}\n" + textwrap.fill(etiquetas[j], 17),
                 transform=ax0.get_xaxis_transform(), ha="center", va="top", fontsize=8, zorder=7,
                 bbox=dict(boxstyle="square", fc="white", ec="black", lw=1.1))

    for j in range(n):
        axt = fig.add_subplot(sub[0, j])
        if imgs[j] is not None:
            axt.imshow(imgs[j])
        else:
            axt.text(0.5, 0.5, "no data", ha="center", va="center", transform=axt.transAxes, fontsize=8)
        axt.set_xticks([])
        axt.set_yticks([])
        axt.set_title(tit_thumbs[j], fontsize=9, fontweight="bold")
        xc = (fx(runs[j][0]) + fx(min(runs[j][1], N - 1))) / 2
        fig.add_artist(ConnectionPatch(xyA=(0.5, 0), coordsA=axt.transAxes, axesA=axt,
                                       xyB=(xc, 1.0), coordsB=ax0.get_xaxis_transform(), axesB=ax0,
                                       color="0.45", ls=":", lw=1))

    if tramos_inv:
        leyenda = list(leyenda) + [Patch(facecolor="0.55", alpha=0.35, label="Rejected by QC")]
    axl.legend(handles=leyenda, loc="center", ncol=min(len(leyenda), 7), frameon=False, fontsize=9)
    if PIE_EN_FIGURA:
        axp.text(0, 1, pie, va="top", ha="left", fontsize=9, family="DejaVu Sans")
    fig.suptitle(titulo, fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(ruta, dpi=DPI_FIG, bbox_inches="tight")
    plt.close(fig)


def _nota_cal(calibrado, lang="en"):
    if lang == "es":
        return ("Ángulos calibrados por participante (mano abierta / puño y brazos extendidos); "
                if calibrado else "ÁNGULOS SIN CALIBRAR; ")
    return ("Angles calibrated per participant (open hand / fist and extended arms); "
            if calibrado else "UNCALIBRATED ANGLES; ")


NOTA_DERIV = {
    "en": ("signals low-pass filtered with a zero-phase 2nd-order Butterworth (forward-backward; cut-off "
           f"{FC_CUERPO_HZ:g} Hz body, {FC_DEDOS_HZ:g} Hz fingers); velocity and acceleration are numerical derivatives "
           "of the filtered signals; acceleration amplifies noise, so interpret it qualitatively. Grey bands = frames "
           "rejected by the quality control (missing, overlapping or occluded hand, tracking jump, implausible angle). "
           "Axes clipped to the 1st-99th percentile. Estimated with MediaPipe (not clinical measurements; approximate "
           "error ±10 to 15°). " + PLANO.capitalize() + " view."),
    "es": ("señales filtradas con un Butterworth pasa-bajos de 2.º orden de fase cero (ida y vuelta; corte "
           f"{FC_CUERPO_HZ:g} Hz cuerpo, {FC_DEDOS_HZ:g} Hz dedos); la velocidad y la aceleración son derivadas "
           "numéricas de las señales filtradas; la aceleración amplifica el ruido, interprétala de forma cualitativa. "
           "Bandas grises = frames rechazados por el control de calidad (mano ausente, solapada u ocluida, salto del "
           "seguimiento, ángulo imposible). Ejes recortados al percentil 1-99. Estimado con MediaPipe (no son "
           "mediciones clínicas; error aproximado ±10 a 15°). Vista " + PLANO + "."),
}


def _envolver(parrafos, ancho=185):
    """Parrafos -> texto envuelto para el pie de una figura (las lineas de fase con sangria)."""
    return "\n".join(textwrap.fill(p, ancho, subsequent_indent="      " if p[:1] == "F" and p[1:2].isdigit() else "")
                     for p in parrafos)


TITULOS = {
    "gesto": {"en": "Sign {s}: gesture phases", "es": "Seña {s}: fases del gesto"},
    "cuerpo": {"en": "Sign {s}: body (angles, velocity and acceleration)",
               "es": "Seña {s}: cuerpo (ángulos, velocidad y aceleración)"},
    "mano": {"en": "Sign {s}: fingers of the {l} hand", "es": "Seña {s}: dedos de la mano {l}"},
    "espacial": {"en": "Sign {s}: hand position, orientation and opening",
                 "es": "Seña {s}: posición, orientación y apertura de las manos"},
}
LADO_ES = {"izq": "izquierda", "der": "derecha"}


def pie_gesto(res, fases, sena, persona, rep, calibrado, lang="en"):
    lado = LADO_TXT[res["lado"]] if lang == "en" else LADO_ES[res["lado"]]
    if lang == "es":
        p = [f"Figura. Fases de la seña {sena} ({persona}, repetición {rep}). Brazo dominante: {lado}. Duración del "
             f"tramo activo: {res['dur']:.2f} s. Naranja = movimiento, azul = sostenimiento; los círculos naranja "
             "marcan los cambios de fase y cada fotograma es el más nítido cerca del centro de su fase. Las fases se "
             "segmentan con la velocidad de la muñeca que más se mueve en cada instante."]
        nota = ("Codo: flexión (0° = completamente extendido); hombro: ángulo tronco-brazo (0° = brazo al costado, "
                "90° = horizontal); flexión de dedos: media de las articulaciones MCP y PIP (0° = recto). Estimado "
                "con MediaPipe (no son mediciones clínicas; error aproximado ±10 a 15°).")
    else:
        p = [f"Figure. Phases of the sign {sena} ({persona}, repetition {rep}). Dominant arm: {lado}. Duration of the "
             f"active segment: {res['dur']:.2f} s. Orange = movement, blue = hold; orange circles mark the changes of "
             "phase and each frame is the sharpest one near the middle of its phase. Phases are segmented with the "
             "speed of whichever wrist moves most at each instant."]
        nota = ("Elbow: flexion (0° = fully extended); shoulder: trunk-arm angle (0° = arm at the side, 90° = "
                "horizontal); finger flexion: mean of the MCP and PIP joints (0° = straight). Estimated with MediaPipe "
                "(not clinical measurements; approximate error ±10 to 15°).")
    for j, (tipo, que, datos, t0, t1) in enumerate(fases):
        p.append(f"F{j + 1} ({t0:.2f}-{t1:.2f} s) {tipo}. {que} {datos}.")
    p.append(_nota_cal(calibrado, lang) + nota)
    return p


def pie_cuerpo(res, fases, sena, persona, rep, calibrado, lang="en"):
    lado = LADO_TXT[res["lado"]] if lang == "en" else LADO_ES[res["lado"]]
    if lang == "es":
        p = [f"Figura. Cinemática del cuerpo en la seña {sena} ({persona}, repetición {rep}). Brazo dominante: "
             f"{lado}. Duración del tramo activo: {res['dur']:.2f} s. Naranja = movimiento, azul = sostenimiento; "
             "los círculos naranja marcan los cambios de fase y cada fotograma es el más nítido cerca del centro de su "
             "fase. Codo: flexión (0° = completamente extendido); hombro: ángulo tronco-brazo (0° = brazo al costado, "
             "90° = horizontal). Paneles 4-5: rapidez y aceleración lineales de la muñeca en anchos de hombro."]
    else:
        p = [f"Figure. Body kinematics in the sign {sena} ({persona}, repetition {rep}). Dominant arm: {lado}. "
             f"Duration of the active segment: {res['dur']:.2f} s. Orange = movement, blue = hold; orange circles mark "
             "the changes of phase and each frame is the sharpest one near the middle of its phase. Elbow: flexion "
             "(0° = fully extended); shoulder: trunk-arm angle (0° = arm at the side, 90° = horizontal). Panels 4-5: "
             "linear wrist speed and acceleration in shoulder widths."]
    for j, f in enumerate(fases):
        p.append(f"F{j + 1} ({f[3]:.2f}-{f[4]:.2f} s) {f[0]}. {f[1]} {f[2]}.")
    p.append(_nota_cal(calibrado, lang) + NOTA_DERIV[lang])
    return p


def pie_mano(res, lado, fases, sena, persona, rep, calibrado, lang="en"):
    if lang == "es":
        p = [f"Figura. Fases de los dedos de la mano {LADO_ES[lado]} en la seña {sena} ({persona}, repetición {rep}), "
             f"vista {PLANO}. Duración del tramo activo: {res['dur']:.2f} s. Naranja = movimiento de dedos, azul = "
             "sostenimiento. Cada recorte es el fotograma más nítido cerca del centro de su fase, ampliado y realzado; "
             "cada dedo tiene su color. Flexión de un dedo = suma de la flexión de sus tres articulaciones."]
        etq = "Flexión en grados (inicio→fin)"
    else:
        p = [f"Figure. Finger phases of the {LADO_TXT[lado]} hand in the sign {sena} ({persona}, repetition {rep}), "
             f"{PLANO} view. Duration of the active segment: {res['dur']:.2f} s. Orange = finger movement, blue = "
             "hold. Each crop is the sharpest frame near the middle of its phase, enlarged and enhanced; each finger "
             "has its own colour. Finger flexion = sum of the flexion of its three joints."]
        etq = "Flexion in degrees (start→end)"
    for j, f in enumerate(fases):
        tipo, que, datos = texto_fase_dedos(f, lang)
        p.append(f"F{j + 1} ({f['t0']:.2f}-{f['t1']:.2f} s) {tipo}. {que} {etq}: {datos}.")
    p.append(_nota_cal(calibrado, lang) + NOTA_DERIV[lang])
    return p


def figura_cuerpo(ruta, res, imgs, an, lm, t, ancho, sena, persona, rep, calibrado, invalido=None):
    """Angulos, velocidad y aceleracion de codo/hombro y rapidez/aceleracion de muneca. Devuelve la tabla por frame
    (cuerpo_cinematica.csv)."""
    fases = describir_fases(res, "en")
    i0, i1, dur = res["i0"], res["i1"], res["dur"]
    sl = slice(i0, i1 + 1)
    x = t[sl] - t[i0]
    col = {"izq": "#1f77b4", "der": "#d62728"}
    brazos = {l: cinematica_brazo(an, t, l) for l in ("izq", "der")}
    mun = {l: cinematica_muneca(lm, t, l, ancho) for l in ("izq", "der")}

    def serie_ang(clave):
        s = []
        for l in ("izq", "der"):
            s.append((f"Elbow {l}", brazos[l]["codo"][clave][sl], col[l], "-", 1.9))
            s.append((f"Shoulder {l}", brazos[l]["hombro"][clave][sl], col[l], "--", 1.9))
        return s

    paneles = [
        {"ylabel": "Joint angle (°)", "series": serie_ang("ang")},
        {"ylabel": "Angular velocity (°/s)", "series": serie_ang("vel")},
        {"ylabel": "Angular acceleration (°/s²)", "series": serie_ang("acc")},
        {"ylabel": "Wrist speed\n(shoulder widths/s)",
         "series": [(f"Wrist {l}", mun[l][0][sl], col[l], "-", 2.0) for l in ("izq", "der")]},
        {"ylabel": "Wrist acceleration\n(shoulder widths/s²)",
         "series": [(f"Wrist {l}", mun[l][1][sl], col[l], "-", 2.0) for l in ("izq", "der")]},
    ]
    leyenda = [Line2D([0], [0], color=col[l], ls=ls, lw=2, label=f"{n_}, {LADO_TXT[l]}")
               for l in ("izq", "der") for n_, ls in (("Elbow", "-"), ("Shoulder", "--"))]
    leyenda += [Line2D([0], [0], color=col[l], lw=2.2, label=f"Wrist, {LADO_TXT[l]} (panels 4-5)")
                for l in ("izq", "der")]
    figura_paneles(ruta, TITULOS["cuerpo"]["en"].format(s=sena), x, paneles, res["runs"], dur,
                   [f[0] for f in fases], imgs, [f"F{j + 1}  {f[3]:.2f}–{f[4]:.2f} s" for j, f in enumerate(fases)],
                   leyenda, _envolver(pie_cuerpo(res, fases, sena, persona, rep, calibrado, "en")),
                   invalido=None if invalido is None else np.asarray(invalido)[sl])

    df = pd.DataFrame({"tiempo_ms": (t * 1000).round().astype(int), "en_tramo_activo": 0, "plano": PLANO})
    df.loc[i0:i1, "en_tramo_activo"] = 1
    for l in ("izq", "der"):
        for nom in ("codo", "hombro"):
            for k, suf in (("ang", "ang"), ("vel", "vel_gs"), ("acc", "acc_gs2")):
                df[f"{nom}_{l}_{suf}"] = brazos[l][nom][k]
        df[f"muneca_{l}_rapidez_hombros_s"], df[f"muneca_{l}_acc_hombros_s2"] = mun[l]
    return df


def figura_mano(ruta, res, lado, kin, fases, frames, sena, persona, rep, calibrado, invalido=None):
    i0, i1, dur, t = res["i0"], res["i1"], res["dur"], res["t"]
    sl = slice(i0, i1 + 1)
    x = t[sl] - t[i0]
    paneles = [
        {"ylabel": "Finger flexion (°)\n(sum of its joints)",
         "series": [(NOMBRE_EN[d], kin[d]["flex"][sl], COLOR_DEDO[d], "-", 2.0) for d in NOMBRES_DEDOS]},
        {"ylabel": "Angular velocity (°/s)",
         "series": [(NOMBRE_EN[d], kin[d]["vel"][sl], COLOR_DEDO[d], "-", 1.8) for d in NOMBRES_DEDOS]},
        {"ylabel": "Angular acceleration (°/s²)",
         "series": [(NOMBRE_EN[d], kin[d]["acc"][sl], COLOR_DEDO[d], "-", 1.6) for d in NOMBRES_DEDOS]},
    ]
    leyenda = [Line2D([0], [0], color=COLOR_DEDO[d], lw=2.4, label=NOMBRE_EN[d]) for d in NOMBRES_DEDOS]
    runs = [[f["a"], f["b"], f["mov"]] for f in fases]
    figura_paneles(ruta, TITULOS["mano"]["en"].format(s=sena, l=LADO_TXT[lado]), x, paneles, runs, dur,
                   [f["tipo"] for f in fases], [(fr["con"] if fr else None) for fr in frames],
                   [f"F{j + 1}  {f['t0']:.2f}–{f['t1']:.2f} s" for j, f in enumerate(fases)], leyenda,
                   _envolver(pie_mano(res, lado, fases, sena, persona, rep, calibrado, "en")),
                   invalido=None if invalido is None else np.asarray(invalido)[sl])


def tablas_mano(res, lado, kin, arts, fases, frames, info, sena, persona, rep, calibrado, t, invalido=None):
    i0, i1 = res["i0"], res["i1"]
    filas = []
    for j, f in enumerate(fases):
        ia, ib = _idx_fase(f["a"], f["b"], i0, i1)
        tipo_es, que_es, _ = texto_fase_dedos(f, "es")
        fila = {"Fase": f"F{j + 1}", "Tipo": "Movimiento" if f["mov"] else "Sostenimiento",
                "Descripcion (ES)": tipo_es, "Que pasa (ES)": que_es,
                "Description (EN)": f["tipo"], "What happens (EN)": f["que"],
                "t inicio (s)": round(f["t0"], 2), "t fin (s)": round(f["t1"], 2),
                "Duracion (s)": round(f["t1"] - f["t0"], 2),
                "Fotograma usado (t en video, s)": round(frames[j]["t"], 2) if frames[j] else np.nan,
                "% frames validos (QC)": (round(100 * float(1 - np.mean(np.asarray(invalido)[ia:ib + 1])), 0)
                                          if invalido is not None else np.nan)}
        for d in NOMBRES_DEDOS:
            fl, v, a = kin[d]["flex"][ia:ib + 1], kin[d]["vel"][ia:ib + 1], kin[d]["acc"][ia:ib + 1]
            n_ = NOMBRE_EN[d]
            fila[f"{n_} flexion inicio (°)"] = round(float(fl[0]), 1) if np.isfinite(fl[0]) else np.nan
            fila[f"{n_} flexion fin (°)"] = round(float(fl[-1]), 1) if np.isfinite(fl[-1]) else np.nan
            fila[f"{n_} flexion media (°)"] = round(float(np.nanmean(fl)), 1) if np.isfinite(fl).any() else np.nan
            fila[f"{n_} vel. pico (°/s)"] = round(float(np.nanmax(np.abs(v))), 1) if np.isfinite(v).any() else np.nan
            fila[f"{n_} acc. pico (°/s²)"] = round(float(np.nanmax(np.abs(a))), 1) if np.isfinite(a).any() else np.nan
        filas.append(fila)

    resumen = []
    tt = t[i0:i1 + 1] - t[i0]
    for d in NOMBRES_DEDOS:
        fl, v, a = kin[d]["flex"][i0:i1 + 1], kin[d]["vel"][i0:i1 + 1], kin[d]["acc"][i0:i1 + 1]
        ok = np.isfinite(fl).any()
        kv, ka = _nanargmax_abs(v), _nanargmax_abs(a)
        resumen.append({
            "Dedo": NOMBRE_EN[d], "Flexion minima (°)": float(np.nanmin(fl)) if ok else np.nan,
            "Flexion maxima (°)": float(np.nanmax(fl)) if ok else np.nan,
            "Rango usado (°)": _rango(fl), "Rango maximo del dedo (°)": ROM_TOTAL[d],
            "% del rango usado": round(100 * _rango(fl) / ROM_TOTAL[d], 1),
            "Vel. pico abs (°/s)": float(np.abs(v[kv])) if kv is not None else np.nan,
            "t del pico de vel. (s)": float(tt[kv]) if kv is not None else np.nan,
            "Acc. pico abs (°/s²)": float(np.abs(a[ka])) if ka is not None else np.nan,
            "t del pico de acc. (s)": float(tt[ka]) if ka is not None else np.nan})

    cin = pd.DataFrame({"tiempo_s": tt})
    art_df = pd.DataFrame({"tiempo_s": tt})
    for d in NOMBRES_DEDOS:
        cin[f"{NOMBRE_EN[d]}_flex"] = kin[d]["flex"][i0:i1 + 1]
        cin[f"{NOMBRE_EN[d]}_vel"] = kin[d]["vel"][i0:i1 + 1]
        cin[f"{NOMBRE_EN[d]}_acc"] = kin[d]["acc"][i0:i1 + 1]
        for k, nom_art in enumerate(ARTICULACIONES[d]):
            art_df[f"{NOMBRE_EN[d]}_{nom_art}_flex"] = arts[d][i0:i1 + 1, k]
    if invalido is not None:
        cin["rechazado_qc"] = np.asarray(invalido)[i0:i1 + 1].astype(int)
    inf = pd.DataFrame([{"Sena": sena, "Persona": persona, "Repeticion": rep, "Mano": LADO_TXT[lado],
                         "Plano": PLANO, "Angulos calibrados": "si" if calibrado else "NO",
                         "Muneca (% de la dominante)": round(100 * info["rel"], 0),
                         "Frames con dedos detectados (%)": round(100 * info["datos"], 0),
                         "Rango flexion media (°)": round(info["rango_medio"], 0),
                         "Mayor rango de un dedo (°)": round(info["rango_dedo"], 0),
                         "Filtro pasa-bajos dedos (Hz)": FC_DEDOS_HZ}])
    return {"Fases_dedos": pd.DataFrame(filas), "Resumen_dedos": pd.DataFrame(resumen),
            "Cinematica": cin, "Articulaciones": art_df, "Info": inf}


def tabla_frames_mano(t, res, kin):
    df = pd.DataFrame({"tiempo_ms": (t * 1000).round().astype(int), "en_tramo_activo": 0, "plano": PLANO})
    df.loc[res["i0"]:res["i1"], "en_tramo_activo"] = 1
    for d in NOMBRES_DEDOS:
        df[f"{d}_flex"], df[f"{d}_vel"], df[f"{d}_acc"] = kin[d]["flex"], kin[d]["vel"], kin[d]["acc"]
    return df


def analisis_dedos_y_cinematica(video, lm, an, res, imgs, sena, persona, rep, carpeta, calibrado, ruta_calib="",
                                blur=True, qc=None):
    """Figuras y tablas de cuerpo y dedos de una repeticion. Devuelve {lado: fases de dedos} de las manos activas.
    Un fallo aqui no borra lo ya guardado."""
    try:
        return _analisis_dedos_y_cinematica(video, lm, an, res, imgs, sena, persona, rep, carpeta, calibrado,
                                            ruta_calib, blur, qc)
    except Exception as e:
        traceback.print_exc()
        print(f"\nAVISO: fallo el analisis de dedos/cinematica ({e}). Lo anterior (fases.png, CSV) quedo guardado.")
        return {}


def _analisis_dedos_y_cinematica(video, lm, an, res, imgs, sena, persona, rep, carpeta, calibrado, ruta_calib, blur,
                                 qc):
    print(f"\n--- Dedos y cinematica (vista {PLANO}) ---")
    if not calibrado:
        print("AVISO: los angulos NO estan calibrados. Las flexiones y los porcentajes de rango de los dedos "
              "no son confiables hasta calibrar.")
    elif ruta_calib:
        revisar_calibracion(ruta_calib, breve=True)

    t = pd.to_numeric(lm["tiempo_ms"], errors="coerce").to_numpy(dtype=float) / 1000.0
    ancho = ancho_hombros(lm)
    fotos = os.path.join(carpeta, "fotogramas")
    os.makedirs(fotos, exist_ok=True)

    def inv(col):
        return None if qc is None or col not in qc.columns else ~qc[col].to_numpy(bool)

    # --- cuerpo
    for j, im in enumerate(imgs):
        if im is not None:
            cv2.imwrite(os.path.join(fotos, f"cuerpo_F{j + 1}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    df_c = figura_cuerpo(os.path.join(carpeta, "cuerpo.png"), res, imgs, an, lm, t, ancho, sena, persona, rep,
                         calibrado, invalido=inv(f"qc_brazo_{res['lado']}_valido"))
    df_c.to_csv(os.path.join(carpeta, "cuerpo_cinematica.csv"), index=False)
    print("Guardado: cuerpo.png y cuerpo_cinematica.csv")

    # --- dedos
    info = manos_activas(lm, an, t, res, ancho)
    activas = [l for l in ("der", "izq") if info[l]["activa"]]
    if not activas:
        print("Ninguna mano mostro movimiento relevante de dedos; no se generan figuras de dedos "
              "(ajusta MOV_MUNECA_REL / RANGO_DEDOS_MIN o usa FORZAR_MANOS).")
    salida = {}
    for lado in activas:
        # gris = ningun dedo de esa mano es valido (las curvas de cada dedo ya traen sus propios huecos)
        invalido = None if qc is None else ~np.any([qc[f"qc_dedo_{lado}_{d}_valido"].to_numpy(bool)
                                                     for d in NOMBRES_DEDOS], axis=0)
        kin, arts = cinematica_mano(an, t, lado, qc)
        fases = fases_dedos(kin, res)
        frames = fotogramas_mano(video, lm, res, lado, fases, blur)
        for j, fr in enumerate(frames):
            if fr:
                cv2.imwrite(os.path.join(fotos, f"mano_{lado}_F{j + 1}.png"), cv2.cvtColor(fr["con"], cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(fotos, f"mano_{lado}_F{j + 1}_limpio.png"),
                            cv2.cvtColor(fr["limpio"], cv2.COLOR_RGB2BGR))
        figura_mano(os.path.join(carpeta, f"dedos_{lado}.png"), res, lado, kin, fases, frames, sena, persona, rep,
                    calibrado, invalido)
        guardar_excel(os.path.join(carpeta, f"dedos_{lado}.xlsx"),
                      tablas_mano(res, lado, kin, arts, fases, frames, info[lado], sena, persona, rep, calibrado, t,
                                  invalido))
        tabla_frames_mano(t, res, kin).to_csv(os.path.join(carpeta, f"dedos_{lado}_cinematica.csv"), index=False)
        print(f"Guardado: dedos_{lado}.png, dedos_{lado}.xlsx, dedos_{lado}_cinematica.csv y "
              f"fotogramas/mano_{lado}_F*.png")
        for j, f in enumerate(fases):
            print(f"   F{j + 1} ({f['t0']:.2f}-{f['t1']:.2f} s) {f['tipo']}: {f['que']}")
        salida[lado] = fases
    return salida


# ======================================================================================================================
# 10b. VARIABLES ESTANDARIZADAS POR FRAME (calidad, angulos, manos, espacio, cinematica, sincronia)
# ======================================================================================================================
# Todo queda en variables.csv (una fila por frame, mismo eje de tiempo para cuerpo y manos) y se resume en fases.xlsx.
# CONVENCION UNICA DE ANGULOS (punto 3): grados; 0 = postura neutra (segmentos alineados / brazo al costado / cabeza
# al frente); positivo = flexion o elevacion. Sufijos: _deg = angulo filtrado, _vel_dps = velocidad angular (grados/s),
# _acc_dps2 = aceleracion angular (grados/s^2). Lados: izq / der (de la persona).
# COORDENADAS (punto 5):
#   _sw   = 2D en la imagen, en anchos de hombro (mediana de la repeticion), origen en el punto medio de los hombros;
#           x + = hacia la derecha de la IMAGEN, y + = hacia arriba.
#   _m    = 3D estimado por MediaPipe (pose world, metros), origen en el punto medio de los hombros;
#           X + = derecha de la imagen, Y + = arriba, Z + = hacia la camara.
# Los frames que el control de calidad rechaza quedan en NaN en las variables de esa mano / brazo (los huecos mas cortos
# que GAP_MAX_S se interpolan). Los datos crudos siguen en landmarks.csv y angulos.csv.
IDX_MANO_POSE = ({"izq": (16, 18, 20), "der": (15, 17, 19)} if INVERTIR_LADOS
                 else {"izq": (15, 17, 19), "der": (16, 18, 20)})       # muneca, menique, indice (modelo de pose)
PREF_LADO = {"izq": "manoI", "der": "manoD"}


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float) if col in df.columns else np.full(len(df), np.nan)


def _pts_mano_img(lm, pref):
    """(T, 21, 2) puntos de la mano en pixeles."""
    w, h = escala_px(lm)
    return np.stack([np.column_stack([_num(lm, f"{pref}_{i}_x") * w, _num(lm, f"{pref}_{i}_y") * h])
                     for i in range(21)], axis=1)


def _pts_mano_3d(lm, pref):
    """(T, 21, 3) de la mano: coordenadas mundo (metros) si existen; si no, imagen (px, z a escala del ancho)."""
    if f"mundo_{pref}_0_x" in lm.columns:
        H = np.stack([pts3d(lm, f"mundo_{pref}", i) for i in range(21)], axis=1)
        if np.isfinite(H).any():
            return H
    w, h = escala_px(lm)
    return np.stack([np.column_stack([_num(lm, f"{pref}_{i}_x") * w, _num(lm, f"{pref}_{i}_y") * h,
                                      _num(lm, f"{pref}_{i}_z") * w]) for i in range(21)], axis=1)


def _caja(P):
    """Recuadro (x0, y0, x1, y1) por frame de un conjunto de puntos (T, K, 2)."""
    return np.nanmin(P[..., 0], axis=1), np.nanmin(P[..., 1], axis=1), np.nanmax(P[..., 0], axis=1), \
        np.nanmax(P[..., 1], axis=1)


def _fraccion_dentro(a, b):
    """Fraccion del area del recuadro a que cae dentro del recuadro b (por frame)."""
    iw = np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])
    ih = np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    area = (a[2] - a[0]) * (a[3] - a[1])
    return np.where(area > 0, inter / np.where(area > 0, area, 1), 0.0)


def calidad_por_frame(lm, an=None):
    """Control de calidad (punto 2). Una fila por frame con banderas 0/1:
      qc_pose_presente            se detecto el cuerpo
      qc_brazo_<l>_vis_min        visibilidad minima de hombro, codo y muneca (si el archivo la tiene)
      qc_brazo_<l>_longitud_ok    humero y antebrazo (metros) dentro de +-TOL_LONG_BRAZO de su mediana
      qc_brazo_<l>_valido         visible y con longitudes coherentes
      qc_mano_<l>_presente        el modelo devolvio los 21 puntos de la mano
      qc_mano_<l>_salto           la mano 'viaja' mas rapido que QC_SALTO_SW_S: salto del seguimiento
      qc_mano_<l>_falanges_mal    cuantas de las 20 falanges (metros) se salen de +-QC_TOL_LONG de su mediana
      qc_mano_<l>_longitud_ok     como maximo QC_MAX_FALANGES_MAL falanges fuera de tolerancia
      qc_mano_<l>_sobre_cara      la mano esta sobre la zona de la cara (solo aviso: puede ser parte de la sena)
      qc_mano_<l>_angulo_imposible  alguna articulacion fuera de un rango anatomico generoso
      qc_manos_solapadas          los recuadros de las dos manos se cubren (dedos entrelazados u ocluidos)
      qc_mano_<l>_valida          presente, sin salto, longitudes coherentes, sin solape y sin angulos imposibles
                                  (se usa para palma, apertura, pinza y separacion de dedos)
      qc_dedo_<l>_<dedo>_valido   ese dedo: mano presente, sin salto ni solape, sus 4 falanges bien y sin angulo
                                  imposible (se usa para los angulos de ese dedo)"""
    n = len(lm)
    t = _num(lm, "tiempo_ms") / 1000.0
    qc = {"frame": lm["frame"].to_numpy(), "tiempo_ms": lm["tiempo_ms"].to_numpy()}
    qc["qc_pose_presente"] = np.isfinite(_num(lm, "pose_0_x")) & np.isfinite(_num(lm, "pose_11_x"))
    tiene_vis = "pose_0_vis" in lm.columns
    mundo = tiene_mundo(lm)
    for lado, idx in (("izq", (L_SHOULDER, L_ELBOW, L_WRIST)), ("der", (R_SHOULDER, R_ELBOW, R_WRIST))):
        ok = np.all([np.isfinite(_num(lm, f"pose_{i}_x")) for i in idx], axis=0)
        if tiene_vis:
            vis = np.nanmin(np.column_stack([_num(lm, f"pose_{i}_vis") for i in idx]), axis=1)
            qc[f"qc_brazo_{lado}_vis_min"] = np.round(vis, 3)
            ok &= np.nan_to_num(vis, nan=1.0) >= VISIBILIDAD_MINIMA
        if mundo:
            P = [pts3d(lm, "mundo_pose", i) for i in idx]
            lo = longitud_ok(P[0], P[1], TOL_LONG_BRAZO) & longitud_ok(P[1], P[2], TOL_LONG_BRAZO)
            qc[f"qc_brazo_{lado}_longitud_ok"] = lo
            ok &= lo
        qc[f"qc_brazo_{lado}_valido"] = ok

    sw = np.nanmedian(np.linalg.norm(xy(lm, 11) - xy(lm, 12), axis=1))
    sw = sw if np.isfinite(sw) and sw > 0 else 1.0
    w, h = escala_px(lm)
    C = np.stack([np.column_stack([_num(lm, f"pose_{i}_x") * w, _num(lm, f"pose_{i}_y") * h]) for i in range(11)], 1)
    fx0, fy0, fx1, fy1 = _caja(C)
    mx, my = (fx1 - fx0) * 0.75, (fy1 - fy0) * 1.0
    caja_cara = (fx0 - mx, fy0 - my, fx1 + mx, fy1 + my)
    cajas = {}
    for lado in ("izq", "der"):
        pref = PREF_LADO[lado]
        P = _pts_mano_img(lm, pref)
        pres = np.isfinite(P).all(axis=(1, 2))
        cajas[lado] = _caja(P)
        centro = np.nanmean(P[:, [0, 5, 9, 13, 17]], axis=1)
        salto = np.zeros(n, bool)
        if n > 1:
            vel = np.linalg.norm(np.diff(centro, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3) / sw
            grande = np.nan_to_num(vel, nan=0.0) > QC_SALTO_SW_S
            salto[1:] |= grande
            salto[:-1] |= grande
        dedo_long = {d: np.ones(n, bool) for d in DEDOS}       # 4 falanges del dedo dentro de tolerancia
        n_mal = np.zeros(n, int)                                # falanges fuera de tolerancia en la mano
        if f"mundo_{pref}_0_x" in lm.columns:
            H = [pts3d(lm, f"mundo_{pref}", i) for i in range(21)]
            if np.isfinite(H[0]).any():
                for d, idx in DEDOS.items():
                    cad = [0] + idx
                    for k in range(4):
                        ok_seg = longitud_ok(H[cad[k + 1]], H[cad[k]], QC_TOL_LONG) | ~pres
                        dedo_long[d] &= ok_seg
                        n_mal += ~ok_seg
        long_ok = n_mal <= QC_MAX_FALANGES_MAL
        dedo_imposible = {d: np.zeros(n, bool) for d in DEDOS}
        if an is not None:
            for d, arts in ARTICULACIONES.items():
                for art in arts:
                    fl = 180.0 - _num(an, f"ang_{pref}_{d}_{art}")
                    dedo_imposible[d] |= np.nan_to_num((fl < -20) | (fl > ROM[art] * 1.3), nan=False)
        imposible = np.any([dedo_imposible[d] for d in DEDOS], axis=0)
        qc[f"qc_mano_{lado}_falanges_mal"] = np.where(pres, n_mal, 0)
        qc[f"qc_mano_{lado}_presente"] = pres
        qc[f"qc_mano_{lado}_salto"] = salto & pres
        qc[f"qc_mano_{lado}_longitud_ok"] = long_ok
        qc[f"qc_mano_{lado}_sobre_cara"] = pres & (_fraccion_dentro(cajas[lado], caja_cara) > QC_SOLAPE_CARA_MIN)
        qc[f"qc_mano_{lado}_angulo_imposible"] = imposible & pres
        for d in DEDOS:
            qc[f"_dedo_{lado}_{d}"] = (dedo_long[d], dedo_imposible[d])
    a, b = cajas["izq"], cajas["der"]
    iw = np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])
    ih = np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])
    inter = np.clip(np.nan_to_num(iw), 0, None) * np.clip(np.nan_to_num(ih), 0, None)
    amin = np.fmin((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    solape = np.where(np.nan_to_num(amin) > 0, inter / np.where(np.nan_to_num(amin) > 0, amin, 1), 0.0)
    qc["qc_manos_solapadas"] = np.nan_to_num(solape) > QC_SOLAPE_MIN
    for lado in ("izq", "der"):
        base_ok = qc[f"qc_mano_{lado}_presente"] & ~qc[f"qc_mano_{lado}_salto"] & ~qc["qc_manos_solapadas"]
        qc[f"qc_mano_{lado}_valida"] = (base_ok & qc[f"qc_mano_{lado}_longitud_ok"]
                                        & ~qc[f"qc_mano_{lado}_angulo_imposible"])
        for d in DEDOS:
            d_long, d_imp = qc.pop(f"_dedo_{lado}_{d}")
            qc[f"qc_dedo_{lado}_{d}_valido"] = base_ok & d_long & ~d_imp
    out = pd.DataFrame(qc)
    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].astype(np.int8)
    return out


def resumen_calidad(qc, res=None):
    """% de frames con cada bandera en 1 (toda la repeticion y tramo activo)."""
    filas = []
    act = np.zeros(len(qc), bool)
    if res is not None:
        act[res["i0"]:res["i1"] + 1] = True
    for c in qc.columns:
        if not c.startswith("qc_") or c.endswith("_vis_min"):
            continue
        if c.endswith("_falanges_mal"):
            v = qc[c].to_numpy(float)
            filas.append({"Bandera": c + " (media de falanges fuera de tolerancia, de 20)",
                          "% frames (video)": round(float(np.nanmean(v)), 2),
                          "% frames (tramo activo)": round(float(np.nanmean(v[act])), 2) if act.any() else np.nan})
            continue
        v = qc[c].to_numpy(float)
        filas.append({"Bandera": c, "% frames (video)": round(100 * np.nanmean(v), 1),
                      "% frames (tramo activo)": round(100 * np.nanmean(v[act]), 1) if act.any() else np.nan})
    for c in [c for c in qc.columns if c.endswith("_vis_min")]:
        v = qc[c].to_numpy(float)
        filas.append({"Bandera": c + " (mediana)", "% frames (video)": round(float(np.nanmedian(v)), 3),
                      "% frames (tramo activo)": round(float(np.nanmedian(v[act])), 3) if act.any() else np.nan})
    return pd.DataFrame(filas)


def angulos_estandar(an, t, qc=None):
    """Angulos con la convencion unica (punto 3), filtrados y con velocidad y aceleracion angulares (punto 7)."""
    cols = {}

    def add(nombre, y, fc, mask=None):
        y = np.asarray(y, float).copy()
        if mask is not None:
            y[~mask] = np.nan
        ys, v, a = derivadas(y, t, fc)
        cols[f"{nombre}_deg"], cols[f"{nombre}_vel_dps"], cols[f"{nombre}_acc_dps2"] = ys, v, a

    def mascara(c):
        return qc[c].to_numpy(bool) if qc is not None and c in qc.columns else None

    for l in ("izq", "der"):
        mb = mascara(f"qc_brazo_{l}_valido")
        add(f"codo_{l}_flex", 180.0 - _num(an, f"ang_codo_{l}"), FC_CUERPO_HZ, mb)
        add(f"hombro_{l}_elev", _num(an, f"ang_hombro_{l}"), FC_CUERPO_HZ, mb)
    for l in ("izq", "der"):
        pref = PREF_LADO[l]
        for d in NOMBRES_DEDOS:
            mm = mascara(f"qc_dedo_{l}_{d}_valido")
            total = np.zeros(len(an))
            for art in ARTICULACIONES[d]:
                fl = 180.0 - _num(an, f"ang_{pref}_{d}_{art}")
                total = total + fl
                add(f"dedo_{l}_{d}_{art}_flex", fl, FC_DEDOS_HZ, mm)
            add(f"dedo_{l}_{d}_total_flex", total, FC_DEDOS_HZ, mm)
    for k, nombre in (("cabeza_roll", "cabeza_incl_lateral"), ("cabeza_pitch", "cabeza_flex"),
                      ("cabeza_yaw", "cabeza_rot")):
        add(nombre, _num(an, "ang_" + k), FC_CUERPO_HZ)
    return pd.DataFrame(cols)


def _ang_vec(u, v):
    """Angulo (grados) entre vectores fila a fila."""
    c = np.sum(u * v, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def rasgos_manos(lm, t, qc=None):
    """Punto 4: orientacion de la palma, eje de la mano, apertura, pinza y separacion entre dedos.
    Normal de la palma n = (indice_MCP - muneca) x (menique_MCP - muneca), con el signo de cada mano para que salga
    de la palma; ejes de camara (x derecha, y abajo, z alejandose). palma_hacia_camara = -n_z (1 = palma de frente a
    la camara, -1 = dorso, ~0 = de canto)."""
    cols = {}
    for lado in ("izq", "der"):
        pref = PREF_LADO[lado]
        H = _pts_mano_3d(lm, pref)
        P = _pts_mano_img(lm, pref)
        mask = qc[f"qc_mano_{lado}_valida"].to_numpy(bool) if qc is not None else np.isfinite(H).all(axis=(1, 2))
        signo = (-1.0 if lado == "der" else 1.0) * (-1.0 if INVERTIR_LADOS else 1.0)
        nrm = np.cross(H[:, 5] - H[:, 0], H[:, 17] - H[:, 0]) * signo
        nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
        palma = np.linalg.norm(H[:, 9] - H[:, 0], axis=1)

        def base(a, b):                                        # falange proximal proyectada en el plano de la palma
            v = H[:, b] - H[:, a]
            return v - np.sum(v * nrm, axis=1, keepdims=True) * nrm

        brutos = {
            "palma_nx": nrm[:, 0], "palma_ny": nrm[:, 1], "palma_nz": nrm[:, 2],
            "apertura": np.linalg.norm(H[:, 4] - H[:, 20], axis=1) / palma,
            "pinza": np.linalg.norm(H[:, 4] - H[:, 8], axis=1) / palma,
            "sep_indice_medio_deg": _ang_vec(base(5, 6), base(9, 10)),
            "sep_medio_anular_deg": _ang_vec(base(9, 10), base(13, 14)),
            "sep_anular_menique_deg": _ang_vec(base(13, 14), base(17, 18)),
            "sep_pulgar_indice_deg": _ang_vec(base(2, 3), base(5, 6)),
        }
        filt = {}
        for k, y in brutos.items():
            y = np.where(mask, y, np.nan)
            filt[k] = filtrar(y, t, FC_DEDOS_HZ)
        nn = np.column_stack([filt["palma_nx"], filt["palma_ny"], filt["palma_nz"]])
        nn = nn / np.linalg.norm(nn, axis=1, keepdims=True)    # el filtro no conserva la norma: se renormaliza
        filt["palma_nx"], filt["palma_ny"], filt["palma_nz"] = nn[:, 0], nn[:, 1], nn[:, 2]
        hacia = -nn[:, 2]
        d = P[:, 9] - P[:, 0]
        eje = np.degrees(np.arctan2(d[:, 0], -d[:, 1]))
        eje[~mask] = np.nan
        for k in ("palma_nx", "palma_ny", "palma_nz"):
            cols[f"mano_{lado}_{k}"] = filt[k]
        cols[f"mano_{lado}_palma_hacia_camara"] = hacia
        cols[f"mano_{lado}_palma_orientacion"] = np.where(~np.isfinite(hacia), "",
                                                          np.where(hacia > 0.5, "camara",
                                                                   np.where(hacia < -0.5, "dorso", "canto")))
        cols[f"mano_{lado}_eje_img_deg"] = eje
        for k in ("apertura", "pinza", "sep_indice_medio_deg", "sep_medio_anular_deg", "sep_anular_menique_deg",
                  "sep_pulgar_indice_deg"):
            cols[f"mano_{lado}_{k}"] = filt[k]
        cols[f"mano_{lado}_apertura_vel_1s"] = np.gradient(filt["apertura"], t) if len(t) > 1 else np.nan
    return pd.DataFrame(cols)


def _distancia(a, b):
    return np.linalg.norm(a - b, axis=1)


def coordenadas_espaciales(lm, t):
    """Punto 5 y cinematica lineal (punto 7) de cada mano. 'Mano' = promedio de muneca, indice y menique del modelo
    de pose (continuo aunque el modelo de mano falle)."""
    cols = {}
    S = (xy(lm, L_SHOULDER) + xy(lm, R_SHOULDER)) / 2
    sw = np.nanmedian(np.linalg.norm(xy(lm, L_SHOULDER) - xy(lm, R_SHOULDER), axis=1))
    sw = sw if np.isfinite(sw) and sw > 0 else 1.0
    caras = {"nariz": xy(lm, 0), "boca": (xy(lm, 9) + xy(lm, 10)) / 2, "ojos": (xy(lm, 2) + xy(lm, 5)) / 2,
             "pecho": S + np.array([0.0, PECHO_Y * sw])}
    mundo = tiene_mundo(lm)
    if mundo:
        WS = (pts3d(lm, "mundo_pose", L_SHOULDER) + pts3d(lm, "mundo_pose", R_SHOULDER)) / 2
        W_nariz = pts3d(lm, "mundo_pose", 0)
        W_boca = (pts3d(lm, "mundo_pose", 9) + pts3d(lm, "mundo_pose", 10)) / 2
    pos2d, pos3d = {}, {}
    for lado, idx in IDX_MANO_POSE.items():
        P2 = sum(xy(lm, i) for i in idx) / 3.0
        r = (P2 - S) / sw
        x, y = filtrar(r[:, 0], t, FC_POSICION_HZ), filtrar(-r[:, 1], t, FC_POSICION_HZ)
        pos2d[lado] = P2
        cols[f"mano_{lado}_x_sw"], cols[f"mano_{lado}_y_sw"] = x, y
        vx, vy = np.gradient(x, t), np.gradient(y, t)
        cols[f"mano_{lado}_vel_sw_s"] = np.hypot(vx, vy)
        cols[f"mano_{lado}_acc_sw_s2"] = np.hypot(np.gradient(vx, t), np.gradient(vy, t))
        for nombre, Q in caras.items():
            cols[f"mano_{lado}_dist_{nombre}_sw"] = filtrar(_distancia(P2, Q) / sw, t, FC_POSICION_HZ)
        if mundo:
            P3 = sum(pts3d(lm, "mundo_pose", i) for i in idx) / 3.0
            pos3d[lado] = P3
            rel = P3 - WS
            X, Y, Z = (filtrar(rel[:, 0], t, FC_POSICION_HZ), filtrar(-rel[:, 1], t, FC_POSICION_HZ),
                       filtrar(-rel[:, 2], t, FC_POSICION_HZ))
            cols[f"mano_{lado}_X_m"], cols[f"mano_{lado}_Y_m"], cols[f"mano_{lado}_Z_m"] = X, Y, Z
            V = np.column_stack([np.gradient(X, t), np.gradient(Y, t), np.gradient(Z, t)])
            A = np.gradient(V, t, axis=0)
            cols[f"mano_{lado}_vel_m_s"] = np.linalg.norm(V, axis=1)
            cols[f"mano_{lado}_acc_m_s2"] = np.linalg.norm(A, axis=1)
            cols[f"mano_{lado}_dist_nariz_m"] = filtrar(_distancia(P3, W_nariz), t, FC_POSICION_HZ)
            cols[f"mano_{lado}_dist_boca_m"] = filtrar(_distancia(P3, W_boca), t, FC_POSICION_HZ)
    cols["manos_dist_sw"] = filtrar(_distancia(pos2d["izq"], pos2d["der"]) / sw, t, FC_POSICION_HZ)
    if mundo:
        cols["manos_dist_m"] = filtrar(_distancia(pos3d["izq"], pos3d["der"]), t, FC_POSICION_HZ)
    return pd.DataFrame(cols)


def etiquetas_fase(n, res, runs):
    """Etiqueta 'F1', 'F2'... por frame del tramo activo ('' fuera de el)."""
    out = np.full(n, "", dtype=object)
    if res is None:
        return out
    for j, (a, b, _) in enumerate(runs):
        ia, ib = _idx_fase(a, b, res["i0"], res["i1"])
        out[ia:ib + 1] = f"F{j + 1}"
    return out


def construir_variables(lm, an, res, fases_lados, qc):
    """variables.csv: una fila por frame con calidad, angulos estandarizados, rasgos de mano, posicion y cinematica,
    y las fases del gesto y de los dedos en el mismo eje de tiempo (punto 9)."""
    t = _num(lm, "tiempo_ms") / 1000.0
    n = len(lm)
    base = pd.DataFrame({"frame": lm["frame"].to_numpy(), "tiempo_ms": lm["tiempo_ms"].to_numpy(),
                         "tiempo_s": np.round(t, 4)})
    base["en_tramo_activo"] = 0
    if res is not None:
        base.loc[res["i0"]:res["i1"], "en_tramo_activo"] = 1
        base["fase_gesto"] = etiquetas_fase(n, res, res["runs"])
        tipos = {f"F{j + 1}": ("movimiento" if mov else "sostenimiento") for j, (_, _, mov) in enumerate(res["runs"])}
        base["fase_gesto_tipo"] = [tipos.get(f, "") for f in base["fase_gesto"]]
    else:
        base["fase_gesto"], base["fase_gesto_tipo"] = "", ""
    for lado in ("der", "izq"):
        fases = (fases_lados or {}).get(lado)
        base[f"fase_dedos_{lado}"] = (etiquetas_fase(n, res, [[f["a"], f["b"], f["mov"]] for f in fases])
                                      if fases and res is not None else "")
    partes = [base, qc.drop(columns=["frame", "tiempo_ms"]), angulos_estandar(an, t, qc), rasgos_manos(lm, t, qc),
              coordenadas_espaciales(lm, t)]
    return pd.concat(partes, axis=1)


# ---------------- definiciones de las columnas de variables.csv ----------------
_DEF_PATRONES = [
    (r"^frame$", "-", "numero de frame del video (empieza en 1)"),
    (r"^tiempo_ms$", "ms", "marca de tiempo del frame desde el inicio del video"),
    (r"^tiempo_s$", "s", "marca de tiempo del frame desde el inicio del video"),
    (r"^en_tramo_activo$", "0/1", "1 = el frame esta dentro del tramo activo del gesto"),
    (r"^fase_gesto$", "-", "fase del gesto (F1, F2...) segun la velocidad de las munecas; vacio fuera del tramo activo"),
    (r"^fase_gesto_tipo$", "-", "movimiento o sostenimiento"),
    (r"^fase_dedos_(izq|der)$", "-", "fase de los dedos de esa mano (solo si la mano participa en la sena)"),
    (r"^qc_brazo_(izq|der)_vis_min$", "0-1", "visibilidad minima de hombro, codo y muneca de ese lado"),
    (r"^qc_mano_(izq|der)_falanges_mal$", "0-20", "numero de falanges de la mano fuera de la tolerancia de longitud"),
    (r"^qc_dedo_(izq|der)_\w+_valido$", "0/1", "1 = los angulos de ese dedo son fiables en este frame"),
    (r"^qc_", "0/1", "bandera de control de calidad (ver docstring de calidad_por_frame)"),
    (r"^codo_(izq|der)_flex_deg$", "grados", "flexion del codo (0 = extendido); filtrada"),
    (r"^hombro_(izq|der)_elev_deg$", "grados", "angulo tronco-brazo (0 = brazo al costado, 90 = horizontal); filtrado"),
    (r"^dedo_(izq|der)_\w+_total_flex_deg$", "grados", "flexion total del dedo = suma de sus 3 articulaciones (0 = recto)"),
    (r"^dedo_(izq|der)_\w+_flex_deg$", "grados", "flexion de la articulacion (0 = recta); filtrada"),
    (r"^cabeza_incl_lateral_deg$", "grados", "inclinacion lateral de la cabeza (0 = neutra)"),
    (r"^cabeza_flex_deg$", "grados", "cabeceo (0 = neutra; + = mirando hacia abajo)"),
    (r"^cabeza_rot_deg$", "grados", "giro de la cabeza (0 = neutra)"),
    (r"_vel_dps$", "grados/s", "velocidad angular (derivada del angulo filtrado)"),
    (r"_acc_dps2$", "grados/s^2", "aceleracion angular (derivada de la velocidad)"),
    (r"^mano_(izq|der)_palma_n[xyz]$", "-", "componente de la normal de la palma (vector unitario; ejes de camara: x "
                                            "derecha de la imagen, y abajo, z alejandose de la camara)"),
    (r"^mano_(izq|der)_palma_hacia_camara$", "-1..1", "1 = palma de frente a la camara, -1 = dorso, 0 = de canto"),
    (r"^mano_(izq|der)_palma_orientacion$", "-", "camara (>0.5) / dorso (<-0.5) / canto (resto)"),
    (r"^mano_(izq|der)_eje_img_deg$", "grados", "direccion muneca->nudillo medio en la imagen (0 = dedos hacia "
                                               "arriba, 90 = hacia la derecha de la imagen, +-180 = hacia abajo)"),
    (r"^mano_(izq|der)_apertura$", "-", "distancia pulgar-menique (puntas) / largo de la palma (muneca-nudillo medio)"),
    (r"^mano_(izq|der)_apertura_vel_1s$", "1/s", "velocidad de cambio de la apertura"),
    (r"^mano_(izq|der)_pinza$", "-", "distancia pulgar-indice (puntas) / largo de la palma"),
    (r"^mano_(izq|der)_sep_\w+_deg$", "grados", "separacion (abduccion) entre las falanges proximales de dos dedos "
                                                "vecinos, en el plano de la palma"),
    (r"^mano_(izq|der)_[xy]_sw$", "anchos de hombro", "posicion 2D de la mano respecto al punto medio de los hombros "
                                                       "(x + = derecha de la imagen, y + = arriba)"),
    (r"^mano_(izq|der)_[XYZ]_m$", "m", "posicion 3D estimada de la mano respecto al punto medio de los hombros "
                                       "(X + = derecha de la imagen, Y + = arriba, Z + = hacia la camara)"),
    (r"^mano_(izq|der)_vel_sw_s$", "anchos de hombro/s", "rapidez 2D de la mano"),
    (r"^mano_(izq|der)_acc_sw_s2$", "anchos de hombro/s^2", "aceleracion 2D de la mano (modulo)"),
    (r"^mano_(izq|der)_vel_m_s$", "m/s", "rapidez 3D estimada de la mano"),
    (r"^mano_(izq|der)_acc_m_s2$", "m/s^2", "aceleracion 3D estimada de la mano (modulo)"),
    (r"^mano_(izq|der)_dist_\w+_sw$", "anchos de hombro", "distancia 2D de la mano a ese punto (nariz, boca, ojos, "
                                                          "pecho = 0.35 anchos de hombro bajo los hombros)"),
    (r"^mano_(izq|der)_dist_\w+_m$", "m", "distancia 3D estimada de la mano a ese punto"),
    (r"^manos_dist_sw$", "anchos de hombro", "distancia 2D entre las dos manos"),
    (r"^manos_dist_m$", "m", "distancia 3D estimada entre las dos manos"),
]


def definiciones_variables(columnas):
    filas = []
    for c in columnas:
        for patron, unidad, texto_def in _DEF_PATRONES:
            if re.search(patron, c):
                filas.append({"columna": c, "unidad": unidad, "significado": texto_def})
                break
        else:
            filas.append({"columna": c, "unidad": "", "significado": ""})
    return pd.DataFrame(filas)


# ---------------- eventos, estadisticas por fase y sincronia (puntos 8 y 9) ----------------
def _tramo(var, ia, ib, col):
    return var[col].to_numpy(float)[ia:ib + 1] if col in var.columns else np.array([np.nan])


def _maxf(v):
    v = np.asarray(v, float)
    return float(np.nanmax(v)) if np.isfinite(v).any() else np.nan


def _meanf(v):
    v = np.asarray(v, float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def _rangof(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(v.max() - v.min()) if len(v) >= 2 else np.nan


def eventos_gesto(var, res):
    t = var["tiempo_s"].to_numpy(float)
    i0, i1, lado = res["i0"], res["i1"], res["lado"]
    v = _tramo(var, i0, i1, f"mano_{lado}_vel_sw_s")
    k = _nanargmax_abs(v)
    nmov = sum(1 for r in res["runs"] if r[2])
    filas = [
        {"Evento": "Inicio del gesto", "Event": "Gesture onset", "t video (s)": round(t[i0], 3), "Valor": np.nan,
         "Unidad": ""},
        {"Evento": "Fin del gesto", "Event": "Gesture offset", "t video (s)": round(t[i1], 3), "Valor": np.nan,
         "Unidad": ""},
        {"Evento": "Duracion del tramo activo", "Event": "Active segment duration", "t video (s)": np.nan,
         "Valor": round(res["dur"], 3), "Unidad": "s"},
        {"Evento": "Fases de movimiento", "Event": "Movement phases", "t video (s)": np.nan, "Valor": nmov,
         "Unidad": "-"},
        {"Evento": "Fases de sostenimiento", "Event": "Hold phases", "t video (s)": np.nan,
         "Valor": len(res["runs"]) - nmov, "Unidad": "-"},
        {"Evento": "Pico de rapidez de la mano dominante", "Event": "Peak speed of the dominant hand",
         "t video (s)": round(t[i0 + k], 3) if k is not None else np.nan,
         "Valor": round(float(v[k]), 3) if k is not None else np.nan, "Unidad": "anchos de hombro/s"},
    ]
    for j, (a, b, mov) in enumerate(res["runs"]):
        ia, _ = _idx_fase(a, b, i0, i1)
        filas.append({"Evento": f"Inicio de F{j + 1} ({'movimiento' if mov else 'sostenimiento'})",
                      "Event": f"Start of F{j + 1} ({'movement' if mov else 'hold'})",
                      "t video (s)": round(t[ia], 3), "Valor": np.nan, "Unidad": ""})
    return pd.DataFrame(filas)


def _moda(valores):
    v = [x for x in valores if isinstance(x, str) and x]
    return collections.Counter(v).most_common(1)[0][0] if v else ""


def estadisticas_fases(var, res):
    """Punto 10: estadisticas de movimiento por fase del gesto."""
    t = var["tiempo_s"].to_numpy(float)
    dom = res["lado"]
    filas = []
    for j, (a, b, mov) in enumerate(res["runs"]):
        ia, ib = _idx_fase(a, b, res["i0"], res["i1"])
        fila = {"Fase": f"F{j + 1}", "Tipo": "Movimiento" if mov else "Sostenimiento",
                "t inicio video (s)": round(t[ia], 3), "t fin video (s)": round(t[ib], 3),
                "Duracion (s)": round(t[ib] - t[ia], 3)}
        for l in ("der", "izq"):
            fila[f"Mano {l}: rapidez pico (anchos/s)"] = _maxf(_tramo(var, ia, ib, f"mano_{l}_vel_sw_s"))
            fila[f"Mano {l}: rapidez media (anchos/s)"] = _meanf(_tramo(var, ia, ib, f"mano_{l}_vel_sw_s"))
            fila[f"Mano {l}: rapidez pico 3D (m/s)"] = _maxf(_tramo(var, ia, ib, f"mano_{l}_vel_m_s"))
            fila[f"Mano {l}: altura media (anchos)"] = _meanf(_tramo(var, ia, ib, f"mano_{l}_y_sw"))
            fila[f"Mano {l}: apertura media"] = _meanf(_tramo(var, ia, ib, f"mano_{l}_apertura"))
            fila[f"Mano {l}: palma predominante"] = _moda(var[f"mano_{l}_palma_orientacion"].iloc[ia:ib + 1])
            fila[f"Mano {l}: % frames validos"] = round(100 * _meanf(_tramo(var, ia, ib, f"qc_mano_{l}_valida")), 0)
        fila[f"Codo {dom}: flexion media (°)"] = _meanf(_tramo(var, ia, ib, f"codo_{dom}_flex_deg"))
        fila[f"Codo {dom}: rango (°)"] = _rangof(_tramo(var, ia, ib, f"codo_{dom}_flex_deg"))
        fila[f"Hombro {dom}: rango (°)"] = _rangof(_tramo(var, ia, ib, f"hombro_{dom}_elev_deg"))
        tot = np.nanmean(np.column_stack([_tramo(var, ia, ib, f"dedo_{dom}_{d}_total_flex_deg")
                                          for d in NOMBRES_DEDOS]), axis=1)
        fila[f"Dedos {dom}: flexion media (°)"] = _meanf(tot)
        fila["Distancia entre manos media (anchos)"] = _meanf(_tramo(var, ia, ib, "manos_dist_sw"))
        filas.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in fila.items()})
    return pd.DataFrame(filas)


def _actividad_dedos(var, lado):
    """Media del valor absoluto de la velocidad de flexion total de los 5 dedos (grados/s)."""
    cols = [f"dedo_{lado}_{d}_total_flex_vel_dps" for d in NOMBRES_DEDOS]
    return np.nanmean(np.abs(np.column_stack([var[c].to_numpy(float) for c in cols])), axis=1)


def sincronia(var, res, fases_lados):
    """Punto 9: relacion temporal cuerpo-dedos. Por fase del gesto: fases de dedos que coinciden, instante del pico
    de rapidez de la mano dominante y del pico de actividad de sus dedos, y su desfase (+ = los dedos despues). Al
    final, correlacion cruzada en todo el tramo activo (desfase con la maxima correlacion, ventana +-0.5 s)."""
    t = var["tiempo_s"].to_numpy(float)
    dom, i0, i1 = res["lado"], res["i0"], res["i1"]
    vel = var[f"mano_{dom}_vel_sw_s"].to_numpy(float)
    act = _actividad_dedos(var, dom)
    filas = []
    for j, (a, b, mov) in enumerate(res["runs"]):
        ia, ib = _idx_fase(a, b, i0, i1)
        fila = {"Fase gesto": f"F{j + 1}", "Tipo": "Movimiento" if mov else "Sostenimiento",
                "t inicio (s)": round(t[ia], 3), "t fin (s)": round(t[ib], 3)}
        for l in ("der", "izq"):
            etq = var[f"fase_dedos_{l}"].iloc[ia:ib + 1]
            presentes = [e for e in dict.fromkeys(etq) if e]
            fases = (fases_lados or {}).get(l) or []
            fila[f"Fases dedos {l}"] = "; ".join(
                f"{e} {texto_fase_dedos(fases[int(e[1:]) - 1], 'es')[0]}" for e in presentes) if fases else ""
        kv, ka = _nanargmax_abs(vel[ia:ib + 1]), _nanargmax_abs(act[ia:ib + 1])
        fila["t pico rapidez mano dom. (s)"] = round(t[ia + kv], 3) if kv is not None else np.nan
        fila["t pico actividad dedos dom. (s)"] = round(t[ia + ka], 3) if ka is not None else np.nan
        fila["Desfase dedos - mano (s)"] = (round(t[ia + ka] - t[ia + kv], 3)
                                            if kv is not None and ka is not None else np.nan)
        filas.append(fila)
    # correlacion cruzada en el tramo activo
    va, aa = vel[i0:i1 + 1].copy(), act[i0:i1 + 1].copy()
    ok = np.isfinite(va) & np.isfinite(aa)
    lag_s, rmax = np.nan, np.nan
    if ok.sum() >= 10:
        va, aa = va[ok], aa[ok]
        va, aa = (va - va.mean()) / (va.std() + 1e-12), (aa - aa.mean()) / (aa.std() + 1e-12)
        dt = 1.0 / _fs(t[i0:i1 + 1])
        L = min(int(round(0.5 / dt)), len(va) - 2)
        mejores = []
        for k in range(-L, L + 1):
            x, y = (va[:len(va) - k], aa[k:]) if k >= 0 else (va[-k:], aa[:len(aa) + k])
            mejores.append((float(np.mean(x * y)), k))
        rmax, kbest = max(mejores)
        lag_s = kbest * dt if abs(kbest) < L else np.nan      # maximo en el borde: desfase no determinado
    filas.append({"Fase gesto": "Tramo activo", "Tipo": "correlacion cruzada",
                  "t inicio (s)": round(t[i0], 3), "t fin (s)": round(t[i1], 3),
                  "Desfase dedos - mano (s)": round(lag_s, 3) if np.isfinite(lag_s) else np.nan,
                  "Correlacion maxima (r)": round(rmax, 3) if np.isfinite(rmax) else np.nan,
                  "Nota": ("" if np.isfinite(lag_s) else
                           "maximo en el borde de la ventana (+-0.5 s) o sin datos suficientes: desfase no determinado")})
    return pd.DataFrame(filas)


# ---------------- figura de posicion, orientacion y apertura de las manos (punto 10) ----------------
def pie_espacial(res, sena, persona, rep, lang="en"):
    if lang == "es":
        p = [f"Figura. Posición, orientación y apertura de las manos en la seña {sena} ({persona}, repetición {rep}). "
             "A: trayectoria 2D de cada mano durante el tramo activo respecto al punto medio de los hombros, en "
             "anchos de hombro (x + = derecha de la imagen, y + = arriba); el color indica el tiempo y las marcas F# "
             "el inicio de cada fase. B: altura de las manos. C: profundidad 3D estimada respecto a los hombros "
             "(+ = hacia la cámara). D: orientación de la palma (1 = de frente a la cámara, -1 = dorso, 0 = de "
             "canto). E: apertura de la mano (distancia pulgar-meñique / largo de la palma). La tabla resume cada "
             "fase del gesto. Naranja = movimiento, azul = sostenimiento; los huecos son frames rechazados por el "
             "control de calidad.",
             NOTA_DERIV["es"][0].upper() + NOTA_DERIV["es"][1:]]
    else:
        p = [f"Figure. Hand position, orientation and opening in the sign {sena} ({persona}, repetition {rep}). A: 2D "
             "trajectory of each hand during the active segment relative to the shoulder midpoint, in shoulder widths "
             "(x + = image right, y + = up); colour encodes time and the F# marks the start of each phase. B: hand "
             "height. C: estimated 3D depth relative to the shoulders (+ = toward the camera). D: palm orientation "
             "(1 = facing the camera, -1 = back of the hand, 0 = edge-on). E: hand opening (thumb-little fingertip "
             "distance / palm length). The table summarises each gesture phase. Orange = movement, blue = hold; gaps "
             "are frames rejected by the quality control.",
             NOTA_DERIV["en"][0].upper() + NOTA_DERIV["en"][1:]]
    return p


def figura_espacial(ruta, var, res, stats, sena, persona, rep, nariz=None):
    i0, i1, dur = res["i0"], res["i1"], res["dur"]
    sl = slice(i0, i1 + 1)
    t = var["tiempo_s"].to_numpy(float)
    x = t[sl] - t[i0]
    col = {"izq": "#1f77b4", "der": "#d62728"}
    fx = lambda i: i / (N - 1) * dur    # noqa: E731
    pie = _envolver(pie_espacial(res, sena, persona, rep, "en"), 190)
    fig = plt.figure(figsize=(15, 15.5 + 0.18 * pie.count("\n")))
    gs = fig.add_gridspec(6, 2, width_ratios=[1.0, 1.35], height_ratios=[1, 1, 1, 1, 1.25, 0.25 + 0.12 * pie.count("\n")],
                          hspace=0.45, wspace=0.18, top=0.95)
    axA = fig.add_subplot(gs[0:4, 0])
    ejes = [fig.add_subplot(gs[k, 1]) for k in range(4)]
    for k in range(1, 4):
        ejes[k].sharex(ejes[0])

    # A: trayectorias
    for l in ("der", "izq"):
        xs, ys = var[f"mano_{l}_x_sw"].to_numpy(float)[sl], var[f"mano_{l}_y_sw"].to_numpy(float)[sl]
        axA.plot(xs, ys, color=col[l], lw=1.4, alpha=0.6, label=f"{LADO_TXT[l].capitalize()} hand")
        sc = axA.scatter(xs, ys, c=x, cmap="viridis", s=12, zorder=3)
        for j, (a, b, mov) in enumerate(res["runs"]):
            ia = _idx_fase(a, b, i0, i1)[0] - i0
            if 0 <= ia < len(xs) and np.isfinite(xs[ia]):
                axA.annotate(f"F{j + 1}", (xs[ia], ys[ia]), fontsize=8, color=col[l], xytext=(3, 3),
                             textcoords="offset points")
    fig.colorbar(sc, ax=axA, orientation="horizontal", fraction=0.035, pad=0.09,
                 label="Time within the active segment (s)")
    ref = {"Shoulders": [(-0.5, 0.0), (0.5, 0.0)], "Chest": [(0.0, -PECHO_Y)]}
    axA.plot([p[0] for p in ref["Shoulders"]], [p[1] for p in ref["Shoulders"]], "s-", color="0.4", ms=8,
             label="Shoulders")
    axA.plot(*ref["Chest"][0], "D", color="0.4", ms=7, label="Chest (approx.)")
    if nariz is not None and all(np.isfinite(nariz)):
        axA.plot(*nariz, "o", color="0.2", ms=9, label="Nose")
    axA.set_aspect("equal", adjustable="datalim")
    axA.grid(alpha=0.3)
    axA.set_xlabel("Horizontal position (shoulder widths; + = image right)")
    axA.set_ylabel("Height (shoulder widths above the shoulder line)")
    axA.set_title("A. Hand trajectories (frontal plane)", fontsize=11, fontweight="bold", loc="left")
    axA.legend(fontsize=8, loc="lower left")

    paneles = [("B. Hand height (shoulder widths)", "y_sw"), ("C. Depth toward the camera (m)", "Z_m"),
               ("D. Palm facing the camera (-1 back … 1 palm)", "palma_hacia_camara"),
               ("E. Hand opening (thumb-little / palm length)", "apertura")]
    for ax, (titulo, suf) in zip(ejes, paneles):
        for l in ("der", "izq"):
            c = f"mano_{l}_{suf}"
            if c in var.columns:
                ax.plot(x, var[c].to_numpy(float)[sl], color=col[l], lw=1.8, label=f"{LADO_TXT[l]}")
        for a, b, mov in res["runs"]:
            ax.axvspan(fx(a), fx(min(b, N - 1)), color="#ffb347" if mov else "#9ecae1", alpha=0.16)
        for bi in sorted({r[0] for r in res["runs"]} | {N - 1}):
            ax.axvline(fx(bi), color="red", ls=":", lw=1)
        ax.set_title(titulo, fontsize=10, fontweight="bold", loc="left")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, dur)
        if suf == "palma_hacia_camara":
            ax.set_ylim(-1.05, 1.05)
        if ax is not ejes[-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
    ejes[0].legend(fontsize=8, loc="upper right", ncol=2)
    ejes[-1].set_xlabel("Time within the active segment (s)")

    # tabla de estadisticas por fase
    axT = fig.add_subplot(gs[4, :])
    axT.axis("off")
    cab = ["Phase", "Type", "Start-end (s)", "Peak speed R/L\n(sw/s)", "Mean height R/L\n(sw)",
           "Mean opening R/L", "Palm R/L", "Valid frames R/L (%)"]
    f2 = lambda v: "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.2f}"   # noqa: E731
    f0 = lambda v: "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.0f}"   # noqa: E731
    pal = {"camara": "palm", "dorso": "back", "canto": "edge", "": "--"}
    filas = []
    for _, r in stats.iterrows():
        filas.append([r["Fase"], "movement" if r["Tipo"] == "Movimiento" else "hold",
                      f"{r['t inicio video (s)']:.2f}-{r['t fin video (s)']:.2f}",
                      f"{f2(r['Mano der: rapidez pico (anchos/s)'])} / {f2(r['Mano izq: rapidez pico (anchos/s)'])}",
                      f"{f2(r['Mano der: altura media (anchos)'])} / {f2(r['Mano izq: altura media (anchos)'])}",
                      f"{f2(r['Mano der: apertura media'])} / {f2(r['Mano izq: apertura media'])}",
                      f"{pal.get(r['Mano der: palma predominante'], '--')} / "
                      f"{pal.get(r['Mano izq: palma predominante'], '--')}",
                      f"{f0(r['Mano der: % frames validos'])} / {f0(r['Mano izq: % frames validos'])}"])
    tabla = axT.table(cellText=filas, colLabels=cab, loc="upper center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8.5)
    tabla.scale(1, 1.6)
    axT.set_title("Statistics per gesture phase (R = right hand, L = left hand)", fontsize=10, fontweight="bold",
                  loc="left")
    if PIE_EN_FIGURA:
        axP = fig.add_subplot(gs[5, :])
        axP.axis("off")
        axP.text(0, 1, pie, va="top", ha="left", fontsize=9, family="DejaVu Sans")
    fig.suptitle(TITULOS["espacial"]["en"].format(s=sena), fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(ruta, dpi=DPI_FIG, bbox_inches="tight")
    plt.close(fig)


def nariz_rel(lm):
    """Posicion mediana de la nariz respecto al punto medio de los hombros, en anchos de hombro (x, y + = arriba)."""
    S = (xy(lm, L_SHOULDER) + xy(lm, R_SHOULDER)) / 2
    sw = np.nanmedian(np.linalg.norm(xy(lm, L_SHOULDER) - xy(lm, R_SHOULDER), axis=1))
    r = (xy(lm, 0) - S) / sw
    return float(np.nanmedian(r[:, 0])), float(-np.nanmedian(r[:, 1]))


# ---------------- documento bilingue con los textos de las figuras (punto 11) ----------------
def escribir_documento_figuras(ruta, encabezado, figuras):
    """figuras: lista de (archivo_png, titulo_es, parrafos_es, titulo_en, parrafos_en). Archivo de texto UTF-8 (con
    BOM para que el Bloc de notas y Word muestren bien las tildes)."""
    lineas = ["DESCRIPCIÓN DE LAS FIGURAS / FIGURE DESCRIPTIONS", "=" * 70, encabezado, ""]
    for png, tit_es, par_es, tit_en, par_en in figuras:
        lineas += ["=" * 70, f"Archivo / File: {png}", "=" * 70, "",
                   "[ESPAÑOL]", tit_es, ""] + [p + "\n" for p in par_es] + \
                  ["[ENGLISH]", tit_en, ""] + [p + "\n" for p in par_en] + [""]
    with open(ruta, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lineas))


# ======================================================================================================================
# 11. DATASETS (se reconstruyen completos: nunca quedan filas duplicadas)
# ======================================================================================================================
def construir_dataset():
    """angulos_frames, landmarks_frames, resumen_repeticiones y secuencias (csv + npz) en dataset/."""
    reps = list(repeticiones_guardadas("angulos.csv"))
    if not reps:
        print("No hay repeticiones en 'resultados/' para armar el dataset.")
        return
    os.makedirs(CARPETA_DATASET, exist_ok=True)
    cols = encabezados_angulos()[2:]
    df_ang, df_lm, resumen, secuencias, meta = [], [], [], [], []
    for carpeta, sena, persona, rep in reps:
        f_lm = os.path.join(carpeta, "landmarks.csv")
        if not os.path.exists(f_lm):
            print(f"  omitido (falta landmarks.csv): {carpeta}")
            continue
        an, lm = pd.read_csv(os.path.join(carpeta, "angulos.csv")), pd.read_csv(f_lm)
        # archivos viejos con menos angulos: mismo juego de columnas en todos (lo que falta queda NaN)
        faltan = [c for c in cols if c not in an.columns]
        if faltan:
            print(f"  {sena}/{persona}_rep{rep}: archivo antiguo, le faltan {len(faltan)} angulos (quedan vacios)")
        an = an.reindex(columns=["frame", "tiempo_ms"] + cols)
        try:
            res = analizar(lm, an, sena)
        except ErrorAnalisis as e:
            print(f"  {sena}/{persona}_rep{rep}: sin tramo activo ({e}); entra a los CSV por frame pero no a "
                  "resumen/secuencias")
            res = None
        activo = np.zeros(len(an), dtype=int)
        if res is not None:
            activo[res["i0"]:res["i1"] + 1] = 1
        for df, destino in ((an, df_ang), (lm, df_lm)):
            d = df.copy()
            d.insert(0, "en_tramo_activo", activo)
            d.insert(0, "rep", rep)
            d.insert(0, "persona", persona)
            d.insert(0, "sena", sena)
            destino.append(d)
        if res is None:
            continue
        seg = an.iloc[res["i0"]:res["i1"] + 1][cols]
        fila = {"sena": sena, "persona": persona, "rep": rep, "brazo_dominante": res["lado"],
                "duracion_s": round(res["dur"], 3), "frames_tramo": len(seg)}
        for c in cols:
            fila[f"{c}_media"], fila[f"{c}_std"] = seg[c].mean(), seg[c].std()
            fila[f"{c}_min"], fila[f"{c}_max"] = seg[c].min(), seg[c].max()
        resumen.append(fila)
        X = np.column_stack([remuestrear_n(seg[c].to_numpy(float), T_SEQ) for c in cols])
        secuencias.append(np.nan_to_num(X, nan=0.0))   # angulo no detectado -> 0 (formato para ML)
        meta.append((sena, persona, rep))

    if not df_ang:
        print("No se pudo armar el dataset.")
        return
    pd.concat(df_ang, ignore_index=True).to_csv(os.path.join(CARPETA_DATASET, "angulos_frames.csv"), index=False)
    pd.concat(df_lm, ignore_index=True).to_csv(os.path.join(CARPETA_DATASET, "landmarks_frames.csv"), index=False)
    if resumen:
        pd.DataFrame(resumen).to_csv(os.path.join(CARPETA_DATASET, "resumen_repeticiones.csv"), index=False)
        _guardar_secuencias("secuencias", secuencias, meta, cols)
    print(f"\nDataset en '{CARPETA_DATASET}': {len(reps)} repeticiones revisadas, "
          f"{len(resumen)} con secuencia valida, senas: {sorted({m_[0] for m_ in meta})}")


def _guardar_secuencias(nombre, secuencias, meta, columnas, plano=None):
    """(repeticiones, T_SEQ, canales) -> nombre.csv (una fila por repeticion) y nombre.npz (X, y, personas, reps)."""
    X = np.stack(secuencias)
    cols_seq = [f"{c}_t{k:02d}" for k in range(T_SEQ) for c in columnas]
    df_seq = pd.DataFrame(X.reshape(len(X), -1), columns=cols_seq)
    if plano:
        df_seq.insert(0, "plano", plano)
    df_seq.insert(0, "rep", [m_[2] for m_ in meta])
    df_seq.insert(0, "persona", [m_[1] for m_ in meta])
    df_seq.insert(0, "sena", [m_[0] for m_ in meta])
    df_seq.to_csv(os.path.join(CARPETA_DATASET, f"{nombre}.csv"), index=False)
    np.savez_compressed(os.path.join(CARPETA_DATASET, f"{nombre}.npz"), X=X,
                        y=np.array([m_[0] for m_ in meta]), personas=np.array([m_[1] for m_ in meta]),
                        reps=np.array([m_[2] for m_ in meta]), columnas=np.array(columnas))


def construir_dataset_cinematica():
    """cinematica_frames, cinematica_resumen y cinematica_secuencias (flexion/velocidad/aceleracion por dedo y
    codo/hombro)."""
    try:
        _construir_dataset_cinematica()
    except Exception as e:
        traceback.print_exc()
        print(f"AVISO: no se pudo armar el dataset de cinematica ({e}).")


def _construir_dataset_cinematica():
    reps = list(repeticiones_guardadas("angulos.csv"))
    if not reps:
        return
    os.makedirs(CARPETA_DATASET, exist_ok=True)
    frames, resumen, seqs, meta, canales_ref = [], [], [], [], None
    cols_ang = encabezados_angulos()[2:]
    for carpeta, sena, persona, rep in reps:
        f_lm = os.path.join(carpeta, "landmarks.csv")
        if not os.path.exists(f_lm):
            continue
        an = pd.read_csv(os.path.join(carpeta, "angulos.csv")).reindex(columns=["frame", "tiempo_ms"] + cols_ang)
        lm = pd.read_csv(f_lm)
        t = pd.to_numeric(an["tiempo_ms"], errors="coerce").to_numpy(dtype=float) / 1000.0
        if len(t) < 20 or np.any(np.diff(t) <= 0):
            print(f"  cinematica: {sena}/{persona}_rep{rep} omitida (tiempos invalidos)")
            continue
        canales = {}
        qc = calidad_por_frame(lm, an)
        for lado in ("izq", "der"):
            kin, _ = cinematica_mano(an, t, lado, qc)
            for d in NOMBRES_DEDOS:
                for k in ("flex", "vel", "acc"):
                    canales[f"{lado}_{d}_{k}"] = kin[d][k]
            kb = cinematica_brazo(an, t, lado)
            for nom in ("codo", "hombro"):
                for k in ("ang", "vel", "acc"):
                    canales[f"{lado}_{nom}_{k}"] = kb[nom][k]
        canales_ref = list(canales)
        try:
            res = analizar(lm, an, sena)
        except ErrorAnalisis:
            res = None
        activo = np.zeros(len(an), dtype=int)
        if res is not None:
            activo[res["i0"]:res["i1"] + 1] = 1
        df = pd.DataFrame({"sena": sena, "persona": persona, "rep": rep, "plano": PLANO,
                           "en_tramo_activo": activo, "frame": an["frame"], "tiempo_ms": an["tiempo_ms"], **canales})
        frames.append(df)
        if res is None:
            continue
        seg = df.iloc[res["i0"]:res["i1"] + 1][canales_ref]
        fila = {"sena": sena, "persona": persona, "rep": rep, "plano": PLANO, "brazo_dominante": res["lado"],
                "duracion_s": round(res["dur"], 3)}
        for c in canales_ref:
            fila[f"{c}_media"], fila[f"{c}_std"] = seg[c].mean(), seg[c].std()
            fila[f"{c}_min"], fila[f"{c}_max"] = seg[c].min(), seg[c].max()
            fila[f"{c}_absmax"] = seg[c].abs().max()
        resumen.append(fila)
        X = np.column_stack([remuestrear_n(seg[c].to_numpy(float), T_SEQ) for c in canales_ref])
        seqs.append(np.nan_to_num(X, nan=0.0))
        meta.append((sena, persona, rep))

    if not frames:
        return
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(CARPETA_DATASET, "cinematica_frames.csv"), index=False)
    if resumen:
        pd.DataFrame(resumen).to_csv(os.path.join(CARPETA_DATASET, "cinematica_resumen.csv"), index=False)
        _guardar_secuencias("cinematica_secuencias", seqs, meta, canales_ref, plano=PLANO)
    print(f"Dataset de cinematica en '{CARPETA_DATASET}': {len(resumen)} repeticiones, {len(canales_ref)} canales "
          "(flexion/velocidad/aceleracion de 5 dedos x 2 manos + codo y hombro x 2 brazos)")


def construir_dataset_variables():
    """dataset/variables_frames.csv (todas las repeticiones, una fila por frame), variables_resumen.csv (una fila por
    repeticion: media, desviacion, minimo, maximo y rango de cada variable numerica en el tramo activo) y
    variables_definiciones.csv."""
    try:
        reps = list(repeticiones_guardadas("variables.csv"))
        if not reps:
            return
        os.makedirs(CARPETA_DATASET, exist_ok=True)
        frames, resumen = [], []
        for carpeta, sena, persona, rep in reps:
            v = pd.read_csv(os.path.join(carpeta, "variables.csv"), low_memory=False)
            v = pd.concat([pd.DataFrame({"sena": sena, "persona": persona, "rep": rep}, index=v.index), v], axis=1)
            frames.append(v)
            act = v[v["en_tramo_activo"] == 1]
            if act.empty:
                continue
            fila = {"sena": sena, "persona": persona, "rep": rep, "frames_tramo": len(act),
                    "duracion_s": round(float(act["tiempo_s"].iloc[-1] - act["tiempo_s"].iloc[0]), 3)}
            num = [c for c in act.columns if c not in ("sena", "persona", "rep", "frame", "tiempo_ms", "tiempo_s",
                                                       "en_tramo_activo")
                   and pd.api.types.is_numeric_dtype(act[c])]
            for c in num:
                s_ = act[c]
                if c.startswith("qc_") and not c.endswith("_vis_min"):
                    fila[f"{c}_pct"] = round(100 * float(s_.mean()), 1)
                    continue
                fila[f"{c}_media"], fila[f"{c}_std"] = s_.mean(), s_.std()
                fila[f"{c}_min"], fila[f"{c}_max"] = s_.min(), s_.max()
                fila[f"{c}_rango"] = s_.max() - s_.min()
            resumen.append(fila)
        todo = pd.concat(frames, ignore_index=True)
        todo.to_csv(os.path.join(CARPETA_DATASET, "variables_frames.csv"), index=False)
        if resumen:
            pd.DataFrame(resumen).to_csv(os.path.join(CARPETA_DATASET, "variables_resumen.csv"), index=False)
        definiciones_variables(todo.columns[3:]).to_csv(os.path.join(CARPETA_DATASET, "variables_definiciones.csv"),
                                                        index=False)
        print(f"Dataset de variables en '{CARPETA_DATASET}': {len(reps)} repeticiones (variables_frames.csv, "
              "variables_resumen.csv, variables_definiciones.csv)")
    except Exception as e:
        traceback.print_exc()
        print(f"AVISO: no se pudo armar el dataset de variables ({e}).")


def construir_datasets():
    construir_dataset()
    construir_dataset_cinematica()
    construir_dataset_variables()


# ======================================================================================================================
# 12. ANGULOS FISIOLOGICOS 3D (coordenadas MUNDO de MediaPipe, convencion ISB)
# ======================================================================================================================
# Marco del tronco: U = arriba, R = derecha de la persona, F = adelante. La quiralidad del sistema de coordenadas se
# deduce de que la NARIZ queda delante de la linea de hombros.
#   - Tronco: por defecto la VERTICAL es el eje y del mundo (signo deducido de la nariz); --tronco cadera usa las
#     caderas (solo si se ven: en videos de cintura para arriba MediaPipe las inventa).
#   - Hombro (ISB): elevation = elevacion total; plane = plano de elevacion (0 = abduccion pura, 90 = flexion pura).
#     flexion/abduction por proyeccion quedan en NaN cuando la proyeccion es inestable.
#   - Control de calidad por LONGITUD DE SEGMENTOS (metros): si humero, antebrazo o una falange se desvia mas de la
#     tolerancia de su mediana en la repeticion, ese frame se descarta (NaN) para los angulos que lo usan.
#   - Muneca: antebrazo expresado en el marco de la mano (h, n, q) con atan2.
#   - Dedos: flexion CON SIGNO (negativa = hiperextension) de indice a menique; pulgar sin signo (MCP e IP).
# No se miden: rotacion axial del hombro, pronosupinacion, CMC del pulgar.
VENTANA_MEDIANA = 5            # filtro de mediana (frames) de los angulos 3D (no rellena frames descartados)
AVISO_FUERA_RANGO = 10.0       # % de frames fuera de rango anatomico a partir del cual se avisa
AVISO_SIN_DATO = 30.0          # % de frames descartados/sin dato a partir del cual se avisa
TOL_LONG_BRAZO = 0.20          # desviacion maxima (fraccion de la mediana) de humero / antebrazo
TOL_LONG_DEDO = 0.30           # idem para los segmentos de un dedo
MIN_PROY = 0.5                 # proyeccion minima del humero para reportar flexion / abduccion por proyeccion
ELEV_MIN_PLANO = 20.0          # por debajo de esta elevacion el plano de elevacion no esta definido
MODO_TRONCO = "vertical"       # "vertical" o "cadera"

IDX_POSE = {"hombro_izq": 11, "hombro_der": 12, "codo_izq": 13, "codo_der": 14,
            "muneca_izq": 15, "muneca_der": 16, "cadera_izq": 23, "cadera_der": 24, "nariz": 0}
DEDOS_IDX = DEDOS
# (articulacion, posicion en la cadena [muneca, p1, p2, p3, p4]); la articulacion p esta entre los segmentos p-1 y p
JUNTAS = {"pulgar": [("mcp", 2), ("ip", 3)],
          "indice": [("mcp", 1), ("pip", 2), ("dip", 3)], "medio": [("mcp", 1), ("pip", 2), ("dip", 3)],
          "anular": [("mcp", 1), ("pip", 2), ("dip", 3)], "menique": [("mcp", 1), ("pip", 2), ("dip", 3)]}
DEDO_EN = {"pulgar": "thumb", "indice": "index", "medio": "middle", "anular": "ring", "menique": "little"}
LADO_EN = {"izq": "left", "der": "right"}
# rango anatomico generoso (grados) para marcar frames imposibles; no es el rango "normal" de una persona
RANGO_OK = {"shoulder_elevation": (0, 180), "shoulder_plane": (-90, 180), "shoulder_flexion": (-80, 190),
            "shoulder_abduction": (-60, 190), "elbow_flexion": (-20, 160),
            "wrist_flexion": (-90, 90), "wrist_radial_deviation": (-50, 40),
            "mcp": (-40, 110), "pip": (-15, 120), "dip": (-15, 100), "thumb_mcp": (0, 90), "thumb_ip": (0, 100)}


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-9, np.nan, n)


def dot(a, b):
    return np.sum(a * b, axis=-1)


def cr(a, b, chi):
    """Producto cruz corregido por la quiralidad del sistema de coordenadas (chi = +1 mano derecha)."""
    return chi * np.cross(a, b)


def angulo_dirs(d1, d2):
    """Angulo (0..180) entre dos direcciones."""
    return np.degrees(np.arccos(np.clip(dot(unit(d1), unit(d2)), -1.0, 1.0)))


def longitud_ok(a, b, tol):
    """True donde el segmento a-b mide lo mismo que su mediana (+-tol). NaN cuenta como no valido."""
    L = np.linalg.norm(a - b, axis=-1)
    m = np.nanmedian(L)
    if not np.isfinite(m) or m <= 0:
        return np.zeros(len(L), dtype=bool)
    return np.abs(L / m - 1.0) <= tol


def enmascarar(x, ok):
    x = np.asarray(x, float).copy()
    x[~ok] = np.nan
    return x


def marco_tronco(P, modo=None):
    """Ejes del tronco: arriba U, derecha de la persona R, adelante F (ya corregido por quiralidad) y chi."""
    modo = modo or MODO_TRONCO
    mid_h = (P["hombro_izq"] + P["hombro_der"]) / 2
    R0 = P["hombro_der"] - P["hombro_izq"]
    if modo == "cadera":
        up = mid_h - (P["cadera_izq"] + P["cadera_der"]) / 2
    else:
        dy = np.nanmedian((P["nariz"] - mid_h)[:, 1])
        if not np.isfinite(dy):
            print("  AVISO: no se vio la nariz; asumo que +y del mundo apunta hacia abajo.")
            s = -1.0
        else:
            s = 1.0 if dy > 0 else -1.0          # arriba = el lado de y donde esta la nariz respecto a los hombros
        up = np.zeros_like(mid_h)
        up[:, 1] = s
    U = unit(up)
    R = unit(R0 - dot(R0, U)[:, None] * U)
    F0 = np.cross(U, R)                           # "adelante" si el sistema es de mano derecha
    d = dot(P["nariz"] - mid_h, F0)
    d = d[np.isfinite(d)]
    chi = 1.0
    if len(d) == 0:
        print("  AVISO: no se vio la nariz; asumo sistema de coordenadas de mano derecha.")
    elif np.median(d) < 0:
        chi = -1.0
    return U, R, chi * F0, chi


def fis_hombro(P, lado, U, R, F, ok):
    """Elevacion, plano de elevacion (ISB), flexion y abduccion por proyeccion."""
    v = unit(P[f"codo_{lado}"] - P[f"hombro_{lado}"])
    O = R if lado == "der" else -R                # hacia afuera del cuerpo
    vu, vo, vf = dot(v, U), dot(v, O), dot(v, F)
    elevacion = np.degrees(np.arccos(np.clip(-vu, -1.0, 1.0)))
    plano = np.degrees(np.arctan2(vf, vo))
    plano[elevacion < ELEV_MIN_PLANO] = np.nan
    flexion = np.degrees(np.arctan2(vf, -vu))
    flexion[np.hypot(vu, vf) < MIN_PROY] = np.nan
    abduccion = np.degrees(np.arctan2(vo, -vu))
    abduccion[np.hypot(vu, vo) < MIN_PROY] = np.nan
    return [enmascarar(x, ok) for x in (elevacion, plano, flexion, abduccion)]


def fis_codo(P, lado, ok):
    a = unit(P[f"hombro_{lado}"] - P[f"codo_{lado}"])
    b = unit(P[f"muneca_{lado}"] - P[f"codo_{lado}"])
    return enmascarar(180.0 - angulo_dirs(a, b), ok)


def marco_mano(H, lado, chi):
    """h = eje de la mano (muneca -> nudillo medio), n = normal palmar, q = direccion radial en el plano de la palma."""
    h = unit(H[9] - H[0])
    rho = unit(H[5] - H[17])                      # indice - menique: apunta al lado radial
    n = unit(cr(rho, h, chi)) if lado == "der" else unit(cr(h, rho, chi))
    q = unit(rho - dot(rho, h)[:, None] * h)
    return h, n, q


def fis_muneca(P, H, lado, chi):
    """Flexion (+ hacia la palma) y desviacion radial (+ hacia el pulgar). Se compara la DIRECCION del antebrazo
    (modelo de pose) con el marco de la mano (modelo de mano); no se usan posiciones porque cada modelo tiene su
    origen."""
    a = unit(P[f"muneca_{lado}"] - P[f"codo_{lado}"])
    h, n, q = marco_mano(H, lado, chi)
    return np.degrees(np.arctan2(-dot(a, n), dot(a, h))), np.degrees(np.arctan2(-dot(a, q), dot(a, h)))


def fis_dedos(H, n):
    """Flexion de cada articulacion = angulo entre segmentos consecutivos (0 = recta). Con signo de indice a menique
    (negativo = hiperextension). Se descartan los frames donde un segmento del dedo cambia de longitud mas de
    TOL_LONG_DEDO."""
    out, oks = {}, {}
    for d, idx in DEDOS_IDX.items():
        cad = [0] + idx
        segs = [unit(H[cad[k + 1]] - H[cad[k]]) for k in range(4)]
        ok = np.ones(len(n), dtype=bool)
        for k in range(4):
            ok &= longitud_ok(H[cad[k + 1]], H[cad[k]], TOL_LONG_DEDO)
        for art, p in JUNTAS[d]:
            ang = angulo_dirs(segs[p - 1], segs[p])
            if d != "pulgar":
                # signo = sentido de giro respecto al eje de flexion del dedo (normal al plano metacarpiano-palma);
                # funciona con flexiones acumuladas > 90 grados
                eje = np.cross(segs[0], n)
                sgn = np.sign(dot(np.cross(segs[p - 1], segs[p]), eje))
                sgn[sgn == 0] = 1.0
                ang = ang * sgn
            out[f"{DEDO_EN[d]}_{art}"] = enmascarar(ang, ok)
        oks[DEDO_EN[d]] = ok
    return out, oks


def pts3d(lm, prefijo, i):
    return np.column_stack([pd.to_numeric(lm[f"{prefijo}_{i}_{e}"], errors="coerce").to_numpy(float) for e in "xyz"])


def tiene_mundo(lm):
    return "mundo_pose_0_x" in lm.columns


def idx_pose_actual():
    """Indices de la pose segun INVERTIR_LADOS."""
    idx = dict(IDX_POSE)
    idx.update({"hombro_izq": L_SHOULDER, "hombro_der": R_SHOULDER, "codo_izq": L_ELBOW, "codo_der": R_ELBOW,
                "muneca_izq": L_WRIST, "muneca_der": R_WRIST, "cadera_izq": L_HIP, "cadera_der": R_HIP})
    return idx


def _fis_calcular(lm, idx=None, modo=None):
    """Devuelve (angulos, chi, qc). qc = banderas de control de calidad por frame (0/1) y metadatos."""
    idx = idx or IDX_POSE
    modo = modo or MODO_TRONCO
    if not tiene_mundo(lm):
        raise ValueError("este landmarks.csv no tiene coordenadas 3D (version antigua): vuelve a analizar el video")
    P = {nombre: pts3d(lm, "mundo_pose", i) for nombre, i in idx.items()}
    U, R, F, chi = marco_tronco(P, modo)
    cols, qc = {}, {}
    qc["qc_pose_landmarks_present"] = np.all([np.isfinite(v).all(axis=1) for v in P.values()], axis=0).astype(int)
    for lado in ("izq", "der"):
        L = LADO_EN[lado]
        ok = (longitud_ok(P[f"hombro_{lado}"], P[f"codo_{lado}"], TOL_LONG_BRAZO)
              & longitud_ok(P[f"codo_{lado}"], P[f"muneca_{lado}"], TOL_LONG_BRAZO))
        qc[f"qc_{L}_arm_segment_length_ok"] = ok.astype(int)
        (cols[f"shoulder_{L}_elevation"], cols[f"shoulder_{L}_plane"],
         cols[f"shoulder_{L}_flexion"], cols[f"shoulder_{L}_abduction"]) = fis_hombro(P, lado, U, R, F, ok)
        cols[f"elbow_{L}_flexion"] = fis_codo(P, lado, ok)
        pref = "mundo_manoI" if lado == "izq" else "mundo_manoD"
        if f"{pref}_0_x" in lm.columns:
            H = [pts3d(lm, pref, i) for i in range(21)]
            qc[f"qc_{L}_hand_landmarks_present"] = np.all([np.isfinite(h).all(axis=1) for h in H], axis=0).astype(int)
            fl, rd = fis_muneca(P, H, lado, chi)
            cols[f"wrist_{L}_flexion"] = enmascarar(fl, ok)
            cols[f"wrist_{L}_radial_deviation"] = enmascarar(rd, ok)
            _, n, _ = marco_mano(H, lado, chi)
            angs, oks = fis_dedos(H, n)
            for nombre, v in angs.items():
                cols[f"{L}_{nombre}_flexion"] = v
            for dedo, o in oks.items():
                qc[f"qc_{L}_{dedo}_segment_length_ok"] = o.astype(int)
    df = pd.DataFrame(cols)
    filtrado = df.rolling(VENTANA_MEDIANA, center=True, min_periods=1).median()
    df = filtrado.where(df.notna())               # el filtro no rellena los frames descartados
    qc = pd.DataFrame(qc)
    qc["trunk_axis"] = modo
    qc["coord_chirality"] = "right-handed" if chi > 0 else "left-handed (corrected)"
    return df, chi, qc


def fis_calcular(lm, idx=None, modo=None):
    """lm: DataFrame de landmarks.csv (con columnas mundo_*). Devuelve (angulos, chi)."""
    df, chi, _ = _fis_calcular(lm, idx, modo)
    return df, chi


def rango_de(c):
    if c.startswith("shoulder"):
        for suf in ("elevation", "plane", "flexion", "abduction"):
            if c.endswith("_" + suf):
                return RANGO_OK[f"shoulder_{suf}"]
    if c.startswith("elbow"):
        return RANGO_OK["elbow_flexion"]
    if c.startswith("wrist"):
        return RANGO_OK["wrist_radial_deviation" if "radial" in c else "wrist_flexion"]
    p = c.split("_")                              # left_index_pip_flexion
    if len(p) >= 4:
        if p[1] == "thumb":
            return RANGO_OK.get("thumb_" + p[2], (-30, 180))
        return RANGO_OK.get(p[2], (-30, 180))
    return (-180, 360)


def resumen_rep(df, i0, i1):
    seg = df.iloc[i0:i1 + 1]
    fila = {}
    for c in df.columns:
        s = seg[c]
        lo, hi = rango_de(c)
        ok = s.dropna()
        fila[f"{c}_min"], fila[f"{c}_max"] = s.min(), s.max()
        fila[f"{c}_rom"] = s.max() - s.min() if len(ok) else np.nan
        fila[f"{c}_rom_p5_p95"] = float(np.percentile(ok, 95) - np.percentile(ok, 5)) if len(ok) >= 5 else np.nan
        fila[f"{c}_mean"] = s.mean()
        fila[f"{c}_pct_fuera_rango"] = round(100.0 * float(((ok < lo) | (ok > hi)).mean()), 1) if len(ok) else np.nan
        fila[f"{c}_pct_sin_dato"] = round(100.0 * float(s.isna().mean()), 1)
    return fila


def fis_definiciones():
    f = []

    def add(col, mide, cero, positivo, nota, unidad="deg"):
        f.append({"column": col, "unit": unidad, "measures": mide, "zero": cero, "positive_direction": positivo,
                  "notes": nota})

    for L in ("left", "right"):
        add(f"shoulder_{L}_elevation", "total elevation of the humerus relative to the trunk (ISB; angle between the "
            "upper arm and the downward trunk axis)", "arm hanging along the trunk",
            "always >= 0 (90 = horizontal, 180 = overhead)",
            "trunk axes: vertical from the world y axis, lateral from the shoulder line")
        add(f"shoulder_{L}_plane", "plane of elevation (ISB): direction of the elevation relative to the frontal plane",
            "pure abduction (arm in the frontal plane)",
            "90 = pure flexion (sagittal plane), negative = behind the frontal plane",
            "undefined (NaN) when elevation < 20 deg")
        add(f"shoulder_{L}_flexion", "shoulder flexion/extension (arm projected on the sagittal plane of the trunk)",
            "arm hanging along the trunk", "+ forward (flexion), - backward (extension)",
            "projection method; NaN when the arm is close to the frontal plane (unstable)")
        add(f"shoulder_{L}_abduction", "shoulder abduction/adduction (arm projected on the frontal plane of the trunk)",
            "arm hanging along the trunk", "+ away from the body (abduction), - across the body (adduction)",
            "projection method; NaN when the arm is close to the sagittal plane; axial rotation cannot be measured")
        add(f"elbow_{L}_flexion", "elbow flexion = 180 deg minus the 3D angle shoulder-elbow-wrist",
            "arm fully extended", "+ flexion", "forearm pronation/supination cannot be measured")
        add(f"wrist_{L}_flexion", "wrist flexion/extension: forearm axis (pose model) expressed in the hand frame "
            "(hand model); atan2(-a.n, a.h)", "hand aligned with the forearm",
            "+ flexion (toward the palm), - extension",
            "combines two models with different origins; assumes both share the camera axes; least accurate angle")
        add(f"wrist_{L}_radial_deviation", "radial/ulnar deviation: atan2(-a.q, a.h) in the hand frame",
            "hand aligned with the forearm", "+ radial (toward the thumb), - ulnar", "same limitations as wrist flexion")
        for d, juntas in JUNTAS.items():
            for art, _ in juntas:
                signed = d != "pulgar"
                add(f"{L}_{DEDO_EN[d]}_{art}_flexion",
                    f"flexion of the {DEDO_EN[d]} {art.upper()} joint = 3D angle between adjacent segments",
                    "finger straight", "+ flexion" + (", - hyperextension (sign from the palm normal)" if signed else ""),
                    ("signed; noisy near 0 deg" if signed else "unsigned (thumb)") +
                    "; frames with an inconsistent finger segment length are discarded")
    na = "n/a"
    add("frame", "video frame index (1-based)", na, na, "", "-")
    add("tiempo_ms", "timestamp of the frame from the start of the video", na, na, "", "ms")
    add("en_tramo_activo", "1 if the frame lies inside the active segment detected from wrist speed", na, na,
        "summary statistics are computed only on active frames", "bool")
    add("qc_pose_landmarks_present", "1 if all pose world landmarks used (nose, shoulders, elbows, wrists, hips) are "
        "finite", na, na, "0 = landmark missing in this frame; dependent angles are NaN", "bool")
    for L in ("left", "right"):
        add(f"qc_{L}_arm_segment_length_ok", f"1 if the {L} humerus and forearm lengths are within "
            f"+-{TOL_LONG_BRAZO:.0%} of their repetition median (3D world coordinates, metres)", na, na,
            "0 = depth/occlusion error suspected; shoulder, elbow and wrist angles of this side are NaN", "bool")
        add(f"qc_{L}_hand_landmarks_present", f"1 if the 21 {L} hand world landmarks are finite", na, na,
            "0 = hand not detected; wrist and finger angles are NaN", "bool")
        for d in DEDOS_IDX:
            add(f"qc_{L}_{DEDO_EN[d]}_segment_length_ok", f"1 if all {DEDO_EN[d]} segment lengths are within "
                f"+-{TOL_LONG_DEDO:.0%} of their repetition median", na, na,
                "0 = frame discarded for that finger's joint angles", "bool")
    add("trunk_axis", "how the trunk up-axis was defined: 'vertical' = world y axis (sign from the nose), "
        "'cadera' = hip midpoint to shoulder midpoint", na, na, "constant per repetition", "-")
    add("coord_chirality", "handedness of the world coordinate system, inferred from the nose being anterior to the "
        "shoulder line; cross products are corrected when left-handed", na, na, "constant per repetition", "-")
    add("NaN (any angle)", "angle not computable in this frame: landmark missing, qc flag = 0, projection unstable "
        "(shoulder flexion/abduction) or elevation < 20 deg (shoulder plane)", na, na,
        "the median filter never fills NaN frames", "-")
    return pd.DataFrame(f)


# ---------------- autotest: esqueletos sinteticos con angulos conocidos ----------------
def _v(*x):
    return np.array(x, float)


def _mundo_prueba(v, esp):
    """Coordenadas de persona (r = derecha, u = arriba, f = adelante) -> mundo de prueba.
    Mundo normal: x = derecha de la imagen, y = abajo, z = hacia dentro de la imagen (mano derecha).
    esp=True refleja z (sistema de mano izquierda) para comprobar la correccion de quiralidad."""
    r, u, f = v
    return np.array([-r, -u, (f if esp else -f)])


def _mano_sintetica(w, h, rho, n, mcp, pip, dip):
    """21 puntos (coordenadas de persona) de una mano con ejes h (dedos), rho (radial), n (palma)."""
    H = [None] * 21
    H[0] = w
    base = {"pulgar": (0.03, 0.02), "indice": (0.025, 0.09), "medio": (0.0, 0.095),
            "anular": (-0.02, 0.088), "menique": (-0.04, 0.075)}
    largos = [None, 0.04, 0.025, 0.02]
    for d, idx in DEDOS_IDX.items():
        o_r, o_h = base[d]
        v0 = o_h * h + o_r * rho
        largos[0] = np.linalg.norm(v0)
        d0 = v0 / largos[0]
        c = [0, 0, mcp, mcp + pip] if d == "pulgar" else [0, mcp, mcp + pip, mcp + pip + dip]
        p = w.copy()
        for k in range(4):
            ang = math.radians(c[k])
            dk = d0 * math.cos(ang) + n * math.sin(ang)
            p = p + largos[k] * dk
            H[idx[k]] = p
    return H


def _sintetico(esp, brazos=None, manos=None, T=6):
    brazos = brazos or {}
    manos = manos or {}
    colgar = _v(0, -1, 0)
    hombros = {"der": _v(0.2, 0, 0), "izq": _v(-0.2, 0, 0)}
    pose = {0: _v(0, 0.25, 0.08), 11: hombros["izq"], 12: hombros["der"],
            23: _v(-0.1, -0.5, 0), 24: _v(0.1, -0.5, 0)}
    idx_codo, idx_mun = {"izq": 13, "der": 14}, {"izq": 15, "der": 16}
    munecas = {}
    for lado in ("izq", "der"):
        du, df_ = brazos.get(lado, (colgar, colgar))
        codo_p = hombros[lado] + 0.30 * du
        mun_p = codo_p + 0.25 * df_
        pose[idx_codo[lado]], pose[idx_mun[lado]] = codo_p, mun_p
        munecas[lado] = mun_p
    datos = {}
    for i, v in pose.items():
        m = _mundo_prueba(v, esp)
        for e, val in zip("xyz", m):
            datos[f"mundo_pose_{i}_{e}"] = np.full(T, val)
    for lado, m in manos.items():
        pref = "mundo_manoD" if lado == "der" else "mundo_manoI"
        H = _mano_sintetica(munecas[lado], m["h"], m["rho"], m["n"], m.get("mcp", 0), m.get("pip", 0), m.get("dip", 0))
        for i, v in enumerate(H):
            mm = _mundo_prueba(v, esp)
            for e, val in zip("xyz", mm):
                datos[f"{pref}_{i}_{e}"] = np.full(T, val)
    return pd.DataFrame(datos)


def _mano_muneca(lado, flex=0.0, radial=0.0, **dedos_):
    a, n_a = _v(0, -1, 0), _v(0, 0, 1)
    rd = _v(1, 0, 0) if lado == "der" else _v(-1, 0, 0)
    if flex:
        t = math.radians(flex)
        d = dict(h=a * math.cos(t) + n_a * math.sin(t), rho=rd, n=n_a * math.cos(t) - a * math.sin(t))
    else:
        p = math.radians(radial)
        d = dict(h=a * math.cos(p) + rd * math.sin(p), rho=rd * math.cos(p) - a * math.sin(p), n=n_a)
    d.update(dedos_)
    return d


def fis_autotest():
    """Comprueba que el codigo recupera angulos conocidos (derecha, izquierda y sistema de coordenadas espejado)."""
    g = math.radians
    colgar = _v(0, -1, 0)
    lat = {"der": _v(1, 0, 0), "izq": _v(-1, 0, 0)}
    adel = _v(0, 0, 1)
    casos = []
    for esp in (False, True):
        et = " [espejo]" if esp else ""
        casos.append((f"reposo{et}", esp, None, {"der": _mano_muneca("der"), "izq": _mano_muneca("izq")}, {
            "shoulder_right_elevation": 0, "shoulder_left_elevation": 0, "elbow_right_flexion": 0,
            "elbow_left_flexion": 0, "wrist_right_flexion": 0, "wrist_left_flexion": 0,
            "wrist_right_radial_deviation": 0, "wrist_left_radial_deviation": 0}))
        casos.append((f"abduccion 90 der{et}", esp, {"der": (lat["der"], lat["der"])}, None, {
            "shoulder_right_elevation": 90, "shoulder_right_plane": 0, "shoulder_right_abduction": 90,
            "shoulder_right_flexion": None, "elbow_right_flexion": 0}))
        casos.append((f"abduccion 90 izq{et}", esp, {"izq": (lat["izq"], lat["izq"])}, None, {
            "shoulder_left_elevation": 90, "shoulder_left_plane": 0, "shoulder_left_abduction": 90}))
        casos.append((f"flexion hombro 90 izq{et}", esp, {"izq": (adel, adel)}, None, {
            "shoulder_left_elevation": 90, "shoulder_left_plane": 90, "shoulder_left_flexion": 90,
            "shoulder_left_abduction": None}))
        d45 = _v(0, -math.cos(g(45)), math.sin(g(45)))
        casos.append((f"flexion hombro 45 der{et}", esp, {"der": (d45, d45)}, None, {
            "shoulder_right_elevation": 45, "shoulder_right_plane": 90, "shoulder_right_flexion": 45,
            "shoulder_right_abduction": 0}))
        e30 = _v(0, -math.cos(g(30)), -math.sin(g(30)))
        casos.append((f"extension hombro 30 der{et}", esp, {"der": (e30, e30)}, None, {
            "shoulder_right_flexion": -30, "shoulder_right_elevation": 30}))
        casos.append((f"codo 90 der{et}", esp, {"der": (colgar, adel)}, None, {
            "elbow_right_flexion": 90, "shoulder_right_elevation": 0}))
        for lado, L in (("der", "right"), ("izq", "left")):
            casos.append((f"muneca flexion +30 {lado}{et}", esp, None, {lado: _mano_muneca(lado, flex=30)},
                          {f"wrist_{L}_flexion": 30, f"wrist_{L}_radial_deviation": 0}))
            casos.append((f"muneca extension -25 {lado}{et}", esp, None, {lado: _mano_muneca(lado, flex=-25)},
                          {f"wrist_{L}_flexion": -25}))
            casos.append((f"desv. radial +20 {lado}{et}", esp, None, {lado: _mano_muneca(lado, radial=20)},
                          {f"wrist_{L}_radial_deviation": 20, f"wrist_{L}_flexion": 0}))
            casos.append((f"desv. cubital -15 {lado}{et}", esp, None, {lado: _mano_muneca(lado, radial=-15)},
                          {f"wrist_{L}_radial_deviation": -15}))
            casos.append((f"dedos 45/60/30 {lado}{et}", esp, None, {lado: _mano_muneca(lado, mcp=45, pip=60, dip=30)},
                          {f"{L}_index_mcp_flexion": 45, f"{L}_index_pip_flexion": 60, f"{L}_index_dip_flexion": 30,
                           f"{L}_little_pip_flexion": 60, f"{L}_thumb_mcp_flexion": 45}))
            casos.append((f"hiperextension MCP -15 {lado}{et}", esp, None,
                          {lado: _mano_muneca(lado, mcp=-15, pip=0, dip=0)},
                          {f"{L}_index_mcp_flexion": -15, f"{L}_middle_mcp_flexion": -15}))
    fallos = 0
    for nombre, esp, brazos, manos, esperado in casos:
        lm = _sintetico(esp, brazos, manos)
        df, chi = fis_calcular(lm, modo="vertical")
        df2, _ = fis_calcular(lm, modo="cadera")
        mal = []
        for col, val in esperado.items():
            if col not in df.columns:
                mal.append(f"{col}: no existe")
                continue
            for etiqueta, d in (("vertical", df), ("cadera", df2)):
                got = float(d[col].median())
                if val is None:
                    if np.isfinite(got):
                        mal.append(f"{col} [{etiqueta}]: esperaba NaN, dio {got:.1f}")
                elif not np.isfinite(got) or abs(got - val) > 0.5:
                    mal.append(f"{col} [{etiqueta}]: esperaba {val}, dio {got:.2f}")
        if (chi < 0) != esp:
            mal.append(f"quiralidad mal deducida (chi={chi})")
        print(f"  {'OK ' if not mal else 'FALLA'} {nombre}")
        for m in mal:
            print(f"        {m}")
        fallos += bool(mal)
    # control de calidad: un segmento con longitud absurda en un frame se descarta
    lm = _sintetico(False, {"der": (adel, adel)})
    lm.loc[2, "mundo_pose_14_z"] += 0.5
    df, _ = fis_calcular(lm)
    _, _, qc = _fis_calcular(lm)
    descartado = (bool(np.isnan(df["elbow_right_flexion"].iloc[2]))
                  and int(qc["qc_right_arm_segment_length_ok"].iloc[2]) == 0
                  and int(qc["qc_right_arm_segment_length_ok"].iloc[0]) == 1)
    print(f"  {'OK ' if descartado else 'FALLA'} frame con segmento de longitud inconsistente se descarta")
    fallos += (not descartado)
    print("\nAUTOTEST " + ("PASO: signos y formulas verificados." if not fallos else f"FALLO: {fallos} caso(s)."))
    return fallos


def calcular_fisiologicos(modo=None):
    """Angulos fisiologicos de todas las repeticiones de resultados/ -> angulos_fisiologicos.csv por repeticion y
    dataset/fisiologicos_(frames|resumen|definiciones).csv."""
    modo = modo or MODO_TRONCO
    idx = idx_pose_actual()
    reps = list(repeticiones_guardadas("landmarks.csv"))
    if not reps:
        print("No hay repeticiones en 'resultados/'.")
        return
    cols_ang = encabezados_angulos()[2:]
    frames, resumen, sin_mundo = [], [], []
    for carpeta, sena, persona, rep in reps:
        lm = pd.read_csv(os.path.join(carpeta, "landmarks.csv"))
        if not tiene_mundo(lm):
            sin_mundo.append(f"{sena}/{persona}_rep{rep}")
            continue
        df, chi, qc = _fis_calcular(lm, idx, modo)
        activo = np.zeros(len(df), dtype=int)
        i0, i1 = 0, len(df) - 1
        f_an = os.path.join(carpeta, "angulos.csv")
        if os.path.exists(f_an):
            try:
                an = pd.read_csv(f_an).reindex(columns=["frame", "tiempo_ms"] + cols_ang)
                res = analizar(lm, an, sena)
                i0, i1 = res["i0"], res["i1"]
                activo[i0:i1 + 1] = 1
            except ErrorAnalisis:
                activo[:] = 1
        else:
            activo[:] = 1
        salida = pd.DataFrame({"frame": lm["frame"], "tiempo_ms": lm["tiempo_ms"], "en_tramo_activo": activo})
        salida = pd.concat([salida, df, qc], axis=1)
        salida.to_csv(os.path.join(carpeta, "angulos_fisiologicos.csv"), index=False)
        ident = {"sena": sena, "persona": persona, "rep": rep}
        frames.append(pd.concat([pd.DataFrame(ident, index=salida.index), salida], axis=1))
        fila = {**ident, "plano": PLANO, "frames_tramo": i1 - i0 + 1, "tronco": modo,
                "orientacion_ejes": "mano derecha" if chi > 0 else "invertida (corregida)"}
        fila.update(resumen_rep(df, i0, i1))
        resumen.append(fila)
        malos = [c[:-len("_pct_fuera_rango")] for c in fila if c.endswith("_pct_fuera_rango")
                 and pd.notna(fila[c]) and fila[c] > AVISO_FUERA_RANGO]
        vacios = [c[:-len("_pct_sin_dato")] for c in fila if c.endswith("_pct_sin_dato")
                  and pd.notna(fila[c]) and fila[c] > AVISO_SIN_DATO]
        aviso = ""
        if malos:
            aviso += f"  | fuera de rango anatomico (> {AVISO_FUERA_RANGO:.0f}%): " + ", ".join(malos)
        if vacios:
            aviso += (f"  | sin dato o descartados (> {AVISO_SIN_DATO:.0f}%): " + ", ".join(vacios[:8])
                      + (" ..." if len(vacios) > 8 else ""))
        print(f"  {sena}/{persona}_rep{rep}: ok" + (("  AVISO" + aviso) if aviso else ""))
    if sin_mundo:
        print(f"\n{len(sin_mundo)} repeticion(es) sin coordenadas 3D (procesadas con una version antigua): "
              + ", ".join(sin_mundo[:6]) + (" ..." if len(sin_mundo) > 6 else "") + "\n  -> vuelve a analizar esos videos.")
    if not frames:
        return
    os.makedirs(CARPETA_DATASET, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(CARPETA_DATASET, "fisiologicos_frames.csv"), index=False)
    pd.DataFrame(resumen).to_csv(os.path.join(CARPETA_DATASET, "fisiologicos_resumen.csv"), index=False)
    fis_definiciones().to_csv(os.path.join(CARPETA_DATASET, "fisiologicos_definiciones.csv"), index=False)
    print(f"\nListo: {len(resumen)} repeticiones. Archivos en '{CARPETA_DATASET}': fisiologicos_frames.csv, "
          "fisiologicos_resumen.csv, fisiologicos_definiciones.csv")


# ======================================================================================================================
# 13. DIAGNOSTICO DE LOS ANGULOS FISIOLOGICOS (solo lee; necesita haber corrido 'fisiologicos')
# ======================================================================================================================
_META_FIS = {"sena", "persona", "rep", "frame", "tiempo_ms", "en_tramo_activo", "trunk_axis", "coord_chirality"}
_SUFIJOS_FIS = ("_elevation", "_plane", "_flexion", "_abduction", "_radial_deviation")


def _columnas_angulo(df):
    return [c for c in df.columns if c not in _META_FIS and not c.startswith("qc_") and c.endswith(_SUFIJOS_FIS)]


def _seccion(titulo):
    print(f"\n{'=' * 78}\n{titulo}\n{'=' * 78}")


def _diag_fuera_rango(act):
    _seccion("1. ANGULOS FUERA DE RANGO ANATOMICO (> 10 % de los frames validos del tramo activo)")
    filas = []
    for c in _columnas_angulo(act):
        s = act[c].dropna()
        if len(s) < 10:
            continue
        lo, hi = rango_de(c)
        bajo, alto = 100 * (s < lo).mean(), 100 * (s > hi).mean()
        if bajo + alto > 10:
            filas.append({"angulo": c, "rango": f"({lo},{hi})", "n": len(s), "% bajo": round(bajo, 1),
                          "% alto": round(alto, 1), "mediana": round(s.median(), 1),
                          "p5": round(s.quantile(.05), 1), "p95": round(s.quantile(.95), 1)})
    if not filas:
        print("  Ninguno supera el 10 %.")
        return
    print(pd.DataFrame(filas).sort_values("% bajo", ascending=False).to_string(index=False))
    print("\n  Lectura: '% bajo' alto = valores por debajo del rango (en dedos suele ser hiperextension aparente por\n"
          "  ruido del signo cerca de 0); '% alto' alto = flexion o desviacion exagerada (sesgo o error de profundidad).")


def _diag_muneca(act):
    _seccion("2. MUNECA POR SENA (mediana [p5, p95], solo frames validos)")
    filas = []
    for lado in ("right", "left"):
        for ang in ("flexion", "radial_deviation"):
            c = f"wrist_{lado}_{ang}"
            if c not in act.columns:
                continue
            for sena, g in act.groupby("sena"):
                s = g[c].dropna()
                if len(s) >= 10:
                    filas.append({"angulo": c, "sena": sena, "n": len(s), "mediana [p5, p95]":
                                  f"{s.median():6.1f} [{s.quantile(.05):6.1f}, {s.quantile(.95):6.1f}]"})
    if filas:
        print(pd.DataFrame(filas).to_string(index=False))
    print("\n  Lectura: si la mediana cambia mucho entre senas con la misma mano, el desfase depende de la orientacion\n"
          "  del antebrazo respecto a la camara (limite de combinar dos modelos). Una validacion con muneca neutra\n"
          "  real (0 grados) dira cuanto vale ese desfase.")


def _diag_causas(act):
    _seccion("3. POR QUE FALTAN DATOS EN LOS DEDOS (% de frames del tramo activo)")
    filas = []
    for lado in ("right", "left"):
        pres = f"qc_{lado}_hand_landmarks_present"
        if pres not in act.columns:
            continue
        faltan_mano = 100 * (act[pres] == 0).mean()
        con_mano = act[act[pres] == 1]
        for d in DEDOS_IDX:
            ok = f"qc_{lado}_{DEDO_EN[d]}_segment_length_ok"
            if ok not in act.columns:
                continue
            filas.append({"mano": lado, "dedo": DEDO_EN[d], "% mano no detectada": round(faltan_mano, 1),
                          "% longitud inconsistente": round(100 * (con_mano[ok] == 0).mean(), 1)
                          if len(con_mano) else np.nan})
    if filas:
        print(pd.DataFrame(filas).to_string(index=False))
    for lado in ("right", "left"):
        c = f"qc_{lado}_arm_segment_length_ok"
        if c in act.columns:
            print(f"  brazo {lado}: {100 * (act[c] == 0).mean():.1f} % con longitud de humero/antebrazo inconsistente")
    print("\n  Lectura: 'longitud inconsistente' alto con la mano bien detectada = la tolerancia es demasiado estricta\n"
          "  para los landmarks de la mano (no conservan la longitud de los huesos); ver seccion 4.")


def _desv_rel(a, b):
    L = np.linalg.norm(a - b, axis=1)
    m = np.nanmedian(L)
    return np.abs(L / m - 1.0) if np.isfinite(m) and m > 0 else np.full(len(L), np.nan)


def _diag_sensibilidad(tols):
    _seccion("4. SENSIBILIDAD A LA TOLERANCIA DE LONGITUD (% de frames del tramo activo que se descartarian)")
    acum = {}
    idx = idx_pose_actual()
    for carpeta, _, _, _ in repeticiones_guardadas("landmarks.csv"):
        f_ang = os.path.join(carpeta, "angulos_fisiologicos.csv")
        if not os.path.exists(f_ang):
            continue
        lm = pd.read_csv(os.path.join(carpeta, "landmarks.csv"))
        act = pd.read_csv(f_ang)["en_tramo_activo"].to_numpy() == 1
        if not tiene_mundo(lm) or len(act) != len(lm):
            continue
        P = {n_: pts3d(lm, "mundo_pose", i) for n_, i in idx.items()}
        for lado in ("izq", "der"):
            L = LADO_EN[lado]
            d = np.maximum(_desv_rel(P[f"hombro_{lado}"], P[f"codo_{lado}"]),
                           _desv_rel(P[f"codo_{lado}"], P[f"muneca_{lado}"]))
            acum.setdefault((L, "arm"), []).append(d[act])
            pref = "mundo_manoI" if lado == "izq" else "mundo_manoD"
            if f"{pref}_0_x" not in lm.columns:
                continue
            H = [pts3d(lm, pref, i) for i in range(21)]
            for dedo, ids in DEDOS_IDX.items():
                cad = [0] + ids
                m = np.vstack([_desv_rel(H[cad[k + 1]], H[cad[k]]) for k in range(4)])
                acum.setdefault((L, DEDO_EN[dedo]), []).append(np.max(m, axis=0)[act])
    if not acum:
        print("  No se encontraron landmarks con coordenadas 3D y angulos_fisiologicos.csv.")
        return
    filas = []
    for (lado, parte), lista in sorted(acum.items()):
        d = np.concatenate(lista)
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        fila = {"mano/brazo": lado, "parte": parte, "n": len(d), "mediana desv.": round(float(np.median(d)), 2),
                "p90 desv.": round(float(np.percentile(d, 90)), 2)}
        for t in tols:
            fila[f"% > {t:.0%}"] = round(100 * float((d > t).mean()), 1)
        filas.append(fila)
    print(pd.DataFrame(filas).to_string(index=False))
    print(f"\n  Tolerancias actuales: brazo {TOL_LONG_BRAZO:.0%}, dedos {TOL_LONG_DEDO:.0%}.\n"
          "  desv. = desviacion relativa del segmento mas alterado respecto a su mediana en la repeticion.\n"
          "  Si a 30 % se descarta mucho pero a 40-50 % casi nada, la tolerancia esta cortando ruido normal del\n"
          "  modelo y conviene subirla (y reportar el valor elegido en el articulo).")


def diagnostico_fisiologicos(tols=(0.2, 0.3, 0.4, 0.5)):
    ruta = os.path.join(CARPETA_DATASET, "fisiologicos_frames.csv")
    if not os.path.exists(ruta):
        raise SystemExit("Falta dataset/fisiologicos_frames.csv: corre primero  py -3.12 sistema_lsc.py fisiologicos")
    frames = pd.read_csv(ruta)
    act = frames[frames["en_tramo_activo"] == 1]
    print(f"{frames[['sena', 'persona', 'rep']].drop_duplicates().shape[0]} repeticiones, "
          f"{len(act)} frames en tramo activo (de {len(frames)}).")
    _diag_fuera_rango(act)
    _diag_muneca(act)
    _diag_causas(act)
    _diag_sensibilidad(tols)


# ======================================================================================================================
# 14. PIPELINE DE ANALISIS (un video y lotes)
# ======================================================================================================================
def procesar_video(video, sena, persona, calib=None, blur=True, detector=None):
    """Analiza UNA repeticion: landmarks y angulos -> calibracion -> control de calidad -> fases -> figuras -> dedos ->
    variables estandarizadas -> resumen (fases.xlsx) -> documento bilingue. Crea una repeticion NUEVA (rep1, rep2...),
    nunca sobrescribe. No reconstruye los datasets (eso va al final del lote)."""
    if not os.path.exists(video):
        raise ErrorAnalisis(f"No existe el video: {video}")

    # 1) video -> landmarks y angulos (la carpeta se crea despues: un fallo aqui no deja repeticiones vacias)
    lm, an_crudo, fps, info_ext = extraer(video, detector)
    rep = siguiente_rep(sena, persona)
    carpeta = carpeta_rep(sena, persona, rep)
    os.makedirs(carpeta, exist_ok=True)
    print(f"Repeticion: {sena} - {persona} - rep{rep}\nCarpeta: {carpeta}")

    ruta_calib = resolver_calibracion(persona, calib)
    an, calibrado = aplicar_calibracion(an_crudo, ruta_calib)
    guardar_datos(lm, "Landmarks", leyenda_landmarks(), os.path.join(carpeta, "landmarks.xlsx"))
    guardar_datos(an, "Angulos", leyenda_angulos(calibrado), os.path.join(carpeta, "angulos.xlsx"), escala_color=True)
    lm.to_csv(os.path.join(carpeta, "landmarks.csv"), index=False)
    an.to_csv(os.path.join(carpeta, "angulos.csv"), index=False)
    print("Guardado: landmarks y angulos (csv y xlsx)")

    # 2) control de calidad por frame
    qc = calidad_por_frame(lm, an)
    for lado in ("der", "izq"):
        print(f"  calidad mano {LADO_ES[lado]}: {100 * qc[f'qc_mano_{lado}_valida'].mean():.0f}% de frames validos "
              f"(presente {100 * qc[f'qc_mano_{lado}_presente'].mean():.0f}%, solapadas "
              f"{100 * qc['qc_manos_solapadas'].mean():.0f}%)")

    # 3) fases del gesto
    try:
        res = analizar(lm, an, sena)
    except ErrorAnalisis as e:
        print(f"\nNo se pudo analizar el gesto: {e}")
        print("Los datos crudos ya se guardaron. Revisa el video (mano visible, cintura para arriba) y repite.")
        try:
            construir_variables(lm, an, None, {}, qc).to_csv(os.path.join(carpeta, "variables.csv"), index=False)
        except Exception:
            traceback.print_exc()
        return carpeta
    fases_en, fases_es = describir_fases(res, "en"), describir_fases(res, "es")
    print(f"Fases del gesto: {len(res['runs'])} ("
          + "-".join("M" if m else "S" for _, _, m in res["runs"]) + ")")
    for aviso in res.get("avisos", []):
        print(f"  AVISO: {aviso}")
    imgs, tiempos = fotogramas_cuerpo_hd(video, lm, res, blur=blur)

    filas = []
    t_vid = res["t"]
    ini_vid = [t_vid[_idx_fase(a_, b_, res["i0"], res["i1"])[0]] for a_, b_, _ in res["runs"]]
    fin_vid = ini_vid[1:] + [t_vid[res["i1"]]]
    for j, ((a_, b_, mov), f_en, f_es) in enumerate(zip(res["runs"], fases_en, fases_es)):
        t0, t1 = f_en[3], f_en[4]
        fila = {"Fase": f"F{j + 1}", "Tipo": "Movimiento" if mov else "Sostenimiento",
                "Descripcion (ES)": f_es[0], "Que pasa (ES)": f_es[1],
                "Description (EN)": f_en[0], "What happens (EN)": f_en[1],
                "t inicio (s)": round(t0, 2), "t fin (s)": round(t1, 2),
                "Duracion (s)": round(t1 - t0, 2),
                "t inicio video (s)": round(float(ini_vid[j]), 3), "t fin video (s)": round(float(fin_vid[j]), 3),
                "Fotograma (t en video, s)": round(tiempos[j], 2)}
        for k, (nombre, dec) in SENALES.items():
            s_ = res["perfil"][k]
            fila[f"{nombre} - inicio"] = round(float(s_[a_]), dec + 1) if np.isfinite(s_[a_]) else np.nan
            fila[f"{nombre} - fin"] = round(float(s_[b_ - 1]), dec + 1) if np.isfinite(s_[b_ - 1]) else np.nan
            fila[f"{nombre} - media"] = (round(float(np.nanmean(s_[a_:b_])), dec + 1)
                                         if np.isfinite(s_[a_:b_]).any() else np.nan)
        filas.append(fila)
    perfil_df = pd.DataFrame({"tiempo_s": np.linspace(0, res["dur"], N),
                              "Velocidad muneca otra (rel.)": res["v_otro"],
                              **{SENALES[k][0]: res["perfil"][k] for k in SENALES}})
    info = pd.DataFrame([{"Video": os.path.basename(video), "Sena": sena, "Persona": persona,
                          "Repeticion": rep, "Brazo dominante": res["lado"],
                          "FPS nominal": round(fps, 2), "FPS efectivo": round(info_ext["fps_efectivo"], 2),
                          "Fuente del tiempo": info_ext["fuente_tiempo"], "Resolucion": info_ext["resolucion"],
                          "Frames": len(lm), "Duracion tramo activo (s)": round(res["dur"], 2),
                          "Inicio del gesto en el video (s)": round(float(res["t"][res["i0"]]), 3),
                          "Numero de fases": len(res["runs"]),
                          "Avisos de fases": "; ".join(res.get("avisos", [])),
                          "Umbrales fases entra / sale": f"{UMBRAL_MOV_ENTRA:g} / {UMBRAL_MOV_SALE:g} (fraccion p95)",
                          "Min. movimiento / pausa (s)": f"{MIN_MOV_S:g} / {MIN_SOST_S:g}",
                          "% frames con muneca detectada": round(res["valido"], 0),
                          "Angulos calibrados": "si" if calibrado else "no",
                          "Archivo de calibracion": os.path.basename(ruta_calib) if calibrado else "",
                          "Detector MediaPipe": info_ext["detector"], "Version MediaPipe": info_ext["mediapipe"],
                          "Preproceso de luz": "si" if PREPROCESO_LUZ else "no",
                          "Filtro cuerpo / dedos / posicion (Hz)": f"{FC_CUERPO_HZ:g} / {FC_DEDOS_HZ:g} / "
                                                                   f"{FC_POSICION_HZ:g}",
                          "Cara difuminada en figura": "si" if blur else "no"}])
    graficar(res, fases_en, imgs, sena, persona, rep, os.path.join(carpeta, "fases.png"), calibrado)
    print("Guardado: fases.png")

    # 4) dedos y cinematica (figuras cuerpo.png y dedos_*.png con los frames rechazados en gris)
    fases_lados = analisis_dedos_y_cinematica(video, lm, an, res, imgs, sena, persona, rep, carpeta, calibrado,
                                              ruta_calib, blur, qc)

    # 5) variables estandarizadas, eventos, estadisticas, sincronia y figura espacial
    hojas = {"Fases": pd.DataFrame(filas), "Perfil": perfil_df, "Info": info}
    hay_espacial = False
    try:
        var = construir_variables(lm, an, res, fases_lados, qc)
        var.to_csv(os.path.join(carpeta, "variables.csv"), index=False)
        stats = estadisticas_fases(var, res)
        hojas["Eventos"] = eventos_gesto(var, res)
        hojas["Estadisticas"] = stats
        hojas["Sincronia"] = sincronia(var, res, fases_lados)
        hojas["Calidad"] = resumen_calidad(qc, res)
        hojas["Definiciones variables"] = definiciones_variables(var.columns)
        figura_espacial(os.path.join(carpeta, "manos_espacial.png"), var, res, stats, sena, persona, rep,
                        nariz_rel(lm))
        hay_espacial = True
        print("Guardado: variables.csv y manos_espacial.png")
    except Exception as e:
        traceback.print_exc()
        print(f"AVISO: no se pudieron calcular las variables estandarizadas ({e}).")
    guardar_excel(os.path.join(carpeta, "fases.xlsx"), hojas, max_ancho=50)
    print("Guardado: fases.xlsx (Fases, Perfil, Info, Eventos, Estadisticas, Sincronia, Calidad, Definiciones)")

    # 6) documento con los textos de las figuras en espanol e ingles
    try:
        figuras = [("fases.png", TITULOS["gesto"]["es"].format(s=sena),
                    pie_gesto(res, fases_es, sena, persona, rep, calibrado, "es"),
                    TITULOS["gesto"]["en"].format(s=sena), pie_gesto(res, fases_en, sena, persona, rep, calibrado)),
                   ("cuerpo.png", TITULOS["cuerpo"]["es"].format(s=sena),
                    pie_cuerpo(res, fases_es, sena, persona, rep, calibrado, "es"),
                    TITULOS["cuerpo"]["en"].format(s=sena), pie_cuerpo(res, fases_en, sena, persona, rep, calibrado))]
        for lado, fases in (fases_lados or {}).items():
            figuras.append((f"dedos_{lado}.png", TITULOS["mano"]["es"].format(s=sena, l=LADO_ES[lado]),
                            pie_mano(res, lado, fases, sena, persona, rep, calibrado, "es"),
                            TITULOS["mano"]["en"].format(s=sena, l=LADO_TXT[lado]),
                            pie_mano(res, lado, fases, sena, persona, rep, calibrado)))
        if hay_espacial:
            figuras.append(("manos_espacial.png", TITULOS["espacial"]["es"].format(s=sena),
                            pie_espacial(res, sena, persona, rep, "es"), TITULOS["espacial"]["en"].format(s=sena),
                            pie_espacial(res, sena, persona, rep)))
        encabezado = (f"Seña / Sign: {sena} | Persona / Participant: {persona} | Repetición / Repetition: {rep}\n"
                      f"Video: {os.path.basename(video)} | Calibración / Calibration: "
                      f"{os.path.basename(ruta_calib) if calibrado else 'sin calibrar / uncalibrated'} | "
                      f"Detector: {info_ext['detector']} (MediaPipe {info_ext['mediapipe']})")
        escribir_documento_figuras(os.path.join(carpeta, "descripcion_figuras.txt"), encabezado, figuras)
        print("Guardado: descripcion_figuras.txt (textos de las figuras en espanol e ingles)")
    except Exception as e:
        traceback.print_exc()
        print(f"AVISO: no se pudo escribir el documento de las figuras ({e}).")
    return carpeta


def personas_conocidas():
    """Personas con archivo calibracion_<persona>.json en la carpeta del proyecto."""
    return [os.path.basename(f)[len("calibracion_"):-len(".json")].lower()
            for f in glob.glob(os.path.join(BASE, "calibracion_*.json"))
            if not os.path.basename(f).startswith("calibracion_respaldo_")]


def separar_nombre(stem, personas):
    """'agarrarnikol001' -> ('AGARRAR', 'Nikol'); 'accidente001' -> ('ACCIDENTE', None)."""
    base = re.sub(r"[\d_\-\s]+$", "", stem.lower())
    for p in sorted(personas, key=len, reverse=True):
        if p in base:
            sena = base.replace(p, "").strip("_- ")
            if sena:
                return sena.upper(), p.title()
    return base.upper(), None


def listar_videos(ruta):
    if os.path.isfile(ruta):
        return [ruta]
    if os.path.isdir(ruta):
        return sorted(os.path.join(d, f) for d, _, fs in os.walk(ruta) for f in fs
                      if os.path.splitext(f)[1].lower() in EXT_VIDEO)
    raise SystemExit(f"No existe la ruta: {ruta}")


def analizar_lote(ruta, sena=None, persona=None, calib=None, blur=True, con_dataset=True, con_fisio=True,
                  tronco=None, detector=None):
    """Un video o una carpeta. La sena y la persona salen del nombre del archivo si no se indican:
        agarrarnikol.mp4 -> AGARRAR, Nikol (si existe calibracion_nikol.json);  accidente001.mp4 -> ACCIDENTE.
    Al final (una sola vez) reconstruye los datasets y calcula los angulos fisiologicos 3D."""
    rutas = ruta if isinstance(ruta, (list, tuple)) else [ruta]          # una ruta o varias (modo interactivo)
    videos = [v for r in rutas for v in listar_videos(os.path.abspath(r))]
    if not videos:
        raise SystemExit(f"No encontre videos ({' '.join(EXT_VIDEO)}) en esa ruta.")
    personas = personas_conocidas()
    print(f"{len(videos)} video(s) a procesar.\n")
    ok, fallidos = 0, []
    for k, video in enumerate(videos, 1):
        s_auto, p_auto = separar_nombre(os.path.splitext(os.path.basename(video))[0], personas)
        s = (sena or s_auto).upper()
        p = (persona or p_auto or "Anonimo").strip().title()
        print(f"{'=' * 70}\n[{k}/{len(videos)}] {os.path.basename(video)}  ->  sena={s}, persona={p}\n{'=' * 70}")
        try:
            procesar_video(video, s, p, calib=calib, blur=blur, detector=detector)
            ok += 1
        except (ErrorAnalisis, RuntimeError) as e:
            print(f"  ERROR: {e}")
            fallidos.append(os.path.basename(video))
        except Exception:
            traceback.print_exc()
            fallidos.append(os.path.basename(video))

    if con_dataset and ok:
        print(f"\n{'=' * 70}\nConstruyendo datasets\n{'=' * 70}")
        construir_datasets()
    if con_fisio and ok:
        print(f"\n{'=' * 70}\nAngulos fisiologicos 3D\n{'=' * 70}")
        try:
            calcular_fisiologicos(tronco)
        except Exception:
            traceback.print_exc()
    print(f"\nListo: {ok} de {len(videos)} video(s) procesados.")
    if fallidos:
        print("Con error: " + ", ".join(fallidos))


def reorganizar():
    """Copia capturas_senas/ y caracterizacion/ a la estructura resultados/SENA/Persona_repN/ (no borra nada)."""
    copiados = 0

    def copiar(origen, destino):
        nonlocal copiados
        if os.path.exists(destino):
            return
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        shutil.copy2(origen, destino)
        copiados += 1

    # landmarks y angulos viejos: capturas_senas/SENA/Persona_repN_(landmarks|angulos).xlsx
    for ruta in glob.glob(os.path.join(CARPETA_VIEJA_CAPT, "*", "*.xlsx")):
        sena = os.path.basename(os.path.dirname(ruta)).upper()
        m = re.match(r"(.+)_rep(\d+)_(landmarks|angulos)\.xlsx$", os.path.basename(ruta))
        if not m:
            continue
        persona, rep, tipo = m.group(1).strip().title(), int(m.group(2)), m.group(3)
        dest = carpeta_rep(sena, persona, rep)
        copiar(ruta, os.path.join(dest, f"{tipo}.xlsx"))
        csv_ = os.path.join(dest, f"{tipo}.csv")
        if not os.path.exists(csv_):
            pd.read_excel(ruta, sheet_name=0).to_csv(csv_, index=False)

    # fases viejas: caracterizacion/SENA_Persona_repN_fases.(png|xlsx)   (la voz .wav se descarta)
    for ruta in glob.glob(os.path.join(CARPETA_VIEJA_CARAC, "*_fases.*")):
        m = re.match(r"([^_]+)_(.+)_rep(\d+)_fases\.(png|xlsx)$", os.path.basename(ruta))
        if not m:
            continue
        sena, persona, rep, ext = m.group(1).upper(), m.group(2).strip().title(), int(m.group(3)), m.group(4)
        copiar(ruta, os.path.join(carpeta_rep(sena, persona, rep), f"fases.{ext}"))
    print(f"Reorganizacion: {copiados} archivos copiados a '{CARPETA_RESULTADOS}' (los originales siguen donde estaban).")


# ======================================================================================================================
# 15. CAPTURA EN TIEMPO REAL (antes completo.py)
# ======================================================================================================================
# Camara en vivo (o videos con U / carpetas con F) con PoseLandmarker + HandLandmarker + FaceLandmarker en paralelo.
# Cada toma se guarda en dataset/<sena>/<sena>_FECHA.npz con:
#   x (N_FRAMES x 183) remuestreado y x_raw/t_raw (todos los frames), capturas JPEG, cara (61 col.), contactos mano-
#   cuerpo (29 col.), profundidad (10 col.) y vector ampliado "fase C" (61 col.); y CSV en resultados/<sena>/.
# Vector de 183 columnas, normalizado por los hombros (origen = punto medio, escala = ancho de hombros):
#   9 puntos de pose x,y (18) + 9 visibilidades + 21 puntos x,y,z de cada mano (126) + 2 banderas de mano +
#   14 angulos de dedos por mano / 180 (28). Lo que no se ve va en 0 con su bandera en 0.
# Teclas:
#   ESC salir | ESPACIO cuenta atras + grabar (otro ESPACIO cancela; durante un video lo cancela)
#   U subir videos | F carpeta de videos (subcarpeta = sena) | L cambiar de sena | N sena nueva | Z deshacer
#   T tiempos por etapa | C cara on/off | E panel fase C | V paneles | P cuerpo | A angulos dibujados
#   J articulacion de dedos (MCP/PIP/DIP) | S captura PNG + fila en capturas/mediciones.csv
# La grafica de fases de cada toma necesita el modulo opcional graficar_secuencia.py en la misma carpeta; sin el se
# guardan igual el .npz y todos los CSV.
INVERTIR_ETIQUETA = True   # etiqueta del modelo de mano (solo se usa si falla la asignacion por pose)
ALPHA = 0.4                # suavizado exponencial de los angulos en pantalla: 1.0 = sin suavizar, 0.2 = muy suave
ANCHO, ALTO = 640, 480
FPS_CAMARA = 30
CAMARA_MJPG = True         # MJPG suele dar mas FPS que YUY2 en webcams USB
CAMARA_DSHOW = True        # Windows: backend DirectShow
PARALELO = True            # manos, pose y cara en hilos a la vez (MediaPipe libera el GIL)
POSE_CADA_N = 1            # 1 = pose en todos los frames; 2 = uno de cada dos (solo camara en vivo)
CARA_ACTIVA = True
CARA_DIBUJAR = True
CONTACTO_DIST = 0.35       # anchos de hombro por debajo de los cuales la mano "toca"
CONTACTO_MAX = 3.0         # distancia tope; tambien el valor cuando falta mano o punto
PROF_TOL_M = 0.25          # (respaldo con la pose) metros que la muneca puede estar delante y contar como contacto
PROF_TOL_REL = 0.12        # r puede ser hasta 12 % menor que 1 (mano algo delante de la cara)
PROF_TOL_REL_ATRAS = 0.30  # r puede ser hasta 30 % mayor que 1 (mano detras de la nariz, p. ej. en la oreja)
CUERPO_ATRAS_M = 0.08      # el pecho y los hombros estan ~8 cm mas lejos de la camara que la cara
EJOS_M = 0.09              # distancia real media entre las comisuras externas de los ojos (m)
FOV_VERTICAL_GRADOS = 63.0 # campo de vision vertical supuesto (solo para pasar r a metros)
ANCHO_HOMBROS_M = 0.40     # ancho de hombros (m) si la pose no lo da
INDICE_CAMARA = 0
VIS_MIN = 0.5              # visibilidad minima de un punto del cuerpo
MARGEN_BORDE = 0.02        # puntos a menos de este margen del borde se descartan
PALMA_INVERTIDA = False    # True si "CAMARA" y "DORSO" salen al reves
PLANO_CODO = 0.70          # fraccion minima de cada segmento en el plano de la imagen para medir el codo en 2D
PLANO_MUNECA = 0.60        # igual, para la muneca (si no se cumple se muestra "--")

SENAS = ["hola", "gracias", "si", "no", "yo"]   # senas iniciales; con la tecla N se anaden mas
N_FRAMES = 30              # frames de cada secuencia guardada
DURACION_S = 10.0          # segundos de grabacion con la camara
CUENTA_ATRAS_S = 3.0
MIN_VALIDOS = 0.7          # fraccion minima de frames con los dos hombros visibles
ESCALA_MIN = 0.05          # hombros mas juntos que esto (fraccion del ancho) -> frame invalido
GUARDAR_CAPTURAS = True
ANCHO_CAPTURA, ALTO_CAPTURA = 320, 240
GENERAR_RESULTADOS = True
CSV_PARA_EXCEL_ES = True   # ';' y ',' decimal (Excel en espanol)
RECORTAR_TRAMO_ACTIVO = True
VIDEO_ESPEJADO = False     # True si el video subido ya esta en espejo
VIDEO_ALTO_MAX = 720

CARPETA_CAPTURAS = os.path.join(BASE, "capturas")
_GRAF = None               # modulo opcional graficar_secuencia (se carga al iniciar la captura)

CONEXIONES_MANO = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                   (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
VIVO_DEDOS = {"Indice": [5, 6, 7, 8], "Medio": [9, 10, 11, 12], "Anular": [13, 14, 15, 16], "Menique": [17, 18, 19, 20]}
PULGAR = [1, 2, 3, 4]

NARIZ = 0
OREJA_IZQ, OREJA_DER = 7, 8
HOMBRO_IZQ, HOMBRO_DER = 11, 12
CODO_IZQ, CODO_DER = 13, 14
MUNECA_IZQ, MUNECA_DER = 15, 16
CADERA_IZQ, CADERA_DER = 23, 24
BRAZO = {"Left": (HOMBRO_IZQ, CODO_IZQ, MUNECA_IZQ), "Right": (HOMBRO_DER, CODO_DER, MUNECA_DER)}
CONEXIONES_POSE = [(HOMBRO_IZQ, HOMBRO_DER), (HOMBRO_IZQ, CODO_IZQ), (CODO_IZQ, MUNECA_IZQ),
                   (HOMBRO_DER, CODO_DER), (CODO_DER, MUNECA_DER), (NARIZ, OREJA_IZQ), (NARIZ, OREJA_DER)]
PUNTOS_POSE = [NARIZ, OREJA_IZQ, OREJA_DER, HOMBRO_IZQ, HOMBRO_DER, CODO_IZQ, CODO_DER, MUNECA_IZQ, MUNECA_DER]
ORDEN_POSE = ["Codo izq", "Codo der", "Brazo izq", "Brazo der", "Cabeza lat"]
COLOR_LADO = {"Left": (0, 255, 0), "Right": (0, 165, 255)}  # BGR: verde / naranja
NOMBRE_LADO = {"Left": "IZQ", "Right": "DER"}
ANCHO_PANEL = 200
VIVO_CICLO_ART = ["apagado", "MCP", "PIP", "DIP"]   # ciclo de la tecla J
NOMBRES_MANO = [f"{d} {k}" for d in VIVO_DEDOS for k in ("MCP", "PIP", "DIP")] + ["Pulgar MCP", "Pulgar IP"]
CAMPOS_CAPTURA = (["archivo", "hora"] + ORDEN_POSE
                  + [f"{NOMBRE_LADO[l]} {c}" for l in ("Left", "Right") for c in NOMBRES_MANO + ["muneca", "palma", "dedos"]]
                  + ["valor_real", "nota"])


# ---------------- utilidades de angulos (vivo) ----------------
def angulo(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def flexion(a, b, c):
    """0 grados = extendido; mayor valor = mas flexionado."""
    return 180.0 - angulo(a, b, c)


def angulo_entre(u, v):
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def fraccion_plano(v):
    """Que parte de un vector 3D queda en el plano de la imagen (1 = de frente, 0 = apunta a la camara)."""
    return float(math.hypot(v[0], v[1]) / (np.linalg.norm(v) + 1e-9))


def suavizar(actual, previo):
    if previo is None:
        return dict(actual)
    return {k: ALPHA * v + (1 - ALPHA) * previo[k] if k in previo else v for k, v in actual.items()}


def vivo_angulos_mano(pts):
    """Flexion de cada articulacion (0 = recto) con los world landmarks de la mano (metros)."""
    res = {}
    for nombre, (mcp, pip, dip, tip) in VIVO_DEDOS.items():
        res[f"{nombre} MCP"] = flexion(pts[0], pts[mcp], pts[pip])
        res[f"{nombre} PIP"] = flexion(pts[mcp], pts[pip], pts[dip])
        res[f"{nombre} DIP"] = flexion(pts[pip], pts[dip], pts[tip])
    cmc, mcp, ip, tip = PULGAR
    res["Pulgar MCP"] = flexion(pts[cmc], pts[mcp], pts[ip])
    res["Pulgar IP"] = flexion(pts[mcp], pts[ip], pts[tip])
    return res


def mano_a_sin_espejo(pts, w, h):
    """Las manos se detectan sobre la imagen en espejo; esto las lleva al sistema de la pose (sin espejo):
    x, y en pixeles y z a la escala del ancho."""
    out = np.empty_like(pts)
    out[:, 0] = (1.0 - pts[:, 0]) * w
    out[:, 1] = pts[:, 1] * h
    out[:, 2] = pts[:, 2] * w
    return out


def visible_vivo(lm):
    """Punto utilizable: buena visibilidad y dentro del cuadro (no pegado al borde)."""
    v = getattr(lm, "visibility", None)
    if v is not None and v < VIS_MIN:
        return False
    return MARGEN_BORDE <= lm.x <= 1 - MARGEN_BORDE and MARGEN_BORDE <= lm.y <= 1 - MARGEN_BORDE


def angulo_muneca(pose_img, pose_mundo, lado, pts_sin_espejo, mano_mundo, w, h):
    """Desviacion 2D entre antebrazo (codo->muneca) y eje de la mano (muneca->nudillo medio). None si el antebrazo o
    la mano apuntan demasiado hacia la camara (ahi el 2D no es fiable)."""
    _, cod, mun = BRAZO[lado]
    if not (visible_vivo(pose_img[cod]) and visible_vivo(pose_img[mun])):
        return None
    ante3d = np.array([pose_mundo[mun].x - pose_mundo[cod].x, pose_mundo[mun].y - pose_mundo[cod].y,
                       pose_mundo[mun].z - pose_mundo[cod].z])
    mano3d = mano_mundo[9] - mano_mundo[0]
    if fraccion_plano(ante3d) < PLANO_MUNECA or fraccion_plano(mano3d) < PLANO_MUNECA:
        return None
    antebrazo = np.array([(pose_img[mun].x - pose_img[cod].x) * w, (pose_img[mun].y - pose_img[cod].y) * h])
    return angulo_entre(antebrazo, pts_sin_espejo[9, :2] - pts_sin_espejo[0, :2])


def orientacion_palma(pts_sin_espejo, lado):
    """CAMARA / DORSO / LADO, segun el giro del triangulo muneca-indice-menique."""
    v1 = pts_sin_espejo[5, :2] - pts_sin_espejo[0, :2]
    v2 = pts_sin_espejo[17, :2] - pts_sin_espejo[0, :2]
    cz = v1[0] * v2[1] - v1[1] * v2[0]
    seno = cz / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    if abs(seno) < 0.25:
        return "LADO"
    a_camara = (seno < 0) if lado == "Right" else (seno > 0)   # mano derecha con la palma a la camara -> cz < 0
    if PALMA_INVERTIDA:
        a_camara = not a_camara
    return "CAMARA" if a_camara else "DORSO"


def direccion_dedos(pts, w, h):
    """Hacia donde apuntan los dedos, tal como se ve en pantalla (imagen en espejo)."""
    dx = (pts[9][0] - pts[0][0]) * w
    dy = (pts[9][1] - pts[0][1]) * h
    if abs(dy) >= abs(dx):
        return "arriba" if dy < 0 else "abajo"
    return "derecha" if dx > 0 else "izquierda"


def angulo_linea(p, q):
    return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))


def normalizar_180(a):
    return (a + 180.0) % 360.0 - 180.0


def calcular_angulos_pose(img, mundo, w, h):
    """Codo en 2D si los segmentos estan de frente (PLANO_CODO), si no en 3D; brazo respecto a la vertical; cabeza
    lateral (orejas vs hombros)."""
    def P(i):
        return np.array([img[i].x * w, img[i].y * h], dtype=np.float32)

    def W(i):
        return np.array([mundo[i].x, mundo[i].y, mundo[i].z], dtype=np.float32)

    res = {}
    for lado, (hom, cod, mun) in BRAZO.items():
        nombre = "izq" if lado == "Left" else "der"
        if all(visible_vivo(img[i]) for i in (hom, cod, mun)):
            if fraccion_plano(W(hom) - W(cod)) >= PLANO_CODO and fraccion_plano(W(mun) - W(cod)) >= PLANO_CODO:
                res[f"Codo {nombre}"] = flexion(P(hom), P(cod), P(mun))   # 2D sobre la imagen
            else:
                res[f"Codo {nombre}"] = flexion(W(hom), W(cod), W(mun))   # 3D del modelo
        if visible_vivo(img[hom]) and visible_vivo(img[cod]):
            d = P(cod) - P(hom)
            res[f"Brazo {nombre}"] = math.degrees(math.atan2(abs(d[0]), d[1]))
    if all(visible_vivo(img[i]) for i in (OREJA_IZQ, OREJA_DER, HOMBRO_IZQ, HOMBRO_DER)):
        if (img[OREJA_IZQ].y + img[OREJA_DER].y) / 2 < (img[HOMBRO_IZQ].y + img[HOMBRO_DER].y) / 2:
            res["Cabeza lat"] = normalizar_180(angulo_linea(P(OREJA_DER), P(OREJA_IZQ))
                                               - angulo_linea(P(HOMBRO_DER), P(HOMBRO_IZQ)))
    return res


def asignar_lados(manos, pose_img, w, h):
    """manos: dicts con "pts" (normalizados, imagen en espejo) y "modelo". Devuelve [(lado, origen), ...]."""
    n = len(manos)
    if n == 0:
        return []
    munecas = {}
    if pose_img is not None:
        for lado, idx in (("Left", MUNECA_IZQ), ("Right", MUNECA_DER)):
            if visible_vivo(pose_img[idx]):
                munecas[lado] = np.array([pose_img[idx].x * w, pose_img[idx].y * h])

    def pos_mano(m):  # muneca de la mano, en el sistema sin espejo
        return np.array([(1.0 - m["pts"][0][0]) * w, m["pts"][0][1] * h])

    umbral = DIST_MAX_MUNECA * w
    if n == 1:
        if munecas:
            pm = pos_mano(manos[0])
            lado, d = min(((l, float(np.linalg.norm(pm - p))) for l, p in munecas.items()), key=lambda t: t[1])
            if d <= umbral:
                return [(lado, "pose")]
        return [(manos[0]["modelo"], "modelo")]

    pos = [pos_mano(manos[0]), pos_mano(manos[1])]
    if munecas:
        penal = float(max(w, h))

        def costo(a0, a1):
            c, ok = 0.0, True
            for lado, p in ((a0, pos[0]), (a1, pos[1])):
                if lado in munecas:
                    d = float(np.linalg.norm(p - munecas[lado]))
                    c += d
                    ok = ok and d <= umbral
                else:
                    c += penal
            return c, ok

        c_a, ok_a = costo("Left", "Right")
        c_b, ok_b = costo("Right", "Left")
        mejor, ok = (("Left", "Right"), ok_a) if c_a <= c_b else (("Right", "Left"), ok_b)
        if ok:
            return [(mejor[0], "pose"), (mejor[1], "pose")]
    # respaldo: posicion en pantalla (en espejo, la izquierda de la pantalla es tu izquierda)
    if manos[0]["pts"][0][0] <= manos[1]["pts"][0][0]:
        return [("Left", "posicion"), ("Right", "posicion")]
    return [("Right", "posicion"), ("Left", "posicion")]


# ---------------- dibujo y paneles (vivo) ----------------
def fondo(frame, x, y, ancho, alto, alpha=0.55):
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + ancho, frame.shape[1]), min(y + alto, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)


def texto(frame, s, x, y, color=(255, 255, 255), escala=0.42, grosor=1):
    cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def dibujar_mano(frame, pts, color):
    h, w = frame.shape[:2]
    px = [(int(p[0] * w), int(p[1] * h)) for p in pts]
    for a, b in CONEXIONES_MANO:
        cv2.line(frame, px[a], px[b], color, 2)
    for p in px:
        cv2.circle(frame, p, 3, (0, 0, 255), -1)


def dibujar_pose(frame, img):
    h, w = frame.shape[:2]

    def pt(i):
        return int((1.0 - img[i].x) * w), int(img[i].y * h)

    for a, b in CONEXIONES_POSE:
        if visible_vivo(img[a]) and visible_vivo(img[b]):
            cv2.line(frame, pt(a), pt(b), (255, 200, 0), 2)
    for i in PUNTOS_POSE:
        if visible_vivo(img[i]):
            cv2.circle(frame, pt(i), 5, (255, 0, 255), -1)
    if all(visible_vivo(img[i]) for i in (HOMBRO_IZQ, HOMBRO_DER, OREJA_IZQ, OREJA_DER)):
        c_h = ((pt(HOMBRO_IZQ)[0] + pt(HOMBRO_DER)[0]) // 2, (pt(HOMBRO_IZQ)[1] + pt(HOMBRO_DER)[1]) // 2)
        c_o = ((pt(OREJA_IZQ)[0] + pt(OREJA_DER)[0]) // 2, (pt(OREJA_IZQ)[1] + pt(OREJA_DER)[1]) // 2)
        cv2.line(frame, c_h, c_o, (0, 255, 255), 3)


def etiqueta_con_borde(frame, s, pos, color):
    cv2.putText(frame, s, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, s, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def linea_punteada(frame, p, q, color, paso=8):
    p, q = np.array(p, dtype=float), np.array(q, dtype=float)
    n = int(float(np.linalg.norm(q - p)) // paso)
    for i in range(0, n, 2):
        a = p + (q - p) * (i / max(n, 1))
        b = p + (q - p) * (min(i + 1, n) / max(n, 1))
        cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), color, 1, cv2.LINE_AA)


def dibujar_verificacion(frame, lms_pose, angulos_pose, datos_lado, ver_cuerpo, modo_art):
    """Escribe cada angulo medido en el punto del cuerpo/mano donde se mide."""
    h, w = frame.shape[:2]
    if ver_cuerpo and lms_pose is not None:
        def pt(i):
            return int((1.0 - lms_pose[i].x) * w), int(lms_pose[i].y * h)

        for lado, (hom, cod, _) in BRAZO.items():
            nombre = "izq" if lado == "Left" else "der"
            color = COLOR_LADO[lado]
            if f"Codo {nombre}" in angulos_pose and visible_vivo(lms_pose[cod]):
                px, py = pt(cod)
                etiqueta_con_borde(frame, f"codo {angulos_pose[f'Codo {nombre}']:.0f}", (px + 8, py - 8), color)
            if f"Brazo {nombre}" in angulos_pose and visible_vivo(lms_pose[hom]):
                hx, hy = pt(hom)
                linea_punteada(frame, (hx, hy), (hx, hy + 70), (255, 255, 255))   # referencia vertical
                etiqueta_con_borde(frame, f"brazo {angulos_pose[f'Brazo {nombre}']:.0f}",
                                   (max(2, hx + (8 if lado == "Right" else -78)), hy + 22), color)
        if "Cabeza lat" in angulos_pose:
            p_oi, p_od = pt(OREJA_IZQ), pt(OREJA_DER)
            cv2.line(frame, p_oi, p_od, (0, 255, 255), 1, cv2.LINE_AA)
            linea_punteada(frame, pt(HOMBRO_IZQ), pt(HOMBRO_DER), (255, 255, 255))
            mx, my = (p_oi[0] + p_od[0]) // 2, (p_oi[1] + p_od[1]) // 2
            etiqueta_con_borde(frame, f"cabeza {angulos_pose['Cabeza lat']:+.0f}", (mx + 10, my - 10), (0, 255, 255))
    if modo_art > 0:
        k = VIVO_CICLO_ART[modo_art]
        for lado, datos in datos_lado.items():
            if datos is None or "pts" not in datos:
                continue
            pts, ang = datos["pts"], datos["angulos"]
            color = COLOR_LADO[lado]
            for nombre, idxs in VIVO_DEDOS.items():
                j = idxs[("MCP", "PIP", "DIP").index(k)]
                px, py = int(pts[j][0] * w), int(pts[j][1] * h)
                cv2.circle(frame, (px, py), 6, color, 1, cv2.LINE_AA)
                etiqueta_con_borde(frame, f"{ang[f'{nombre} {k}']:.0f}", (px + 6, py - 6), color)
            if k in ("MCP", "DIP"):
                clave, j = ("Pulgar MCP", PULGAR[1]) if k == "MCP" else ("Pulgar IP", PULGAR[2])
                px, py = int(pts[j][0] * w), int(pts[j][1] * h)
                cv2.circle(frame, (px, py), 6, color, 1, cv2.LINE_AA)
                etiqueta_con_borde(frame, f"{ang[clave]:.0f}", (px + 6, py - 6), color)


def guardar_captura(frame, angulos_pose, datos_lado):
    """PNG + fila en capturas/mediciones.csv (para validar contra goniometro). Devuelve el nombre del PNG."""
    os.makedirs(CARPETA_CAPTURAS, exist_ok=True)
    ahora = datetime.now()
    archivo = f"captura_{ahora:%Y%m%d_%H%M%S}_{ahora.microsecond // 1000:03d}.png"
    cv2.imwrite(os.path.join(CARPETA_CAPTURAS, archivo), frame)
    fila = {"archivo": archivo, "hora": f"{ahora:%H:%M:%S}"}
    for nombre in ORDEN_POSE:
        if nombre in angulos_pose:
            fila[nombre] = f"{angulos_pose[nombre]:.1f}"
    for lado in ("Left", "Right"):
        datos = datos_lado.get(lado)
        if datos is None:
            continue
        pref = NOMBRE_LADO[lado]
        for n_ in NOMBRES_MANO:
            fila[f"{pref} {n_}"] = f"{datos['angulos'][n_]:.1f}"
        fila[f"{pref} muneca"] = "" if datos["muneca"] is None else f"{datos['muneca']:.1f}"
        fila[f"{pref} palma"] = datos["palma"]
        fila[f"{pref} dedos"] = datos["dedos"]
    ruta = os.path.join(CARPETA_CAPTURAS, "mediciones.csv")
    nuevo = not os.path.exists(ruta)
    with open(ruta, "a", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: Excel lo abre bien
        esc = csv.DictWriter(f, fieldnames=CAMPOS_CAPTURA, restval="")
        if nuevo:
            esc.writeheader()
        esc.writerow(fila)
    return archivo


def panel_mano(frame, lado, datos, x0, y0=8):
    color = COLOR_LADO[lado]
    titulo = f"MANO {NOMBRE_LADO[lado]}"
    fondo(frame, x0, y0, ANCHO_PANEL, 16 * 11 + 8)
    if datos is None:
        texto(frame, f"{titulo}  --", x0 + 6, y0 + 16, (150, 150, 150), 0.5, 1)
        return
    texto(frame, f"{titulo} {datos['conf']:.2f} ({datos['origen']})", x0 + 6, y0 + 16, color, 0.5, 1)
    y = y0 + 34
    for k, cx in (("MCP", 80), ("PIP", 122), ("DIP", 164)):
        texto(frame, k, x0 + cx, y, (160, 160, 160), 0.38)
    ang = datos["angulos"]
    for nombre in VIVO_DEDOS:
        y += 16
        texto(frame, nombre, x0 + 6, y)
        for k, cx in (("MCP", 80), ("PIP", 122), ("DIP", 164)):
            texto(frame, f"{ang[f'{nombre} {k}']:4.0f}", x0 + cx, y)
    y += 16
    texto(frame, f"Pulgar MCP {ang['Pulgar MCP']:4.0f}   IP {ang['Pulgar IP']:4.0f}", x0 + 6, y)
    y += 16
    m = datos["muneca"]
    texto(frame, "Muneca: " + (f"{m:4.0f}" if m is not None else "--"), x0 + 6, y,
          (255, 255, 255) if m is not None else (150, 150, 150))
    y += 16
    texto(frame, f"Palma: {datos['palma']}", x0 + 6, y)
    y += 16
    texto(frame, f"Dedos: {datos['dedos']}", x0 + 6, y)


def panel_pose(frame, angulos, x0, y0):
    fondo(frame, x0, y0, ANCHO_PANEL, 20 * (len(ORDEN_POSE) + 1) + 6)
    texto(frame, "CUERPO", x0 + 6, y0 + 16, (255, 200, 0), 0.5, 1)
    y = y0 + 16
    for nombre in ORDEN_POSE:
        y += 18
        if nombre in angulos:
            texto(frame, f"{nombre}:", x0 + 6, y)
            texto(frame, f"{angulos[nombre]:6.1f}", x0 + 100, y)
        else:
            texto(frame, f"{nombre}:", x0 + 6, y, (150, 150, 150))
            texto(frame, "--", x0 + 130, y, (150, 150, 150))


# ---------------- cara (fase A) ----------------
BLENDSHAPES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
PUNTOS_CARA = [("frente", 10), ("boca", 13), ("barbilla", 152)]     # indices del FaceLandmarker (478 puntos)
DIBUJO_CARA = {"cejas": [70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
               "ojos": [33, 133, 159, 145, 362, 263, 386, 374],
               "labios": [61, 291, 0, 17, 13, 14, 78, 308, 82, 87, 312, 317]}
COLOR_CARA = {"cejas": (0, 200, 255), "ojos": (255, 255, 0), "labios": (180, 105, 255)}


def nombres_columnas_cara():
    cols = ["cara_presente", "cabeza_yaw", "cabeza_pitch", "cabeza_roll"] + [f"bs_{n_}" for n_ in BLENDSHAPES]
    for n_, _ in PUNTOS_CARA:
        cols += [f"cara_{n_}_x", f"cara_{n_}_y"]
    return cols


N_COL_CARA = len(nombres_columnas_cara())
_BS0 = 4
_PT0 = _BS0 + len(BLENDSHAPES)


def angulos_cabeza(matriz):
    """(yaw, pitch, roll) en grados desde la matriz 4x4 de la cara. yaw + = gira hacia SU izquierda; pitch + = mira
    hacia abajo; roll + = inclina hacia SU hombro derecho."""
    R = np.array(matriz, dtype=np.float64).reshape(4, 4)[:3, :3]
    R = R / (np.linalg.norm(R, axis=0) + 1e-9)             # quita la escala
    sy = math.hypot(R[0, 0], R[1, 0])
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(-R[2, 0], sy))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    pitch = (pitch + 90.0) % 180.0 - 90.0                    # evita el salto de +-180 si la matriz sale invertida
    return yaw, pitch, roll


def construir_cara(res_cara, lms_pose, w, h):
    """Vector de N_COL_CARA valores de la cara del frame (todo en 0 si no hay cara)."""
    vec = np.zeros(N_COL_CARA, np.float32)
    if res_cara is None or not res_cara.face_landmarks:
        return vec
    vec[0] = 1.0
    if getattr(res_cara, "facial_transformation_matrixes", None):
        yaw, pitch, roll = angulos_cabeza(res_cara.facial_transformation_matrixes[0])
        vec[1:4] = [yaw / 180.0, pitch / 180.0, roll / 180.0]
    if getattr(res_cara, "face_blendshapes", None):
        d = {c.category_name: c.score for c in res_cara.face_blendshapes[0]}
        vec[_BS0:_PT0] = [d.get(n_, 0.0) for n_ in BLENDSHAPES]
    if lms_pose is not None and visible_vivo(lms_pose[HOMBRO_IZQ]) and visible_vivo(lms_pose[HOMBRO_DER]):
        P = lambda i: np.array([lms_pose[i].x * w, lms_pose[i].y * h], dtype=np.float64)   # noqa: E731
        origen = (P(HOMBRO_IZQ) + P(HOMBRO_DER)) / 2
        escala = float(np.linalg.norm(P(HOMBRO_IZQ) - P(HOMBRO_DER)))
        if escala >= ESCALA_MIN * w:
            lm = res_cara.face_landmarks[0]
            for k, (_, i) in enumerate(PUNTOS_CARA):
                vec[_PT0 + 2 * k:_PT0 + 2 * k + 2] = (np.array([lm[i].x * w, lm[i].y * h]) - origen) / escala
    return vec


def bs(vec, *nombres):
    """Promedio de uno o mas blendshapes del vector de cara."""
    return float(np.mean([vec[_BS0 + BLENDSHAPES.index(n_)] for n_ in nombres]))


def dibujar_cara(frame, lm):
    h, w = frame.shape[:2]
    for grupo, idx in DIBUJO_CARA.items():
        for i in idx:
            cv2.circle(frame, (int((1.0 - lm[i].x) * w), int(lm[i].y * h)), 2, COLOR_CARA[grupo], -1)


def panel_cara(frame, vec, x0, y0):
    fondo(frame, x0, y0, ANCHO_PANEL, 112)
    if vec is None or vec[0] < 0.5:
        texto(frame, "CARA --  (C apaga)", x0 + 6, y0 + 16, (160, 160, 160), 0.45, 1)
        return
    texto(frame, "CARA  (C apaga)", x0 + 6, y0 + 16, (0, 255, 255), 0.45, 1)
    filas = [("Cejas arriba", bs(vec, "browInnerUp", "browOuterUpLeft", "browOuterUpRight")),
             ("Ceno fruncido", bs(vec, "browDownLeft", "browDownRight")),
             ("Ojos cerrados", bs(vec, "eyeBlinkLeft", "eyeBlinkRight")),
             ("Boca abierta", bs(vec, "jawOpen")),
             ("Sonrisa", bs(vec, "mouthSmileLeft", "mouthSmileRight"))]
    for k, (nom, val) in enumerate(filas):
        texto(frame, nom, x0 + 6, y0 + 34 + 14 * k, (255, 255, 255), 0.4)
        texto(frame, f"{val:4.2f}", x0 + 130, y0 + 34 + 14 * k, (0, 255, 0), 0.4)
    yaw, pitch, roll = vec[1] * 180, vec[2] * 180, vec[3] * 180
    texto(frame, f"Cabeza y/p/r {yaw:4.0f}/{pitch:4.0f}/{roll:4.0f}", x0 + 6, y0 + 106, (255, 200, 0), 0.4)


def indices_remuestreo(tiempos, n):
    t = np.asarray(tiempos, dtype=np.float64)
    objetivo_t = np.linspace(t[0], t[-1], n)
    return np.abs(t[None, :] - objetivo_t[:, None]).argmin(axis=1)


def guardar_csv_tabla(ruta, tiempos, datos, columnas, excel_es=True):
    """Una fila por frame: frame, t_toma_s y las columnas dadas."""
    sep = ";" if excel_es else ","
    fmt = (lambda v: f"{float(v):.4f}".replace(".", ",")) if excel_es else (lambda v: f"{float(v):.4f}")
    with open(ruta, "w", newline="", encoding="utf-8-sig") as f:
        w_ = csv.writer(f, delimiter=sep)
        w_.writerow(["frame", "t_toma_s"] + list(columnas))
        for i, (t, fila) in enumerate(zip(tiempos, datos)):
            w_.writerow([i, fmt(t)] + [fmt(v) for v in fila])


# ---------------- vector por frame (183 columnas) ----------------
PUNTOS_VECTOR = [(NARIZ, "nariz"), (OREJA_IZQ, "oreja_izq"), (OREJA_DER, "oreja_der"),
                 (HOMBRO_IZQ, "hombro_izq"), (HOMBRO_DER, "hombro_der"), (CODO_IZQ, "codo_izq"),
                 (CODO_DER, "codo_der"), (MUNECA_IZQ, "muneca_izq"), (MUNECA_DER, "muneca_der")]
LADOS_VEC = (("Left", "IZQ"), ("Right", "DER"))
CAMPOS_INDICE = ["archivo", "sena", "fecha", "frames_capturados", "fps_medio", "validos_pct", "con_mano_pct"]


def nombres_columnas():
    cols = []
    for _, n_ in PUNTOS_VECTOR:
        cols += [f"pose_{n_}_x", f"pose_{n_}_y"]
    cols += [f"visible_{n_}" for _, n_ in PUNTOS_VECTOR]
    for _, pref in LADOS_VEC:
        for i in range(21):
            cols += [f"{pref}_p{i}_x", f"{pref}_p{i}_y", f"{pref}_p{i}_z"]
    cols += [f"mano_{pref}" for _, pref in LADOS_VEC]
    for _, pref in LADOS_VEC:
        cols += [f"ang_{pref}_{n_.replace(' ', '_')}" for n_ in NOMBRES_MANO]
    return cols


N_COLUMNAS = len(nombres_columnas())
_NP = len(PUNTOS_VECTOR)
COL_MASK_MANOS = slice(3 * _NP + 2 * 21 * 3, 3 * _NP + 2 * 21 * 3 + 2)
_IX_VEC = {}


def _ix_vec():
    if not _IX_VEC:
        _IX_VEC.update({n_: i for i, n_ in enumerate(nombres_columnas())})
    return _IX_VEC


def construir_vector(lms_pose, w, h, datos_lado):
    """Vector del frame (N_COLUMNAS) normalizado por los hombros, o None si no se ven los dos hombros."""
    if lms_pose is None or not (visible_vivo(lms_pose[HOMBRO_IZQ]) and visible_vivo(lms_pose[HOMBRO_DER])):
        return None

    def P(i):
        return np.array([lms_pose[i].x * w, lms_pose[i].y * h], dtype=np.float64)

    origen = (P(HOMBRO_IZQ) + P(HOMBRO_DER)) / 2
    escala = float(np.linalg.norm(P(HOMBRO_IZQ) - P(HOMBRO_DER)))
    if escala < ESCALA_MIN * w:
        return None
    pose = np.zeros((_NP, 2))
    vis = np.zeros(_NP)
    for k, (i, _) in enumerate(PUNTOS_VECTOR):
        if visible_vivo(lms_pose[i]):
            pose[k] = (P(i) - origen) / escala
            vis[k] = 1.0
    manos = np.zeros((2, 21, 3))
    presente = np.zeros(2)
    angs = np.zeros((2, len(NOMBRES_MANO)))
    for j, (lado, _) in enumerate(LADOS_VEC):
        d = datos_lado.get(lado)
        if d is None:
            continue
        pts = np.array(d["pts_se"], dtype=np.float64)   # x, y en px; z (relativa a la muneca) a escala del ancho
        pts[:, :2] -= origen
        pts /= escala
        manos[j] = pts
        presente[j] = 1.0
        angs[j] = [d["angulos"][n_] / 180.0 for n_ in NOMBRES_MANO]
    return np.concatenate([pose.ravel(), vis, manos.ravel(), presente, angs.ravel()]).astype(np.float32)


def vivo_remuestrear(tiempos, vectores, validos, n):
    """n frames equiespaciados en el tiempo (el mas cercano a cada instante; no mezcla frames)."""
    idx = indices_remuestreo(tiempos, n)
    return (np.stack([vectores[i] for i in idx]).astype(np.float32),
            np.array([validos[i] for i in idx], dtype=bool))


# ---------------- contactos mano-cuerpo (fase B) ----------------
OBJETIVOS_CONTACTO = ["nariz", "frente", "boca", "barbilla", "pecho", "hombro_izq", "hombro_der"]
REFS_MANO = [("palma", [0, 5, 9, 13, 17]), ("indice", [8])]
LADOS_CONTACTO = ("DER", "IZQ")
PECHO_Y = 0.35             # punto del pecho: 0.35 anchos de hombro bajo el punto medio de los hombros


def nombres_columnas_contacto():
    cols = []
    for lado in LADOS_CONTACTO:
        for ref, _ in REFS_MANO:
            cols += [f"dist_{lado}_{ref}_{o}" for o in OBJETIVOS_CONTACTO]
    cols.append("dist_palma_palma")
    return cols


N_COL_CONTACTO = len(nombres_columnas_contacto())


def nombres_columnas_prof():
    return ["adelante_cara_pose_DER", "adelante_cara_pose_IZQ", "adelante_cuerpo_pose_DER", "adelante_cuerpo_pose_IZQ",
            "ancho_hombros_m", "ratio_mano_cara_DER", "ratio_mano_cara_IZQ", "z_cabeza_m", "z_mano_DER_m",
            "z_mano_IZQ_m"]


N_COL_PROF = len(nombres_columnas_prof())
REF_Z_MAX_FRAMES = 45      # frames (~1,5 s) que se conserva la ultima z de la cara cuando la mano la tapa
_REF_Z = {"z": None, "n": 0, "n_vista": 0, "k": None, "fuente": "ninguna"}


def construir_prof(lms_pose, mundo_pose, datos_lado=None, res_cara=None, w=640, h=480):
    """N_COL_PROF valores (NaN si no hay dato):
      0-3 metros que la muneca DER/IZQ esta por delante de la nariz y de los hombros (profundidad de la pose);
      4 ancho de hombros (m); 5-6 r = Z_mano / Z_cara por tamano aparente (r < 1: mano mas cerca de la camara);
      7 distancia estimada cabeza-camara (m); 8-9 distancia estimada mano-camara (m)."""
    v = np.full(N_COL_PROF, np.nan, np.float32)
    if lms_pose is not None and mundo_pose is not None:
        z_cuerpo = z_nariz = None
        if visible_vivo(lms_pose[HOMBRO_IZQ]) and visible_vivo(lms_pose[HOMBRO_DER]):
            a, b = mundo_pose[HOMBRO_IZQ], mundo_pose[HOMBRO_DER]
            z_cuerpo = (a.z + b.z) / 2
            v[4] = math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
        if visible_vivo(lms_pose[NARIZ]):
            z_nariz = mundo_pose[NARIZ].z
        for k, idx in enumerate((MUNECA_DER, MUNECA_IZQ)):
            if visible_vivo(lms_pose[idx]):
                zm = mundo_pose[idx].z
                if z_nariz is not None:
                    v[k] = z_nariz - zm
                if z_cuerpo is not None:
                    v[2 + k] = z_cuerpo - zm
    # tamano aparente: referencia de profundidad de la cabeza = la cara si se ve; si la mano la tapa, la ultima z de
    # cara (hasta REF_Z_MAX_FRAMES frames); si no, los hombros calibrados con la cara cuando se veia
    f_px = (h / 2.0) / math.tan(math.radians(FOV_VERTICAL_GRADOS) / 2.0)
    z_hom = None
    if lms_pose is not None and visible_vivo(lms_pose[HOMBRO_IZQ]) and visible_vivo(lms_pose[HOMBRO_DER]):
        d_h = math.hypot((lms_pose[HOMBRO_IZQ].x - lms_pose[HOMBRO_DER].x) * w,
                         (lms_pose[HOMBRO_IZQ].y - lms_pose[HOMBRO_DER].y) * h)
        if d_h > 20:
            z_hom = f_px * ANCHO_HOMBROS_M / d_h
    _REF_Z["n"] += 1
    z_cab = None
    if res_cara is not None and res_cara.face_landmarks:
        lm = res_cara.face_landmarks[0]
        d_ojos = math.hypot((lm[33].x - lm[263].x) * w, (lm[33].y - lm[263].y) * h)
        cos_yaw = 1.0
        if getattr(res_cara, "facial_transformation_matrixes", None):
            cos_yaw = max(math.cos(math.radians(angulos_cabeza(res_cara.facial_transformation_matrixes[0])[0])), 0.7)
        if d_ojos > 5:
            z_cab = f_px * EJOS_M / (d_ojos / cos_yaw)
            _REF_Z.update(z=z_cab, n_vista=_REF_Z["n"], fuente="cara")
            if z_hom is not None:                                       # calibra hombros -> cara (promedio movil)
                k = z_cab / z_hom
                _REF_Z["k"] = k if _REF_Z["k"] is None else 0.9 * _REF_Z["k"] + 0.1 * k
    if z_cab is None:
        if _REF_Z["z"] is not None and _REF_Z["n"] - _REF_Z["n_vista"] <= REF_Z_MAX_FRAMES:
            z_cab, _REF_Z["fuente"] = _REF_Z["z"], "ultima"
        elif z_hom is not None and _REF_Z["k"] is not None:
            z_cab, _REF_Z["fuente"] = z_hom * _REF_Z["k"], "hombros"
        else:
            _REF_Z["fuente"] = "ninguna"
    if datos_lado is not None and z_cab is not None:
        s_cara = z_cab / f_px                                           # metros por pixel a la distancia de la cara
        v[7] = z_cab
        for k, lado in enumerate(("Right", "Left")):                    # DER, IZQ
            d = datos_lado.get(lado)
            if d is None or "mundo" not in d:
                continue
            pw = np.asarray(d["mundo"], dtype=np.float64)[:, :2]
            pp = np.asarray(d["pts_se"], dtype=np.float64)[:, :2]
            sw = float(np.sum((pw - pw.mean(axis=0)) ** 2))
            sp = float(np.sum((pp - pp.mean(axis=0)) ** 2))
            if sw > 0 and sp > 0:
                s_mano = math.sqrt(sw / sp)                             # metros por pixel a la distancia de la mano
                v[5 + k] = s_mano / s_cara
                v[8 + k] = f_px * s_mano
    return v


def calcular_contactos(x, cara=None, prof=None):
    """x: (T, N_COLUMNAS); cara: (T, N_COL_CARA) o None; prof: (T, N_COL_PROF) o None (sin profundidad: 2D).
    Devuelve (T, N_COL_CONTACTO): distancias en anchos de hombro (tope CONTACTO_MAX donde falte mano o punto)."""
    ix = _ix_vec()
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    T = len(x)
    out = np.full((T, N_COL_CONTACTO), CONTACTO_MAX, np.float32)

    def pose_pt(n_):
        return x[:, [ix[f"pose_{n_}_x"], ix[f"pose_{n_}_y"]]], x[:, ix[f"visible_{n_}"]] > 0

    obj = {"nariz": pose_pt("nariz"), "hombro_izq": pose_pt("hombro_izq"), "hombro_der": pose_pt("hombro_der")}
    hombros_ok = obj["hombro_izq"][1] & obj["hombro_der"][1]
    obj["pecho"] = (np.tile([0.0, PECHO_Y], (T, 1)), hombros_ok)
    for k, (n_, _) in enumerate(PUNTOS_CARA):
        if cara is not None:
            c = np.atleast_2d(np.asarray(cara, dtype=np.float64))
            p = c[:, [_PT0 + 2 * k, _PT0 + 2 * k + 1]]
            obj[n_] = (p, (c[:, 0] > 0) & (np.abs(p).sum(axis=1) > 0) & hombros_ok)
        else:
            obj[n_] = (np.zeros((T, 2)), np.zeros(T, bool))

    def punto_mano(lado, idxs):
        pts = np.mean([x[:, [ix[f"{lado}_p{i}_x"], ix[f"{lado}_p{i}_y"]]] for i in idxs], axis=0)
        return pts, x[:, ix[f"mano_{lado}"]] > 0

    if prof is not None:
        prof = np.atleast_2d(np.asarray(prof, dtype=np.float64))
        if prof.shape[1] < N_COL_PROF:                                  # tomas con el formato viejo de 5 columnas
            prof = np.hstack([prof, np.full((len(prof), N_COL_PROF - prof.shape[1]), np.nan)])
        ancho = np.where(np.isfinite(prof[:, 4]) & (prof[:, 4] > 0.05), prof[:, 4], ANCHO_HOMBROS_M)
        z_cab = np.where(np.isfinite(prof[:, 7]) & (prof[:, 7] > 0.1), prof[:, 7], 0.6)

    def hueco(lado_k, o):
        """Hueco de profundidad entre la mano y el objetivo, en anchos de hombro (0 si no hay dato)."""
        if prof is None:
            return 0.0
        es_cara = o in ("nariz", "frente", "boca", "barbilla")
        r = prof[:, 5 + lado_k]
        usa_r = np.isfinite(r)
        rr = np.where(usa_r, r, 1.0)
        if not es_cara:                                                 # pecho y hombros: algo mas lejos que la cara
            rr = rr * z_cab / (z_cab + CUERPO_ATRAS_M)
        g_r = np.maximum.reduce([(1.0 - rr) - PROF_TOL_REL, (rr - 1.0) - PROF_TOL_REL_ATRAS,
                                 np.zeros_like(rr)]) * z_cab / ancho
        gp = np.nan_to_num(prof[:, lado_k if es_cara else 2 + lado_k], nan=0.0)   # respaldo con la pose
        g_p = np.maximum(gp - PROF_TOL_M, 0.0) / ancho
        return np.where(usa_r, g_r, g_p)

    col, palmas = 0, {}
    for lk, lado in enumerate(LADOS_CONTACTO):
        for ref, idxs in REFS_MANO:
            pm, okm = punto_mano(lado, idxs)
            if ref == "palma":
                palmas[lado] = (pm, okm)
            for o in OBJETIVOS_CONTACTO:
                po, oko = obj[o]
                d = np.hypot(np.linalg.norm(pm - po, axis=1), hueco(lk, o))
                out[:, col] = np.where(okm & oko, np.minimum(d, CONTACTO_MAX), CONTACTO_MAX)
                col += 1
    (p_d, okd), (p_i, oki) = palmas["DER"], palmas["IZQ"]
    d = np.linalg.norm(p_d - p_i, axis=1)
    out[:, col] = np.where(okd & oki, np.minimum(d, CONTACTO_MAX), CONTACTO_MAX)
    return out


def mas_cercano(vec_con, lado):
    """(objetivo, distancia) mas cercano a la palma de esa mano, o (None, None) si no hay dato."""
    i0 = LADOS_CONTACTO.index(lado) * len(REFS_MANO) * len(OBJETIVOS_CONTACTO)
    d = np.asarray(vec_con[i0:i0 + len(OBJETIVOS_CONTACTO)])
    k = int(np.argmin(d))
    if d[k] >= CONTACTO_MAX - 1e-6:
        return None, None
    return OBJETIVOS_CONTACTO[k], float(d[k])


def panel_contacto(frame, vec_con, x0, y0, prof=None):
    fondo(frame, x0, y0, ANCHO_PANEL, 82)
    if prof is not None:
        f = lambda v, fmt: "--" if not np.isfinite(v) else format(v, fmt)   # noqa: E731
        adel = [(1.0 - prof[5 + k]) * prof[7] if np.isfinite(prof[5 + k]) and np.isfinite(prof[7]) else np.nan
                for k in range(2)]
        marca = "" if _REF_Z["fuente"] == "cara" else "*"       # * = cara tapada: referencia estimada
        texto(frame, f"mano/cara D {f(prof[5], '.2f')}  I {f(prof[6], '.2f')} {marca}", x0 + 6, y0 + 63,
              (255, 200, 0), 0.4)
        texto(frame, f"adelante(m) D {f(adel[0], '+.2f')}  I {f(adel[1], '+.2f')}", x0 + 6, y0 + 77,
              (255, 200, 0), 0.4)
    texto(frame, "CONTACTO", x0 + 6, y0 + 16, (0, 255, 255), 0.45, 1)
    for k, lado in enumerate(("IZQ", "DER")):
        nom, val = mas_cercano(vec_con, lado) if vec_con is not None else (None, None)
        if nom is None:
            texto(frame, f"{lado.capitalize()}: --", x0 + 6, y0 + 33 + 14 * k, (160, 160, 160), 0.42)
        else:
            toca = val < CONTACTO_DIST
            texto(frame, f"{lado.capitalize()}: {nom} {val:4.2f}" + ("  TOCA" if toca else ""),
                  x0 + 6, y0 + 33 + 14 * k, (0, 255, 0) if toca else (255, 255, 255), 0.42)


# ---------------- vector ampliado (fase C) ----------------
SALTO_MAX = 0.5            # anchos de hombro de referencia entre dos frames; mas que eso = fallo de la pose
VENTANA_DERIV = 5          # muestras de la media movil antes de derivar
PUNTOS_DERIV = ["muneca_izq", "muneca_der", "codo_izq", "codo_der"]
RASGOS_MANO = ["palma_nx", "palma_ny", "palma_nz", "eje_x", "eje_y", "eje_z",
               "sep_indice_medio", "sep_medio_anular", "sep_anular_menique", "sep_pulgar_indice", "pinza", "apertura"]
IDX_PALMA = [0, 5, 9, 13, 17]


def nombres_columnas_corp():
    return ["cadera_izq_x", "cadera_izq_y", "cadera_der_x", "cadera_der_y", "visible_cadera_izq", "visible_cadera_der",
            "z_nariz", "z_codo_izq", "z_codo_der", "z_muneca_izq", "z_muneca_der", "giro_tronco",
            "hombros_ancho", "hombros_x", "hombros_y"]


N_COL_CORP = len(nombres_columnas_corp())


def nombres_columnas_ext():
    cols = []
    for _, pref in LADOS_VEC:
        cols += [f"{pref}_{n_}" for n_ in RASGOS_MANO]
    cols += nombres_columnas_corp()
    cols += [f"vel_{n_}_{e}" for n_ in PUNTOS_DERIV for e in "xy"]
    cols += [f"acel_{n_}_{e}" for n_ in PUNTOS_DERIV for e in "xy"]
    for _, pref in LADOS_VEC:
        cols += [f"vel_palma_{pref}", f"acel_palma_{pref}", f"vel_dedos_{pref}"]
    return cols


N_COL_EXT = len(nombres_columnas_ext())
_COL_CORP0 = 2 * len(RASGOS_MANO)          # primera columna de cuerpo dentro de ext


def construir_corp(lms_pose, mundo_pose, w, h):
    """N_COL_CORP valores que solo se pueden medir al grabar (caderas, profundidad de la pose, escala de hombros)."""
    v = np.zeros(N_COL_CORP, np.float32)
    if lms_pose is None or mundo_pose is None:
        return v
    if not (visible_vivo(lms_pose[HOMBRO_IZQ]) and visible_vivo(lms_pose[HOMBRO_DER])):
        return v

    def P(i):
        return np.array([lms_pose[i].x * w, lms_pose[i].y * h], dtype=np.float64)

    origen = (P(HOMBRO_IZQ) + P(HOMBRO_DER)) / 2
    escala = float(np.linalg.norm(P(HOMBRO_IZQ) - P(HOMBRO_DER)))
    if escala < ESCALA_MIN * w:
        return v
    v[12], v[13], v[14] = escala / w, origen[0] / w, origen[1] / w
    for k, i in enumerate((CADERA_IZQ, CADERA_DER)):
        if visible_vivo(lms_pose[i]):
            v[2 * k:2 * k + 2] = (P(i) - origen) / escala
            v[4 + k] = 1.0
    a, b = mundo_pose[HOMBRO_IZQ], mundo_pose[HOMBRO_DER]
    ancho_w = math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
    if ancho_w > 0.05:
        zh = (a.z + b.z) / 2
        for k, i in enumerate((NARIZ, CODO_IZQ, CODO_DER, MUNECA_IZQ, MUNECA_DER)):
            if visible_vivo(lms_pose[i]):
                v[6 + k] = (zh - mundo_pose[i].z) / ancho_w        # + = mas cerca de la camara que los hombros
        v[11] = (b.z - a.z) / ancho_w                              # + = hombro izquierdo mas adelante
    return v


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _ang_u(u, v):
    """Angulo entre vectores (T,3) en grados/180."""
    return np.degrees(np.arccos(np.clip(np.sum(_unit(u) * _unit(v), axis=-1), -1.0, 1.0))) / 180.0


def rasgos_mano(pts, signo, presente):
    """pts: (T, 21, 3) de la mano como en el vector. signo: +1 mano izquierda, -1 derecha (normal saliendo de la
    palma). Devuelve (T, 12) en el orden de RASGOS_MANO (ceros donde no hay mano)."""
    P = np.array(pts, dtype=np.float64)
    P[..., 2] *= -1.0                                                  # z hacia la camara
    n = _unit(np.cross(P[:, 5] - P[:, 0], P[:, 17] - P[:, 0]) * signo)
    eje = _unit(P[:, 9] - P[:, 0])

    def base(a, b):                                                    # falange proximal en el plano de la palma
        v = P[:, b] - P[:, a]
        return v - np.sum(v * n, axis=1, keepdims=True) * n

    sep = [_ang_u(base(5, 6), base(9, 10)), _ang_u(base(9, 10), base(13, 14)),
           _ang_u(base(13, 14), base(17, 18)), _ang_u(base(2, 3), base(5, 6))]
    palma = np.linalg.norm(P[:, 9] - P[:, 0], axis=1) + 1e-9
    pinza = np.linalg.norm(P[:, 4] - P[:, 8], axis=1) / palma
    abre = np.linalg.norm(P[:, 4] - P[:, 20], axis=1) / palma
    out = np.column_stack([n, eje] + sep + [pinza, abre])
    out[~np.asarray(presente, bool)] = 0.0
    return out


def _derivar(sig, t, ok, salto=None):
    """Velocidad y aceleracion de una senal con huecos (se rellenan, se suaviza y se anulan donde no hay dato)."""
    T = len(sig)
    z = np.zeros(T)
    ok = np.array(ok, bool)
    sig = np.asarray(sig, dtype=np.float64)
    if salto is not None and T > 1:                                    # saltos imposibles = fallo de la pose
        mal = np.abs(np.diff(sig)) > salto
        ok[1:] &= ~mal
        ok[:-1] &= ~mal
    if T < 3 or ok.sum() < 3:
        return z, z.copy()
    idx = np.arange(T)
    s_ = np.interp(idx, idx[ok], sig[ok])
    k = VENTANA_DERIV
    s_ = np.convolve(np.pad(s_, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    tt = np.asarray(t, dtype=np.float64) + np.arange(T) * 1e-9          # tiempos estrictamente crecientes
    vel = np.gradient(s_, tt)
    acel = np.gradient(vel, tt)
    vel[~ok] = 0.0
    acel[~ok] = 0.0
    return vel, acel


def calcular_ext(x, t, corp=None):
    """x: (T, N_COLUMNAS); t: (T,) segundos; corp: (T, N_COL_CORP) o None (tomas viejas: esas columnas en NaN).
    Devuelve (T, N_COL_EXT). Velocidades en anchos de hombro DE REFERENCIA (percentil 90 del ancho de la toma)."""
    ix = _ix_vec()
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    t = np.asarray(t, dtype=np.float64)
    T = len(x)
    bloques, palmas = [], {}
    for lado, pref in LADOS_VEC:
        pts = np.stack([x[:, [ix[f"{pref}_p{i}_x"], ix[f"{pref}_p{i}_y"], ix[f"{pref}_p{i}_z"]]] for i in range(21)],
                       axis=1)
        pres = x[:, ix[f"mano_{pref}"]] > 0
        bloques.append(rasgos_mano(pts, 1.0 if lado == "Left" else -1.0, pres))
        palmas[pref] = (pts[:, IDX_PALMA, :2].mean(axis=1), pres)
    if corp is None:
        c = np.full((T, N_COL_CORP), np.nan)
    else:
        c = np.atleast_2d(np.asarray(corp, dtype=np.float64))
        if c.shape[1] < N_COL_CORP:
            c = np.hstack([c, np.full((len(c), N_COL_CORP - c.shape[1]), np.nan)])
        c = c[:, :N_COL_CORP].copy()
    # al girar el cuerpo el ancho de hombros en la imagen se encoge y las coordenadas normalizadas se inflan: con el
    # ancho y el centro de los hombros de cada frame se reconstruye la posicion y se mide en anchos de referencia
    esc, org = c[:, 12], c[:, 13:15]
    buena = np.isfinite(esc) & (esc > 0)
    estable = buena.sum() >= max(3, T // 2)
    ref = float(np.percentile(esc[buena], 90)) if estable else 1.0

    def a_ref(xy_):
        if not estable:
            return xy_, np.ones(T, bool)
        return (org + xy_ * esc[:, None]) / ref, buena

    if estable:                                                        # caderas en unidades de referencia
        for j in (0, 2):
            h_ = c[:, j:j + 2] * esc[:, None] / ref
            c[:, j:j + 2] = np.where(buena[:, None], h_, 0.0)
    bloques.append(c)
    vel_c, acel_c = [], []
    for n_ in PUNTOS_DERIV:
        ok = x[:, ix[f"visible_{n_}"]] > 0
        xy_, b_ = a_ref(x[:, [ix[f"pose_{n_}_x"], ix[f"pose_{n_}_y"]]])
        vx, ax = _derivar(xy_[:, 0], t, ok & b_, SALTO_MAX)
        vy, ay = _derivar(xy_[:, 1], t, ok & b_, SALTO_MAX)
        vel_c += [vx, vy]
        acel_c += [ax, ay]
    bloques.append(np.column_stack(vel_c))
    bloques.append(np.column_stack(acel_c))
    for _, pref in LADOS_VEC:
        centro, pres = palmas[pref]
        xy_, b_ = a_ref(centro)
        vx, ax = _derivar(xy_[:, 0], t, pres & b_, SALTO_MAX)
        vy, ay = _derivar(xy_[:, 1], t, pres & b_, SALTO_MAX)
        cols_ang = [ix[f"ang_{pref}_{n_.replace(' ', '_')}"] for n_ in NOMBRES_MANO]
        vd = np.mean([np.abs(_derivar(x[:, k], t, pres)[0]) for k in cols_ang], axis=0)
        bloques.append(np.column_stack([np.hypot(vx, vy), np.hypot(ax, ay), vd]))
    out = np.hstack(bloques).astype(np.float32)
    assert out.shape[1] == N_COL_EXT, out.shape
    return out


def panel_ext(frame, fila, x0, y0):
    """Panel FASE C con la ultima fila de calcular_ext (o None)."""
    fondo(frame, x0, y0, 220, 92)
    texto(frame, "FASE C  (tecla E)", x0 + 6, y0 + 15, (0, 255, 255), 0.45, 1)
    if fila is None:
        return
    cols = nombres_columnas_ext()
    g = lambda nombre: fila[cols.index(nombre)]                       # noqa: E731
    f = lambda v, fmt: "--" if not np.isfinite(v) else format(v, fmt)  # noqa: E731

    def mano(pref):
        return g(f"{pref}_palma_nx") != 0 or g(f"{pref}_palma_ny") != 0 or g(f"{pref}_palma_nz") != 0

    def por_mano(fn, fmt):
        return "  ".join(f"{p[0]} {format(fn(p), fmt) if mano(p) else '--'}" for p in ("DER", "IZQ"))

    texto(frame, "palma nz  " + por_mano(lambda p: g(f"{p}_palma_nz"), "+.2f"), x0 + 6, y0 + 30, (255, 255, 255), 0.4)
    texto(frame, "dedos sep " + por_mano(lambda p: 180 * np.mean([g(f"{p}_sep_indice_medio"), g(f"{p}_sep_medio_anular"),
                                                                g(f"{p}_sep_anular_menique")]), ".0f"),
          x0 + 6, y0 + 43, (255, 255, 255), 0.4)
    texto(frame, "pinza     " + por_mano(lambda p: g(f"{p}_pinza"), ".2f"), x0 + 6, y0 + 56, (255, 255, 255), 0.4)
    texto(frame, "vel palma " + por_mano(lambda p: g(f"vel_palma_{p}"), ".1f"), x0 + 6, y0 + 69, (255, 255, 255), 0.4)
    cad = "caderas si" if (g("visible_cadera_izq") > 0 or g("visible_cadera_der") > 0) else "caderas no"
    texto(frame, f"tronco giro {f(g('giro_tronco'), '+.2f')}  {cad}", x0 + 6, y0 + 82, (255, 200, 0), 0.4)


# ---------------- tomas: guardar, borrar, actualizar ----------------
def _guardar_json_si_falta(nombre, contenido):
    ruta = os.path.join(CARPETA_DATASET, nombre)
    if not os.path.exists(ruta):
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(contenido, f, ensure_ascii=False, indent=0)


def guardar_secuencia(sena, x, validos, n_capturados, fps, validos_pct, con_mano_pct, tiempos=None, vectores=None,
                      validos_raw=None, imagenes=None, cara_rs=None, cara_raw=None, con_rs=None, con_raw=None,
                      prof_raw=None, origen=None, ext_rs=None, ext_raw=None):
    """Guarda dataset/<sena>/<sena>_FECHA.npz y anade una fila a indice.csv. Devuelve la ruta relativa."""
    carpeta = os.path.join(CARPETA_DATASET, sena)
    os.makedirs(carpeta, exist_ok=True)
    ahora = datetime.now()
    nombre = f"{sena}_{ahora:%Y%m%d_%H%M%S}.npz"
    extra = {}
    if tiempos is not None and vectores is not None:      # toma completa a la velocidad original
        extra = {"x_raw": np.stack(vectores).astype(np.float32), "t_raw": np.asarray(tiempos, dtype=np.float64),
                 "validos_raw": np.asarray(validos_raw, dtype=bool)}
        if imagenes is not None and len(imagenes) == len(vectores):
            extra["img_off"] = np.cumsum([0] + [len(b) for b in imagenes]).astype(np.int64)
            extra["img_data"] = np.frombuffer(b"".join(imagenes), dtype=np.uint8)
    if cara_rs is not None and cara_raw is not None:
        extra["cara"], extra["cara_raw"] = np.asarray(cara_rs, np.float32), np.asarray(cara_raw, np.float32)
    if con_rs is not None and con_raw is not None:
        extra["contacto"], extra["contacto_raw"] = np.asarray(con_rs, np.float32), np.asarray(con_raw, np.float32)
    if origen:
        extra["origen"] = np.array(str(origen))
    if prof_raw is not None:
        extra["prof_raw"] = np.asarray(prof_raw, dtype=np.float32)
    if ext_rs is not None and ext_raw is not None:
        extra["ext"], extra["ext_raw"] = np.asarray(ext_rs, np.float32), np.asarray(ext_raw, np.float32)
    np.savez_compressed(os.path.join(carpeta, nombre), x=x, validos=validos, **extra)
    _guardar_json_si_falta("columnas.json", nombres_columnas())
    if "ext" in extra:
        _guardar_json_si_falta("columnas_ext.json", nombres_columnas_ext())
    if "prof_raw" in extra:
        _guardar_json_si_falta("columnas_prof.json", nombres_columnas_prof())
    if "contacto" in extra:
        _guardar_json_si_falta("columnas_contacto.json", nombres_columnas_contacto())
    if "cara" in extra:
        _guardar_json_si_falta("columnas_cara.json", nombres_columnas_cara())

    rel = f"{sena}/{nombre}"
    ruta_idx = os.path.join(CARPETA_DATASET, "indice.csv")
    nuevo = not os.path.exists(ruta_idx)
    with open(ruta_idx, "a", newline="", encoding="utf-8-sig") as f:
        esc = csv.DictWriter(f, fieldnames=CAMPOS_INDICE)
        if nuevo:
            esc.writeheader()
        esc.writerow({"archivo": rel, "sena": sena, "fecha": f"{ahora:%Y-%m-%d %H:%M:%S}",
                      "frames_capturados": n_capturados, "fps_medio": f"{fps:.1f}",
                      "validos_pct": f"{validos_pct:.0f}", "con_mano_pct": f"{con_mano_pct:.0f}"})
    return rel


def borrar_secuencia(rel):
    """Borra el .npz, sus CSV/PNG de resultados y su fila en indice.csv."""
    ruta = os.path.join(CARPETA_DATASET, *rel.split("/"))
    if os.path.exists(ruta):
        os.remove(ruta)
    base = os.path.join(CARPETA_RESULTADOS, os.path.dirname(rel), os.path.splitext(os.path.basename(rel))[0])
    for sufijo in ("_datos.csv", "_fases.csv", "_cara.csv", "_contacto.csv", "_ext.csv", ".png"):
        with contextlib.suppress(OSError):
            os.remove(base + sufijo)
    ruta_idx = os.path.join(CARPETA_DATASET, "indice.csv")
    if os.path.exists(ruta_idx):
        with open(ruta_idx, newline="", encoding="utf-8-sig") as f:
            filas = [r for r in csv.DictReader(f) if r["archivo"] != rel]
        with open(ruta_idx, "w", newline="", encoding="utf-8-sig") as f:
            esc = csv.DictWriter(f, fieldnames=CAMPOS_INDICE)
            esc.writeheader()
            esc.writerows(filas)


def contar_secuencias():
    c = {}
    for s in SENAS:
        d = os.path.join(CARPETA_DATASET, s)
        c[s] = len([f for f in os.listdir(d) if f.endswith(".npz")]) if os.path.isdir(d) else 0
    return c


def cargar_senas_existentes():
    """Anade a SENAS las carpetas que ya existen en dataset/ (senas creadas con la tecla N en otras sesiones)."""
    if os.path.isdir(CARPETA_DATASET):
        for nombre in sorted(os.listdir(CARPETA_DATASET)):
            if os.path.isdir(os.path.join(CARPETA_DATASET, nombre)) and nombre not in SENAS:
                SENAS.append(nombre)


def actualizar_tomas():
    """Anade ext / ext_raw a las tomas viejas del dataset (las columnas de cuerpo quedan en NaN si no se grabaron)."""
    rutas = sorted(glob.glob(os.path.join(CARPETA_DATASET, "*", "*.npz")))
    if not rutas:
        print(f"No hay tomas en {CARPETA_DATASET}")
        return
    nuevas = viejas = 0
    for ruta in rutas:
        with np.load(ruta, allow_pickle=False) as d:
            datos = {k: d[k] for k in d.files}
        if "x_raw" not in datos or "t_raw" not in datos:
            print(f"  omitida (sin x_raw/t_raw): {os.path.relpath(ruta, CARPETA_DATASET)}")
            continue
        corp = None
        if "ext_raw" in datos:                      # conserva las columnas de cuerpo grabadas (12 antes, 15 ahora)
            n_c = datos["ext_raw"].shape[1] - 2 * len(RASGOS_MANO) - 22
            if n_c > 0:
                corp = datos["ext_raw"][:, _COL_CORP0:_COL_CORP0 + n_c]
        ext_raw = calcular_ext(datos["x_raw"], datos["t_raw"], corp)
        datos["ext_raw"] = ext_raw
        datos["ext"] = ext_raw[indices_remuestreo(datos["t_raw"], len(datos["x"]))]
        tmp = ruta + ".tmp.npz"
        np.savez_compressed(tmp, **datos)
        os.replace(tmp, ruta)
        nuevas += corp is not None
        viejas += corp is None
        if GENERAR_RESULTADOS:
            sena = os.path.basename(os.path.dirname(ruta))
            carpeta_r = os.path.join(CARPETA_RESULTADOS, sena)
            os.makedirs(carpeta_r, exist_ok=True)
            guardar_csv_tabla(os.path.join(carpeta_r, os.path.splitext(os.path.basename(ruta))[0] + "_ext.csv"),
                              datos["t_raw"] - datos["t_raw"][0], ext_raw, nombres_columnas_ext(), CSV_PARA_EXCEL_ES)
    ruta_cols = os.path.join(CARPETA_DATASET, "columnas_ext.json")
    with open(ruta_cols, "w", encoding="utf-8") as f:
        json.dump(nombres_columnas_ext(), f, ensure_ascii=False, indent=0)
    print(f"Listo: {nuevas + viejas} tomas actualizadas ({viejas} sin caderas, z de pose ni escala de hombros: "
          f"NaN en esas {N_COL_CORP} columnas).\nColumnas: {ruta_cols}")


_cola_resultados = []   # mensajes que deja el hilo para mostrarlos en pantalla
_hilos = []


def lanzar_resultados(sena, rel, x, validos, dur, imagenes=None, cara_raw=None, tiempos=None, con_raw=None,
                      ext_raw=None):
    """CSV de cara, contactos y fase C; y (con graficar_secuencia.py) CSV de datos/fases + grafica en segundo plano."""
    if not GENERAR_RESULTADOS:
        return
    carpeta_r = os.path.join(CARPETA_RESULTADOS, sena)
    base_nombre = os.path.splitext(rel.split("/")[1])[0]
    if tiempos is not None:
        t_rel = np.asarray(tiempos) - tiempos[0]
        tablas = [("cara", cara_raw, nombres_columnas_cara()),
                  ("contacto", con_raw, nombres_columnas_contacto()
                   + (nombres_columnas_prof() if con_raw is not None and con_raw.shape[1] > N_COL_CONTACTO else [])),
                  ("ext", ext_raw, nombres_columnas_ext())]
        for sufijo, datos, columnas in tablas:
            if datos is None:
                continue
            try:
                os.makedirs(carpeta_r, exist_ok=True)
                ruta = os.path.join(carpeta_r, f"{base_nombre}_{sufijo}.csv")
                guardar_csv_tabla(ruta, t_rel, datos, columnas, CSV_PARA_EXCEL_ES)
                print(f"   {sufijo}: {ruta}")
            except Exception as e:
                print(f"   No se pudo guardar el CSV de {sufijo}: {e}")
    if _GRAF is None:
        return
    nombre = rel.split("/")[1]

    def tarea():
        try:
            r = _GRAF.generar_resultados(x, validos, nombres_columnas(), dur, sena, nombre, carpeta_r,
                                         excel_es=CSV_PARA_EXCEL_ES, recorte=RECORTAR_TRAMO_ACTIVO, imagenes=imagenes)
            for k in ("datos", "fases", "figura"):
                if r[k]:
                    print(f"   {k}: {r[k]}")
            msg = "Resultados listos (CSV" + (" + grafica" if r["figura"] else "") + f"): resultados/{sena}/"
            if r["aviso"]:
                print("   Aviso:", r["aviso"])
                msg += " - ver terminal"
        except Exception as e:
            msg = "No se pudieron generar los resultados (ver terminal)"
            print(f"   {msg}: {e}")
        _cola_resultados.append(msg)

    h = threading.Thread(target=tarea, daemon=True)
    _hilos.append(h)
    h.start()


def finalizar_secuencia(sena, tiempos, vectores, validos, imagenes=None, cara=None, prof=None, origen=None):
    """Valida, remuestrea y guarda la toma. Devuelve (ruta_relativa o None, mensaje corto para pantalla)."""
    n = len(validos)
    if n < 5:
        return None, "Descartada: muy pocos frames (FPS muy bajo)"
    pct = 100.0 * sum(validos) / n
    if pct < 100.0 * MIN_VALIDOS:
        return None, f"Descartada: hombros visibles solo en {pct:.0f}%"
    x, v = vivo_remuestrear(tiempos, vectores, validos, N_FRAMES)
    con_mano = 100.0 * float(np.mean(x[:, COL_MASK_MANOS].max(axis=1) > 0))
    fps = n / max(tiempos[-1] - tiempos[0], 1e-6)
    idx_rs = indices_remuestreo(tiempos, N_FRAMES)
    cara_raw = np.stack(cara).astype(np.float32) if cara is not None and len(cara) == n else None
    cara_rs = cara_raw[idx_rs] if cara_raw is not None else None
    x_raw = np.stack(vectores).astype(np.float32)
    prof_all = np.stack(prof).astype(np.float32) if prof is not None and len(prof) == n else None
    prof_raw = prof_all[:, :N_COL_PROF] if prof_all is not None else None          # profundidad mano-cara
    corp_raw = prof_all[:, N_COL_PROF:] if prof_all is not None and prof_all.shape[1] > N_COL_PROF else None
    con_raw = calcular_contactos(x_raw, cara_raw, prof_raw)
    ext_raw = calcular_ext(x_raw, tiempos, corp_raw)                                # fase C
    rel = guardar_secuencia(sena, x, v, n, fps, pct, con_mano, tiempos, vectores, validos, imagenes,
                            cara_rs, cara_raw, con_raw[idx_rs], con_raw, prof_raw, origen, ext_raw[idx_rs], ext_raw)
    con_csv = np.hstack([con_raw, prof_raw]) if prof_raw is not None else con_raw
    lanzar_resultados(sena, rel, x_raw, np.asarray(validos, bool), tiempos[-1] - tiempos[0], imagenes, cara_raw,
                      tiempos, con_csv, ext_raw)
    return rel, f"Guardada: {sena.upper()}  (validos {pct:.0f}%, manos {con_mano:.0f}%)"


# ---------------- videos subidos (U / F) y camara ----------------
def elegir_videos():
    """Selector de archivos (varios a la vez). Sin tkinter, pide la ruta por terminal."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        raiz = tk.Tk()
        raiz.withdraw()
        raiz.attributes("-topmost", True)
        rutas = filedialog.askopenfilenames(
            title="Elige uno o varios videos",
            filetypes=[("Videos", " ".join("*" + e for e in EXT_VIDEO)), ("Todos", "*.*")])
        raiz.destroy()
        return list(rutas)
    except Exception:
        s = input("Ruta del video (ENTER = cancelar): ").strip().strip('"')
        return [s] if s else []


def elegir_carpeta():
    try:
        import tkinter as tk
        from tkinter import filedialog
        raiz = tk.Tk()
        raiz.withdraw()
        raiz.attributes("-topmost", True)
        ruta = filedialog.askdirectory(title="Elige la carpeta con los videos")
        raiz.destroy()
        return ruta or ""
    except Exception:
        return input("Ruta de la carpeta (ENTER = cancelar): ").strip().strip('"')


def listar_videos_carpeta(carpeta, sena_actual):
    """[(ruta, sena)]: videos sueltos -> sena actual; videos en subcarpetas -> nombre de la subcarpeta de 1er nivel."""
    out = []
    if not carpeta or not os.path.isdir(carpeta):
        return out
    nombres = sorted(os.listdir(carpeta))
    for n_ in nombres:
        ruta = os.path.join(carpeta, n_)
        if os.path.isfile(ruta) and n_.lower().endswith(EXT_VIDEO):
            out.append((ruta, sena_actual))
    for n_ in nombres:
        sub = os.path.join(carpeta, n_)
        sena = limpiar_nombre(n_)
        if not os.path.isdir(sub) or not sena:
            continue
        for raiz, _, archivos in os.walk(sub):
            for a in sorted(archivos):
                if a.lower().endswith(EXT_VIDEO):
                    out.append((os.path.join(raiz, a), sena))
    return out


def abrir_video(ruta):
    """(VideoCapture, fps, total_frames) o (None, 0, 0) si no se puede abrir."""
    vcap = cv2.VideoCapture(ruta)
    if not vcap.isOpened():
        return None, 0.0, 0
    return vcap, fps_valido(vcap.get(cv2.CAP_PROP_FPS)), int(vcap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)


def ajustar_frame_video(frame):
    """Escala el frame al tamano de trabajo sin deformarlo (los angulos dependen de la proporcion)."""
    h0, w0 = frame.shape[:2]
    esc = min(ANCHO / w0, VIDEO_ALTO_MAX / h0)
    if abs(esc - 1.0) > 1e-3:
        frame = cv2.resize(frame, (max(int(w0 * esc), 2), max(int(h0 * esc), 2)),
                           interpolation=cv2.INTER_AREA if esc < 1 else cv2.INTER_LINEAR)
    if VIDEO_ESPEJADO:
        frame = cv2.flip(frame, 1)
    return frame


def dibujar_entrada_nombre(frame, texto_nombre):
    h, w = frame.shape[:2]
    fondo(frame, w // 2 - 200, h // 2 - 40, 400, 80, 0.8)
    texto(frame, "Nombre de la sena nueva:", w // 2 - 190, h // 2 - 15, (200, 200, 200), 0.5, 1)
    texto(frame, texto_nombre + "_", w // 2 - 190, h // 2 + 12, (0, 255, 255), 0.7, 2)
    texto(frame, "ENTER aceptar | ESC cancelar", w // 2 - 190, h // 2 + 32, (160, 160, 160), 0.4)


def dibujar_estado_secuencia(frame, estado, sena, conteo, restante, progreso, info_video=""):
    h, w = frame.shape[:2]
    fondo(frame, 0, h - 62, w - 120, 38)
    texto(frame, f"Sena: {sena.upper()}  ({conteo.get(sena, 0)} grabadas)", 8, h - 44, (0, 255, 255), 0.5, 1)
    texto(frame, "ESPACIO grabar | U video | F carpeta | L cambiar | N nueva | Z deshacer", 8, h - 28,
          (200, 200, 200), 0.4)
    if estado == "cuenta":
        cv2.putText(frame, str(int(math.ceil(restante))), (w // 2 - 20, h // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5, cv2.LINE_AA)
    elif estado in ("grabando", "video"):
        color = (0, 0, 255) if estado == "grabando" else (255, 200, 0)
        cv2.circle(frame, (w - 105, h - 38), 7, color, -1)
        texto(frame, "REC" if estado == "grabando" else "VIDEO", w - 92, h - 33, color, 0.55, 2)
        x0, y0, ancho = w // 2 - 100, 10, 200
        cv2.rectangle(frame, (x0, y0), (x0 + ancho, y0 + 12), (255, 255, 255), 1)
        cv2.rectangle(frame, (x0, y0), (x0 + int(ancho * progreso), y0 + 12), color, -1)
        if info_video:
            texto(frame, info_video, x0, y0 + 28, (255, 200, 0), 0.45, 1)


def abrir_camara():
    """DirectShow + MJPG + buffer minimo (si falla, modo normal)."""
    cap = None
    if CAMARA_DSHOW and hasattr(cv2, "CAP_DSHOW") and os.name == "nt":
        cap = cv2.VideoCapture(INDICE_CAMARA, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = None
    if cap is None:
        cap = cv2.VideoCapture(INDICE_CAMARA)
    if CAMARA_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
    cap.set(cv2.CAP_PROP_FPS, FPS_CAMARA)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara.")
    print(f"Camara: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
          f"a {cap.get(cv2.CAP_PROP_FPS):.0f} FPS pedidos")
    return cap


def _cargar_graficador():
    """graficar_secuencia.py es opcional (no forma parte de este archivo)."""
    global _GRAF
    if BASE not in sys.path:
        sys.path.insert(0, BASE)
    try:
        import graficar_secuencia
        _GRAF = graficar_secuencia
    except Exception as e:
        _GRAF = None
        print(f"AVISO: no se cargo graficar_secuencia.py ({e}). Se guardan las tomas y los CSV, sin grafica de fases.")


def captura_en_vivo():
    mp = importar_mediapipe()
    v = mp.tasks.vision
    print(f"MediaPipe {mp.__version__} (API de tareas)")
    _cargar_graficador()
    asegurar_modelo(URL_MANO, MODELO_MANO)
    asegurar_modelo(URL_POSE, MODELO_POSE)
    cara_ok = True
    try:
        asegurar_modelo(URL_CARA, MODELO_CARA)
    except Exception as e:
        cara_ok = False
        print(f"AVISO: no se pudo obtener el modelo de cara ({e}). Se sigue sin cara.")

    op_manos = v.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODELO_MANO), running_mode=v.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.6, min_tracking_confidence=0.6)
    op_pose = v.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODELO_POSE), running_mode=v.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.5)
    op_cara = v.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODELO_CARA), running_mode=v.RunningMode.VIDEO,
        num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5, min_tracking_confidence=0.5)
    cara_on = cara_ok and CARA_ACTIVA
    cap = abrir_camara()
    pool = ThreadPoolExecutor(max_workers=2) if PARALELO else None
    contador, res_pose_prev = 0, None
    t_cam = t_det = 0.0
    ver_tiempos = False

    cargar_senas_existentes()
    escribiendo, texto_nombre = False, ""
    estado, t_estado = "espera", 0.0       # espera / cuenta / grabando / video
    buf_t, buf_v, buf_ok, buf_img, buf_cara, buf_prof = [], [], [], [], [], []
    idx_sena = 0
    historial = []                         # secuencias guardadas en esta sesion (para Z)
    conteo = contar_secuencias()
    mostrar_panel = mostrar_cuerpo = mostrar_ext = ver_cuerpo = True
    cont_ext, fila_ext = 0, None
    buf_ext = collections.deque(maxlen=9)  # ultimos frames para velocidad / aceleracion del panel
    modo_art = 0
    mensaje, msg_hasta = "", 0.0
    ultimo = None                          # (frame sin paneles de texto, angulos cuerpo, datos manos) para la tecla S
    suav_manos, suav_mun, suav_pose = {}, {}, None
    t_inicio = time.time()
    t_prev = t_inicio

    vcap = None
    cola_videos = []
    fps_video, total_video, idx_video, nombre_video, sena_video = 30.0, 0, 0, "", ""
    lote_ok = lote_mal = 0
    ts_last = 0            # ultimo timestamp (ms) enviado a MediaPipe; debe ser siempre creciente
    ts_base = 0

    def abrir_siguiente_video():
        """Abre el siguiente video legible de la cola. Devuelve True si hay uno en proceso."""
        nonlocal vcap, fps_video, total_video, nombre_video, sena_video, estado, idx_video, ts_base
        nonlocal buf_t, buf_v, buf_ok, buf_img, buf_cara, buf_prof, suav_manos, suav_mun, suav_pose, lote_mal
        while cola_videos:
            ruta_v, sena_video = cola_videos.pop(0)
            vcap, fps_video, total_video = abrir_video(ruta_v)
            if vcap is not None:
                nombre_video = os.path.basename(ruta_v)
                estado, idx_video = "video", 0
                ts_base = ts_last + 1000
                buf_t, buf_v, buf_ok, buf_img, buf_cara, buf_prof = [], [], [], [], [], []
                suav_manos, suav_mun, suav_pose = {}, {}, None
                return True
            print(f"No se pudo abrir: {ruta_v}")
            lote_mal += 1
        return False

    with contextlib.ExitStack() as pila:
        det_manos = pila.enter_context(v.HandLandmarker.create_from_options(op_manos))
        det_pose = pila.enter_context(v.PoseLandmarker.create_from_options(op_pose))
        det_cara = pila.enter_context(v.FaceLandmarker.create_from_options(op_cara)) if cara_ok else None
        while True:
            if estado == "video":
                ok, frame_raw = vcap.read()
                if not ok:    # termino el video: guardar la toma y pasar al siguiente
                    vcap.release()
                    vcap = None
                    rel, mensaje = finalizar_secuencia(sena_video, buf_t, buf_v, buf_ok, buf_img, buf_cara, buf_prof,
                                                       origen=nombre_video)
                    mensaje = f"{nombre_video}: {mensaje}"
                    msg_hasta = time.time() + 3.5
                    print(f"{mensaje}  -> {rel}" if rel else mensaje)
                    if rel:
                        historial.append(rel)
                        conteo = contar_secuencias()
                        lote_ok += 1
                    else:
                        lote_mal += 1
                    estado = "espera"
                    if not abrir_siguiente_video():
                        t_inicio = time.time() - (ts_last + 1000) / 1000.0   # la camara retoma el reloj
                        if lote_ok + lote_mal > 1:
                            print(f"Tanda terminada: {lote_ok} tomas guardadas, {lote_mal} videos sin guardar (se "
                                  f"descartan si no se ven los hombros en al menos {MIN_VALIDOS:.0%} de los frames).")
                        lote_ok = lote_mal = 0
                    continue
                frame_raw = ajustar_frame_video(frame_raw)
            else:
                t0 = time.time()
                ok, frame_raw = cap.read()
                t_cam = (time.time() - t0) * 1000
                if not ok:
                    break

            frame = cv2.flip(frame_raw, 1)
            h, w = frame.shape[:2]
            ts_ms = (ts_base + int(idx_video * 1000.0 / fps_video)) if estado == "video" \
                else int((time.time() - t_inicio) * 1000)
            ts_ms = max(ts_ms, ts_last + 1)    # MediaPipe exige timestamps siempre crecientes
            ts_last = ts_ms

            t0 = time.time()
            img_manos = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_pose = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB))
            contador += 1
            hacer_pose = (estado == "video" or POSE_CADA_N <= 1 or res_pose_prev is None
                          or contador % POSE_CADA_N == 0)
            hacer_cara = cara_on and det_cara is not None
            res_cara = None
            if pool is not None:
                f_pose = pool.submit(det_pose.detect_for_video, img_pose, ts_ms) if hacer_pose else None
                f_cara = pool.submit(det_cara.detect_for_video, img_pose, ts_ms) if hacer_cara else None
                res_manos = det_manos.detect_for_video(img_manos, ts_ms)
                res_pose = f_pose.result() if f_pose is not None else res_pose_prev
                res_cara = f_cara.result() if f_cara is not None else None
            else:
                res_manos = det_manos.detect_for_video(img_manos, ts_ms)
                res_pose = det_pose.detect_for_video(img_pose, ts_ms) if hacer_pose else res_pose_prev
                res_cara = det_cara.detect_for_video(img_pose, ts_ms) if hacer_cara else None
            res_pose_prev = res_pose
            t_det = (time.time() - t0) * 1000

            # ---- cuerpo ----
            lms_pose = mundo_pose = None
            angulos_pose = {}
            if res_pose.pose_landmarks and res_pose.pose_world_landmarks:
                lms_pose = res_pose.pose_landmarks[0]
                mundo_pose = res_pose.pose_world_landmarks[0]
                angulos_pose = calcular_angulos_pose(lms_pose, mundo_pose, w, h)
                suav_pose = suavizar(angulos_pose, suav_pose)
                angulos_pose = {k: suav_pose[k] for k in angulos_pose}
                if mostrar_cuerpo:
                    dibujar_pose(frame, lms_pose)
            else:
                suav_pose = None

            # ---- manos ----
            manos = []
            for lms, mundo_lms, handed in zip(res_manos.hand_landmarks, res_manos.hand_world_landmarks,
                                              res_manos.handedness):
                modelo = handed[0].category_name
                if INVERTIR_ETIQUETA:
                    modelo = "Right" if modelo == "Left" else "Left"
                manos.append({"pts": np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32),
                              "mundo": np.array([[p.x, p.y, p.z] for p in mundo_lms], dtype=np.float32),
                              "modelo": modelo, "conf": handed[0].score})
            if cara_on and CARA_DIBUJAR and res_cara is not None and res_cara.face_landmarks:
                dibujar_cara(frame, res_cara.face_landmarks[0])

            datos_lado, vistas = {}, set()
            for m, (lado, origen) in zip(manos, asignar_lados(manos, lms_pose, w, h)):
                pts_se = mano_a_sin_espejo(m["pts"], w, h)
                angulos = suavizar(vivo_angulos_mano(m["mundo"]), suav_manos.get(lado))
                suav_manos[lado] = angulos
                vistas.add(lado)
                dibujar_mano(frame, m["pts"], COLOR_LADO[lado])
                mun_cruda = (angulo_muneca(lms_pose, mundo_pose, lado, pts_se, m["mundo"], w, h)
                             if lms_pose is not None else None)
                if mun_cruda is None:
                    suav_mun.pop(lado, None)
                    mun_val = None
                else:
                    prev = suav_mun.get(lado)
                    mun_val = mun_cruda if prev is None else ALPHA * mun_cruda + (1 - ALPHA) * prev
                    suav_mun[lado] = mun_val
                datos_lado[lado] = {"pts": m["pts"], "pts_se": pts_se, "mundo": m["mundo"], "conf": m["conf"],
                                    "origen": origen, "angulos": angulos, "muneca": mun_val,
                                    "palma": orientacion_palma(pts_se, lado), "dedos": direccion_dedos(m["pts"], w, h)}
            for dic in (suav_manos, suav_mun):
                for k in [k for k in dic if k not in vistas]:
                    del dic[k]

            # ---- vector del frame + grabacion ----
            vector = construir_vector(lms_pose, w, h, datos_lado)
            cara_vec = construir_cara(res_cara, lms_pose, w, h)
            prof_vec = np.concatenate([construir_prof(lms_pose, mundo_pose, datos_lado, res_cara, w, h),
                                       construir_corp(lms_pose, mundo_pose, w, h)])   # 10 de profundidad + 15 de cuerpo
            t_ahora = time.time()
            restante, progreso = 0.0, 0.0
            if estado == "cuenta":
                restante = CUENTA_ATRAS_S - (t_ahora - t_estado)
                if restante <= 0:
                    estado, t_estado = "grabando", t_ahora
                    buf_t, buf_v, buf_ok, buf_img, buf_cara, buf_prof = [], [], [], [], [], []
            if estado in ("grabando", "video"):
                if estado == "video":
                    t_rel = idx_video / fps_video          # tiempo del video, no del reloj
                    progreso = min(idx_video / max(total_video, 1), 1.0)
                else:
                    t_rel = t_ahora - t_estado
                    progreso = min(t_rel / DURACION_S, 1.0)
                buf_t.append(t_rel)
                buf_v.append(vector if vector is not None else np.zeros(N_COLUMNAS, np.float32))
                buf_ok.append(vector is not None)
                buf_cara.append(cara_vec)
                buf_prof.append(prof_vec)
                if GUARDAR_CAPTURAS:     # frame con el esqueleto dibujado, sin paneles ni texto
                    ok_j, jpg = cv2.imencode(".jpg", cv2.resize(frame, (ANCHO_CAPTURA, ALTO_CAPTURA)),
                                             [cv2.IMWRITE_JPEG_QUALITY, 80])
                    buf_img.append(jpg.tobytes() if ok_j else b"")
                if estado == "video":
                    idx_video += 1
                elif t_rel >= DURACION_S:
                    estado = "espera"
                    rel, mensaje = finalizar_secuencia(SENAS[idx_sena], buf_t, buf_v, buf_ok, buf_img, buf_cara,
                                                       buf_prof)
                    msg_hasta = time.time() + 3.5
                    print(f"{mensaje}  -> {rel}" if rel else mensaje)
                    if rel:
                        historial.append(rel)
                        conteo = contar_secuencias()

            if _cola_resultados:
                mensaje, msg_hasta = _cola_resultados.pop(0), time.time() + 4.0

            # ---- paneles ----
            if mostrar_panel:
                panel_mano(frame, "Left", datos_lado.get("Left"), 5)
                panel_mano(frame, "Right", datos_lado.get("Right"), w - ANCHO_PANEL - 5)
            if mostrar_cuerpo:
                panel_pose(frame, angulos_pose, 5, 8 + 16 * 11 + 16)
            if cara_on and mostrar_panel:
                panel_cara(frame, cara_vec, w - ANCHO_PANEL - 5, 200)
            if mostrar_panel:
                vec_con = (calcular_contactos(vector[None, :], cara_vec[None, :], prof_vec[None, :N_COL_PROF])[0]
                           if vector is not None else None)
                panel_contacto(frame, vec_con, w - ANCHO_PANEL - 5, 316, prof_vec[:N_COL_PROF])
            if mostrar_ext and mostrar_panel:
                t_ext = idx_video / max(fps_video, 1e-6) if estado == "video" else t_ahora
                if buf_ext and t_ext <= buf_ext[-1][0]:
                    buf_ext.clear()
                buf_ext.append((t_ext, vector if vector is not None else np.zeros(N_COLUMNAS, np.float32),
                                prof_vec[N_COL_PROF:]))
                cont_ext += 1
                if len(buf_ext) >= 3 and cont_ext % 3 == 0:     # el panel se refresca cada 3 frames (ahorra FPS)
                    fila_ext = calcular_ext(np.stack([b[1] for b in buf_ext]), [b[0] for b in buf_ext],
                                            np.stack([b[2] for b in buf_ext]))[-1]
                panel_ext(frame, fila_ext, 5 + ANCHO_PANEL + 5, 314)

            dibujar_verificacion(frame, lms_pose, angulos_pose, datos_lado, ver_cuerpo and mostrar_cuerpo, modo_art)

            # ---- FPS / estado ----
            t_now = time.time()
            fps = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            texto(frame, f"FPS: {fps:4.1f}", w - 110, h - 12, (0, 255, 0), 0.55, 2)
            if ver_tiempos:
                resto = 1000.0 / max(fps, 1e-6) - t_cam - t_det
                fondo(frame, w - 250, h - 62, 245, 44)
                texto(frame, f"camara {t_cam:4.0f} ms  deteccion {t_det:4.0f} ms", w - 245, h - 46, (255, 255, 255), 0.42)
                texto(frame, f"dibujo/otros {max(resto, 0):4.0f} ms", w - 245, h - 28, (255, 255, 255), 0.42)
            info_v = f"{nombre_video} -> {sena_video.upper()}  (+{len(cola_videos)} en cola)" if estado == "video" else ""
            dibujar_estado_secuencia(frame, estado, sena_video if estado == "video" else SENAS[idx_sena], conteo,
                                     restante, progreso, info_v)
            if escribiendo:
                dibujar_entrada_nombre(frame, texto_nombre)
            ultimo = (frame.copy(), dict(angulos_pose), dict(datos_lado))
            if modo_art > 0:
                texto(frame, f"Dedos: {VIVO_CICLO_ART[modo_art]}", 5, h - 12, (0, 255, 255), 0.5, 1)
            if time.time() < msg_hasta:
                fondo(frame, 0, h // 2 - 18, w, 36)
                texto(frame, mensaje, 10, h // 2 + 6, (0, 255, 0), 0.6, 2)
            cv2.imshow("Seguimiento: manos + cuerpo", frame)

            # ---- teclado ----
            tecla = cv2.waitKey(1) & 0xFF
            if escribiendo:   # mientras escribes, las teclas son letras (ESC cancela, no sale)
                if tecla == 13:
                    nuevo = limpiar_nombre(texto_nombre)
                    if nuevo:
                        if nuevo not in SENAS:
                            SENAS.append(nuevo)
                        idx_sena = SENAS.index(nuevo)
                        conteo = contar_secuencias()
                    escribiendo = False
                elif tecla == 27:
                    escribiendo = False
                elif tecla == 8:
                    texto_nombre = texto_nombre[:-1]
                elif 32 <= tecla < 127 and len(texto_nombre) < 20:
                    texto_nombre += chr(tecla)
                continue
            letra = chr(tecla).lower() if 32 < tecla < 127 else ""
            if tecla == 27:
                break
            elif letra == "a":
                ver_cuerpo = not ver_cuerpo
            elif letra == "j":
                modo_art = (modo_art + 1) % len(VIVO_CICLO_ART)
            elif letra == "s":
                if ultimo is not None:
                    nombre_png = guardar_captura(*ultimo)
                    mensaje, msg_hasta = f"Guardado: capturas/{nombre_png}", time.time() + 2.5
                    print(f"Captura guardada en {CARPETA_CAPTURAS} ({nombre_png})")
            elif letra == "t":
                ver_tiempos = not ver_tiempos
            elif letra == "c":
                if det_cara is None:
                    mensaje, msg_hasta = "Cara no disponible (falta el modelo)", time.time() + 2.0
                else:
                    cara_on = not cara_on
                    mensaje, msg_hasta = "Cara: " + ("ENCENDIDA" if cara_on else "apagada"), time.time() + 1.5
            elif letra == "e":
                mostrar_ext = not mostrar_ext
            elif letra == "v":
                mostrar_panel = not mostrar_panel
            elif letra == "p":
                mostrar_cuerpo = not mostrar_cuerpo
            elif tecla == 32:   # ESPACIO
                if estado == "espera":
                    estado, t_estado = "cuenta", time.time()
                elif estado == "video":     # cancelar video(s)
                    if vcap is not None:
                        vcap.release()
                        vcap = None
                    cola_videos.clear()
                    estado = "espera"
                    t_inicio = time.time() - (ts_last + 1000) / 1000.0
                    mensaje, msg_hasta = "Video cancelado", time.time() + 1.5
                else:
                    estado = "espera"
                    mensaje, msg_hasta = "Cancelado", time.time() + 1.5
            elif letra in ("u", "f") and estado == "espera":
                if letra == "f":
                    carpeta_v = elegir_carpeta()
                    cola_videos = listar_videos_carpeta(carpeta_v, SENAS[idx_sena])
                    for _, sn in cola_videos:
                        if sn not in SENAS:
                            SENAS.append(sn)
                    if carpeta_v and not cola_videos:
                        mensaje, msg_hasta = "No hay videos en esa carpeta", time.time() + 2.5
                    elif cola_videos:
                        resumen = collections.Counter(sn for _, sn in cola_videos)
                        print("Carpeta: " + ", ".join(f"{k}: {n_}" for k, n_ in resumen.items()))
                else:
                    cola_videos = [(r, SENAS[idx_sena]) for r in elegir_videos()]
                lote_ok = lote_mal = 0
                abrir_siguiente_video()
            elif letra == "l" and estado == "espera":
                idx_sena = (idx_sena + 1) % len(SENAS)
            elif letra == "n" and estado == "espera":
                escribiendo, texto_nombre = True, ""
            elif letra == "z" and estado == "espera" and historial:
                borrar_secuencia(historial.pop())
                conteo = contar_secuencias()
                mensaje, msg_hasta = "Ultima secuencia borrada", time.time() + 1.5

    if vcap is not None:
        vcap.release()
    if any(h_.is_alive() for h_ in _hilos):
        print("Terminando de generar los resultados de la ultima toma...")
        for h_ in _hilos:
            h_.join(timeout=30)
    cap.release()
    if pool is not None:
        pool.shutdown(wait=False)
    cv2.destroyAllWindows()


# ======================================================================================================================
# 16. LINEA DE COMANDOS
# ======================================================================================================================
def _crear_parser():
    ap = argparse.ArgumentParser(
        prog="sistema_lsc.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Sistema de adquisicion y medicion de datos para Lengua de Senas Colombiana (LSC).",
        epilog="Ejemplo: py -3.12 sistema_lsc.py analizar videos_mediapipe --persona Nikol")
    sub = ap.add_subparsers(dest="comando", metavar="COMANDO")
    detectores = ("auto", "holistic", "pose+manos")

    p = sub.add_parser("analizar", help="analiza un video o una carpeta de videos (pipeline completo)")
    p.add_argument("ruta", help="video o carpeta de videos")
    p.add_argument("--sena", help="fuerza el nombre de la sena (si no, sale del nombre del archivo)")
    p.add_argument("--persona", help="persona (si no, sale del nombre del archivo o 'Anonimo')")
    p.add_argument("--calib", help="archivo de calibracion (si no, calibracion_<persona>.json o calibracion.json)")
    p.add_argument("--sin-blur", action="store_true", help="no difuminar la cara en los fotogramas")
    p.add_argument("--sin-dataset", action="store_true", help="no reconstruir los datasets al final")
    p.add_argument("--sin-fisio", action="store_true", help="no calcular los angulos fisiologicos 3D")
    p.add_argument("--tronco", choices=("vertical", "cadera"), default=MODO_TRONCO, help="eje del tronco (3D)")
    p.add_argument("--detector", choices=detectores, default=DETECTOR, help="detector de MediaPipe")

    p = sub.add_parser("calibrar", help="calibracion con 3 videos: mano abierta, puno, brazos extendidos")
    p.add_argument("abierta")
    p.add_argument("puno")
    p.add_argument("brazos")
    p.add_argument("--persona", help="guarda calibracion_<persona>.json (si no, calibracion.json)")
    p.add_argument("--salida", help="archivo de salida")
    p.add_argument("--detector", choices=detectores, default=DETECTOR)

    p = sub.add_parser("calibrar-auto", help="calibracion desde videos con toda la secuencia")
    p.add_argument("videos", nargs="+", help="videos con la secuencia completa")
    p.add_argument("--persona", required=True)
    p.add_argument("--solo-abierta", nargs="*", default=[], metavar="VIDEO", help="videos solo de palmas abiertas")
    p.add_argument("--salida", help="por defecto calibracion_<persona>.json")
    p.add_argument("--abierta", type=tramo_segundos, help="segundos A-B (un solo video)")
    p.add_argument("--puno", type=tramo_segundos, help="segundos A-B (un solo video)")
    p.add_argument("--brazos", type=tramo_segundos, help="segundos A-B (un solo video)")
    p.add_argument("--detector", choices=detectores, default=DETECTOR)

    p = sub.add_parser("revisar-calib", help="revisa la calidad de un archivo de calibracion")
    p.add_argument("archivo", nargs="?", default=os.path.basename(CALIB_JSON))

    sub.add_parser("dataset", help="reconstruye los datasets de dataset/")
    p = sub.add_parser("fisiologicos", help="angulos fisiologicos 3D de todas las repeticiones")
    p.add_argument("--tronco", choices=("vertical", "cadera"), default=MODO_TRONCO)
    sub.add_parser("autotest", help="verifica formulas y signos de los angulos 3D")
    sub.add_parser("definiciones", help="imprime la tabla de definiciones de los angulos 3D")
    p = sub.add_parser("diagnostico", help="explica con numeros los avisos de 'fisiologicos'")
    p.add_argument("--tol", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5], help="tolerancias a evaluar")
    sub.add_parser("reorganizar", help="copia la estructura antigua a resultados/ y arma el dataset")
    sub.add_parser("vivo", help="captura en tiempo real con la camara (antes completo.py)")
    sub.add_parser("actualizar-tomas", help="anade el vector ampliado (fase C) a las tomas viejas")
    return ap


def _ruta_en_proyecto(ruta):
    """Una ruta relativa se busca primero donde se ejecuta y luego en la carpeta del proyecto."""
    if os.path.isabs(ruta) or os.path.exists(ruta):
        return os.path.abspath(ruta)
    return os.path.join(BASE, ruta)


def _preguntar(texto_pregunta, defecto=""):
    """input() con valor por defecto (ENTER lo acepta)."""
    r = input(f"{texto_pregunta}{f' [{defecto}]' if defecto else ''}: ").strip().strip('"')
    return r or defecto


def _si_no(texto_pregunta, defecto=True):
    r = input(f"{texto_pregunta} ({'S/n' if defecto else 's/N'}): ").strip().lower()
    return defecto if not r else r.startswith("s")


def modo_interactivo():
    """Menu para usar el programa sin escribir comandos (boton Ejecutar de VS Code o doble clic).
    Los videos y carpetas se eligen con una ventana del explorador de archivos."""
    opciones = [
        ("Analizar uno o varios VIDEOS", "videos"),
        ("Analizar una CARPETA de videos", "carpeta"),
        ("Calibrar a una persona (videos con mano abierta, puno y brazos extendidos)", "calibrar"),
        ("Reconstruir datasets y angulos fisiologicos 3D", "datos"),
        ("Captura en tiempo real con la camara", "vivo"),
        ("Salir", "salir"),
    ]
    while True:
        print("\n" + "=" * 70 + "\n SISTEMA LSC - que quieres hacer?\n" + "=" * 70)
        for k, (txt, _) in enumerate(opciones, 1):
            print(f"  {k}. {txt}")
        r = input("Opcion: ").strip()
        if not r.isdigit() or not 1 <= int(r) <= len(opciones):
            print("Escribe el numero de una opcion.")
            continue
        accion = opciones[int(r) - 1][1]
        try:
            if accion == "salir":
                return
            if accion in ("videos", "carpeta"):
                print("Se abre una ventana para elegir (puede quedar detras de VS Code)...")
                rutas = elegir_videos() if accion == "videos" else [c for c in [elegir_carpeta()] if c]
                if not rutas:
                    print("No elegiste nada.")
                    continue
                print("Elegido: " + "; ".join(rutas))
                print("Deja en blanco (ENTER) para sacar el dato del nombre de cada archivo.")
                sena = _preguntar("Nombre de la sena (ej: AGUA)").upper() or None
                persona = _preguntar("Nombre de la persona (ej: Pablo)").title() or None
                blur = _si_no("Difuminar la cara en las imagenes?", True)
                analizar_lote(rutas, sena, persona, blur=blur)
                print(f"\nResultados en: {CARPETA_RESULTADOS}")
            elif accion == "calibrar":
                salida = None
                if _si_no("Es una calibracion GENERAL para todas las personas?", False):
                    persona, salida = "General", CALIB_JSON
                    if os.path.exists(CALIB_JSON):
                        if not _si_no(f"Ya existe {os.path.basename(CALIB_JSON)}. Reemplazarla? (se guarda una copia "
                                      "de respaldo)", True):
                            continue
                        respaldo = os.path.join(BASE, f"calibracion_respaldo_{datetime.now():%Y%m%d_%H%M%S}.json")
                        shutil.copy2(CALIB_JSON, respaldo)
                        print(f"Copia de respaldo: {os.path.basename(respaldo)}")
                else:
                    persona = _preguntar("Nombre de la persona a calibrar (el MISMO que usaras al analizar, ej: P01)")
                    if not persona:
                        continue
                print("Elige los videos con la secuencia completa (mano abierta, puno y brazos extendidos)...")
                videos = elegir_videos()
                if not videos:
                    print("No elegiste nada.")
                    continue
                solo_ab = []
                if _si_no("Tienes ademas videos SOLO de palmas abiertas?", False):
                    solo_ab = elegir_videos()
                calibrar_auto(videos, persona, solo_ab, salida)
            elif accion == "datos":
                construir_datasets()
                calcular_fisiologicos()
            elif accion == "vivo":
                captura_en_vivo()
        except SystemExit as e:                      # errores esperados (ruta inexistente, etc.): seguir en el menu
            if e.code not in (None, 0):
                print(f"\nERROR: {e.code}")
        except KeyboardInterrupt:
            print("\nCancelado.")
        except Exception:
            traceback.print_exc()
            print("\nOcurrio un error (arriba el detalle). Puedes intentar otra opcion.")


def main(argv=None):
    if argv is None and len(sys.argv) == 1:           # sin argumentos (boton Ejecutar / doble clic) -> menu
        modo_interactivo()
        return
    a = _crear_parser().parse_args(argv)
    c = a.comando
    if c is None:
        _crear_parser().print_help()
    elif c == "analizar":
        analizar_lote(a.ruta, a.sena, a.persona, _ruta_en_proyecto(a.calib) if a.calib else None, not a.sin_blur,
                      not a.sin_dataset, not a.sin_fisio, a.tronco, a.detector)
    elif c == "calibrar":
        calibrar(a.abierta, a.puno, a.brazos, ruta_calibracion_salida(a.persona, a.salida), a.detector)
    elif c == "calibrar-auto":
        calibrar_auto(a.videos, a.persona, a.solo_abierta, a.salida, a.abierta, a.puno, a.brazos, a.detector)
    elif c == "revisar-calib":
        revisar_calibracion(_ruta_en_proyecto(a.archivo))
    elif c == "dataset":
        construir_datasets()
    elif c == "fisiologicos":
        calcular_fisiologicos(a.tronco)
    elif c == "autotest":
        sys.exit(1 if fis_autotest() else 0)
    elif c == "definiciones":
        with pd.option_context("display.max_colwidth", 90, "display.width", 250):
            print(fis_definiciones().to_string(index=False))
    elif c == "diagnostico":
        diagnostico_fisiologicos(a.tol)
    elif c == "reorganizar":
        reorganizar()
        construir_datasets()
    elif c == "vivo":
        captura_en_vivo()
    elif c == "actualizar-tomas":
        actualizar_tomas()


if __name__ == "__main__":
    main()