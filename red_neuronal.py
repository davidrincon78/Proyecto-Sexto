"""
red_neuronal.py  ·  Red neuronal (perceptrón multicapa) en NumPy puro
=====================================================================
No depende de PyTorch ni TensorFlow: solo NumPy.

Contiene:
  - RedNeuronal        : MLP con ReLU, dropout, Adam, L2 y pesos por clase.
                         Tiene .predecir(f) con el mismo formato que el
                         clasificador k-NN de traductor.py.
  - dividir_grupos     : reparto 80 % entrenamiento / 20 % validación por
                         grupos (un grupo = un gesto), estratificado por seña.
  - matriz_confusion   : cuenta aciertos y errores por seña.
"""
import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def dividir_grupos(clases_por_grupo, frac_val=0.20, semilla=0):
    """
    Reparte GRUPOS (no muestras sueltas) entre entrenamiento y validación,
    separando ~frac_val de los gestos de cada seña. Así las versiones
    recortadas de un mismo gesto nunca quedan una en cada lado.
    Una seña con un solo gesto va entera a entrenamiento.
    Devuelve (indices_train, indices_val).
    """
    rng = np.random.default_rng(semilla)
    por_clase = {}
    for i, cl in enumerate(clases_por_grupo):
        por_clase.setdefault(cl, []).append(i)
    train, val = [], []
    for cl in sorted(por_clase):
        ids = [int(i) for i in rng.permutation(por_clase[cl])]
        n = len(ids)
        if n < 2:
            n_val = 0
        else:
            n_val = min(max(int(round(n * frac_val)), 1), n - 1)
        val += ids[:n_val]
        train += ids[n_val:]
    return sorted(train), sorted(val)


def matriz_confusion(y_real, y_pred, k):
    m = np.zeros((k, k), dtype=int)
    for a, b in zip(y_real, y_pred):
        m[int(a), int(b)] += 1
    return m


class RedNeuronal:
    """MLP:  entrada → [Dense+ReLU+Dropout]×N → Dense → softmax."""

    def __init__(self, n_entrada, clases, ocultas=(256, 128), dropout=0.3,
                 l2=1e-3, semilla=0):
        self.clases = list(clases)
        self.dropout = float(dropout)
        self.l2 = float(l2)
        self.rng = np.random.default_rng(semilla)
        tam = [int(n_entrada), *[int(h) for h in ocultas], len(self.clases)]
        self.W = [self.rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32)
                  for a, b in zip(tam[:-1], tam[1:])]
        self.b = [np.zeros(b, np.float32) for b in tam[1:]]
        self.mu = np.zeros(tam[0], np.float32)
        self.sigma = np.ones(tam[0], np.float32)
        self.centros = None      # centroide de cada seña (para rechazar desconocidos)
        self.umbral = None       # distancia máxima aceptada al centroide
        self.precision = None    # precisión sobre el 20 % de validación

    @property
    def n_entrada(self):
        return self.W[0].shape[0]

    # ------------------------------------------------------------ interno
    def _norm(self, X):
        return ((np.asarray(X, np.float32) - self.mu) / self.sigma).astype(np.float32)

    def _forward(self, A, entrenando=False):
        caches = []
        for i in range(len(self.W) - 1):
            z = A @ self.W[i] + self.b[i]
            h = np.maximum(z, 0)
            mascara = None
            if entrenando and self.dropout > 0:
                mascara = ((self.rng.random(h.shape) >= self.dropout)
                           .astype(np.float32) / (1.0 - self.dropout))
                h = h * mascara
            caches.append((A, z, mascara))
            A = h
        return A @ self.W[-1] + self.b[-1], caches, A

    def _gradientes(self, X, y, pesos):
        """Pérdida (entropía cruzada ponderada) y gradientes de todos los parámetros."""
        n = len(X)
        logits, caches, ultima = self._forward(X, entrenando=True)
        p = softmax(logits)
        w = pesos[y]
        perdida = float(-np.sum(w * np.log(p[np.arange(n), y] + 1e-9)) / n)
        dz = p.copy()
        dz[np.arange(n), y] -= 1.0
        dz *= (w / n)[:, None]
        L = len(self.W)
        gW, gb = [None] * L, [None] * L
        gW[-1] = ultima.T @ dz + self.l2 * self.W[-1]
        gb[-1] = dz.sum(axis=0)
        da = dz @ self.W[-1].T
        for i in range(L - 2, -1, -1):
            entrada, z, mascara = caches[i]
            if mascara is not None:
                da = da * mascara
            dzi = da * (z > 0)
            gW[i] = entrada.T @ dzi + self.l2 * self.W[i]
            gb[i] = dzi.sum(axis=0)
            if i > 0:
                da = dzi @ self.W[i].T
        return perdida, gW, gb

    def _evaluar(self, X, y, pesos):
        p = softmax(self._forward(X)[0])
        perdida = float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-9)))
        return perdida, float(np.mean(p.argmax(1) == y))

    # ------------------------------------------------------------ entrenamiento
    def entrenar(self, X, y, Xv=None, yv=None, epocas=150, lr=2e-3, lote=32,
                 ruido=0.10, callback=None):
        """
        X, y   : muestras de entrenamiento (80 %).  y = índices de clase.
        Xv, yv : muestras de validación (20 %), solo para MEDIR (no se entrena con ellas).
        callback(epoca, epocas, metricas) se llama unas 20 veces durante el proceso.
        """
        X = np.asarray(X, np.float32)
        y = np.asarray(y, int)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 0.1
        Xn = self._norm(X)
        Xvn = self._norm(Xv) if Xv is not None and len(Xv) else None
        yv = np.asarray(yv, int) if Xvn is not None else None

        k = len(self.clases)
        cuenta = np.bincount(y, minlength=k).astype(np.float32)
        pesos = (len(y) / (k * np.maximum(cuenta, 1))).astype(np.float32)

        # Adam
        mW = [np.zeros_like(w) for w in self.W]
        vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]
        vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps, paso = 0.9, 0.999, 1e-8, 0
        historial = []
        cada = max(1, epocas // 20)

        for ep in range(epocas):
            lr_ep = lr * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * ep / epocas)))
            orden = self.rng.permutation(len(Xn))
            for ini in range(0, len(Xn), lote):
                idx = orden[ini:ini + lote]
                xb = Xn[idx]
                if ruido > 0:
                    xb = xb + self.rng.normal(0, ruido, xb.shape).astype(np.float32)
                _, gW, gb = self._gradientes(xb, y[idx], pesos)
                paso += 1
                c1, c2 = 1 - b1 ** paso, 1 - b2 ** paso
                for i in range(len(self.W)):
                    for P, G, M, V in ((self.W, gW, mW, vW), (self.b, gb, mb, vb)):
                        M[i] = b1 * M[i] + (1 - b1) * G[i]
                        V[i] = b2 * V[i] + (1 - b2) * G[i] ** 2
                        P[i] -= lr_ep * (M[i] / c1) / (np.sqrt(V[i] / c2) + eps)

            if ep % cada == 0 or ep == epocas - 1:
                perd, acc = self._evaluar(Xn, y, pesos)
                m = {"loss": perd, "acc": acc, "val_loss": None, "val_acc": None}
                if Xvn is not None:
                    m["val_loss"], m["val_acc"] = self._evaluar(Xvn, yv, pesos)
                historial.append((ep + 1, m))
                if callback:
                    callback(ep + 1, epocas, m)
        return historial

    def calibrar_rechazo(self, X, y, Xv=None, yv=None, factor=1.6):
        """
        Guarda el centroide de cada seña y fija el umbral para descartar gestos
        desconocidos: factor × percentil 90 de las distancias de las muestras de
        VALIDACIÓN (que la red no vio) a su centroide verdadero.
        """
        Xn, y = self._norm(X), np.asarray(y, int)
        self.centros = np.stack([Xn[y == i].mean(axis=0) for i in range(len(self.clases))])
        self.umbral = None
        if Xv is not None and len(Xv) >= 3:
            d = self._distancias(self._norm(Xv), np.asarray(yv, int))
            self.umbral = float(factor * np.percentile(d, 90))

    def _distancias(self, Xn, idx_clase):
        return np.sqrt(np.mean((Xn - self.centros[idx_clase]) ** 2, axis=1))

    # ------------------------------------------------------------ predicción
    def probabilidades(self, X):
        return softmax(self._forward(self._norm(X))[0])

    def predecir(self, f):
        """Mismo formato que el clasificador k-NN de traductor.py."""
        f = np.asarray(f, np.float32).reshape(1, -1)
        p = self.probabilidades(f)[0]
        i = int(p.argmax())
        d = 0.0
        if self.centros is not None:
            d = float(self._distancias(self._norm(f), np.array([i]))[0])
        fuera = bool(self.umbral is not None and d > self.umbral)
        return {"sena": self.clases[i], "conf": float(p[i]),
                "probs": {k: float(v) for k, v in zip(self.clases, p)},
                "dist": d, "fuera": fuera}

    # ------------------------------------------------------------ guardar / cargar
    def guardar(self, ruta):
        datos = {"n_capas": np.array(len(self.W)), "mu": self.mu, "sigma": self.sigma,
                 "clases": np.array(self.clases),
                 "umbral": np.array(np.nan if self.umbral is None else self.umbral),
                 "precision": np.array(np.nan if self.precision is None else self.precision)}
        if self.centros is not None:
            datos["centros"] = self.centros
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            datos[f"W{i}"], datos[f"b{i}"] = w, b
        np.savez_compressed(ruta, **datos)

    @classmethod
    def cargar(cls, ruta):
        d = np.load(ruta)
        n = int(d["n_capas"])
        ocultas = [d[f"W{i}"].shape[1] for i in range(n - 1)]
        red = cls(d["W0"].shape[0], [str(c) for c in d["clases"]], ocultas)
        red.W = [d[f"W{i}"].astype(np.float32) for i in range(n)]
        red.b = [d[f"b{i}"].astype(np.float32) for i in range(n)]
        red.mu, red.sigma = d["mu"], d["sigma"]
        if "centros" in d.files:
            red.centros = d["centros"]
        u, p = float(d["umbral"]), float(d["precision"])
        red.umbral = None if np.isnan(u) else u
        red.precision = None if np.isnan(p) else p
        return red