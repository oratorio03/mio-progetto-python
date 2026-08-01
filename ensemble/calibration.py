"""
calibration.py — calibrazione delle probabilità.

Un modello può ordinare bene le partite e sbagliare i livelli: dire 70% dove
la frequenza reale è 62%. Sul betting questo è fatale, perché l'edge si calcola
sul livello, non sull'ordinamento.

La calibrazione isotonica è non parametrica e monotona: non cambia l'ordine
delle previsioni, aggiusta solo i livelli. Nel multiclasse si calibra una
classe alla volta (one-vs-rest) e si rinormalizza.

Regola anti-leak: il calibratore va addestrato su predizioni out-of-fold o su
una finestra temporale precedente, mai sugli stessi dati usati per il modello.
"""

import numpy as np

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


class IdentityCalibrator:
    """Nessuna calibrazione — utile come riferimento nei confronti."""

    def fit(self, probs, y):
        return self

    def transform(self, probs):
        return np.asarray(probs, dtype=float)

    def fit_transform(self, probs, y):
        return self.fit(probs, y).transform(probs)


class IsotonicCalibrator:
    """
    Calibrazione isotonica one-vs-rest con rinormalizzazione.

    Funziona sia con probabilità binarie (vettore o matrice n×2)
    sia con matrici multiclasse n×k.

    Due protezioni contro il difetto tipico dell'isotonica su pochi dati:

      shrinkage  l'output è una media fra probabilità calibrata e originale,
                 con peso n/(n + shrink_n): con poche osservazioni la
                 calibrazione corregge poco, con molte corregge quasi tutto;
      floor      nessuna probabilità viene mai portata a zero. L'isotonica
                 assegna 0 a un intero blocco se in quel blocco l'evento non
                 si è mai verificato: sul betting significa quota infinita,
                 e una singola occorrenza manda la log loss a infinito.
    """

    def __init__(self, out_of_bounds="clip", min_samples=150,
                 shrink_n=400.0, floor=0.004):
        self.out_of_bounds = out_of_bounds
        self.min_samples = min_samples
        self.shrink_n = shrink_n
        self.floor = floor
        self.models_ = []
        self.n_classes_ = None
        self.n_fit_ = 0
        self.weight_ = 0.0
        self.fitted_ = False

    @staticmethod
    def _as_matrix(probs):
        p = np.asarray(probs, dtype=float)
        if p.ndim == 1:
            p = np.column_stack([1.0 - p, p])
        return p

    def fit(self, probs, y):
        P = self._as_matrix(probs)
        y = np.asarray(y, dtype=int)
        mask = np.isfinite(P).all(axis=1) & np.isfinite(y)
        P, y = P[mask], y[mask]
        self.n_classes_ = P.shape[1]
        self.models_ = []

        if len(y) < self.min_samples:
            # troppo pochi dati: meglio non calibrare che calibrare a caso
            self.fitted_ = False
            return self

        for k in range(self.n_classes_):
            target = (y == k).astype(float)
            if target.sum() == 0 or target.sum() == len(target):
                self.models_.append(None)
                continue
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                                     out_of_bounds=self.out_of_bounds)
            iso.fit(P[:, k], target)
            self.models_.append(iso)

        self.n_fit_ = int(len(y))
        self.weight_ = float(self.n_fit_ / (self.n_fit_ + self.shrink_n))
        self.fitted_ = True
        return self

    def transform(self, probs):
        P = self._as_matrix(probs)
        if not self.fitted_:
            return P
        out = np.empty_like(P)
        for k in range(P.shape[1]):
            iso = self.models_[k] if k < len(self.models_) else None
            out[:, k] = P[:, k] if iso is None else iso.predict(P[:, k])

        out = self.weight_ * out + (1.0 - self.weight_) * P
        out = np.clip(out, self.floor, 1.0)
        s = out.sum(axis=1, keepdims=True)
        out = np.where(s > 0, out / s, P)
        return out

    def fit_transform(self, probs, y):
        return self.fit(probs, y).transform(probs)


class SigmoidCalibrator:
    """Calibrazione di Platt: più stabile dell'isotonica con pochi dati, ma parametrica."""

    def __init__(self, min_samples=30):
        self.min_samples = min_samples
        self.models_ = []
        self.fitted_ = False

    @staticmethod
    def _as_matrix(probs):
        p = np.asarray(probs, dtype=float)
        if p.ndim == 1:
            p = np.column_stack([1.0 - p, p])
        return p

    @staticmethod
    def _logit(p):
        p = np.clip(p, _EPS, 1 - _EPS)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, probs, y):
        P = self._as_matrix(probs)
        y = np.asarray(y, dtype=int)
        mask = np.isfinite(P).all(axis=1) & np.isfinite(y)
        P, y = P[mask], y[mask]
        self.models_ = []
        if len(y) < self.min_samples:
            self.fitted_ = False
            return self
        for k in range(P.shape[1]):
            target = (y == k).astype(int)
            if target.sum() in (0, len(target)):
                self.models_.append(None)
                continue
            lr = LogisticRegression(C=1e6, solver="lbfgs")
            lr.fit(self._logit(P[:, k]), target)
            self.models_.append(lr)
        self.fitted_ = True
        return self

    def transform(self, probs):
        P = self._as_matrix(probs)
        if not self.fitted_:
            return P
        out = np.empty_like(P)
        for k in range(P.shape[1]):
            lr = self.models_[k] if k < len(self.models_) else None
            out[:, k] = P[:, k] if lr is None else lr.predict_proba(self._logit(P[:, k]))[:, 1]
        out = np.clip(out, _EPS, 1.0)
        return out / out.sum(axis=1, keepdims=True)

    def fit_transform(self, probs, y):
        return self.fit(probs, y).transform(probs)


def make_calibrator(kind="isotonic"):
    kind = (kind or "none").lower()
    if kind == "isotonic":
        return IsotonicCalibrator()
    if kind in ("sigmoid", "platt"):
        return SigmoidCalibrator()
    return IdentityCalibrator()
