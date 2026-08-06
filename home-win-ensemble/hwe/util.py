"""
util.py — Funzioni numeriche condivise: logit, log-loss, calibrazione, pesi.
"""

import numpy as np

EPS = 1e-6


def logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=float), -35, 35)))


def log_loss(y, p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


class PlattCalibrator:
    """
    Ricalibrazione monodimensionale: p → sigmoid(a·logit(p) + b).

    a < 1 significa che il modello era sovra-sicuro e le probabilità vengono
    schiacciate verso il centro; a > 1 il contrario.
    """

    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, p, y):
        y = np.asarray(y, dtype=float)
        if len(y) < 20 or len(np.unique(y)) < 2:
            return self
        # discesa di Newton su due parametri: nessuna dipendenza esterna
        X = np.column_stack([logit(p), np.ones(len(y))])
        w = np.array([1.0, 0.0])
        for _ in range(100):
            prob = sigmoid(X @ w)
            grad = X.T @ (prob - y)
            hess = (X * np.clip(prob * (1 - prob), 1e-6, None)[:, None]).T @ X
            hess += np.eye(2) * 1e-6
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                break
            w -= step
            if np.max(np.abs(step)) < 1e-9:
                break
        if np.all(np.isfinite(w)):
            self.a, self.b = float(w[0]), float(w[1])
        return self

    def transform(self, p):
        return sigmoid(self.a * logit(p) + self.b)

    def recenter(self, p_reference, target_mean):
        """
        Sposta l'intercetta perché la media calibrata di `p_reference` torni a
        `target_mean`, lasciando la pendenza dov'è.

        Serve perché le due cose che il calibratore corregge si stimano bene in
        posti diversi. La PENDENZA (quanto il modello è troppo sicuro) va
        stimata fuori campione, altrimenti non si vede. Il LIVELLO no: un
        modello con intercetta prevede già, sul proprio training, una media
        pari alla frequenza osservata. Se la finestra out-of-fold ha avuto una
        frequenza di vittorie diversa dal training nel suo complesso, il
        calibratore si porta dietro quello scarto e lo applica a tutte le
        previsioni future. Misurato: media prevista 0.512 dove il modello
        grezzo diceva 0.467.
        """
        target_mean = float(np.clip(target_mean, 1e-3, 1 - 1e-3))
        base = self.a * logit(p_reference)

        low, high = -20.0, 20.0
        for _ in range(80):
            middle = (low + high) / 2.0
            if sigmoid(base + middle).mean() < target_mean:
                low = middle
            else:
                high = middle
        self.b = (low + high) / 2.0
        return self


def optimise_weights(P, y, shrink=0.15):
    """
    Pesi non negativi a somma 1 che minimizzano la log-loss della media pesata.
    Parametrizzazione softmax: nessun vincolo esplicito da gestire.

    `shrink` mescola i pesi ottimi con quelli uniformi: su poche centinaia di
    partite l'ottimo può azzerare un modello per puro rumore campionario, e un
    filo di shrinkage tiene tutti in gioco senza rinunciare al segnale.
    """
    P = np.clip(np.asarray(P, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    k = P.shape[1]
    uniform = np.full(k, 1.0 / k)
    if len(y) < 30 or len(np.unique(y)) < 2:
        return uniform

    def softmax(theta):
        e = np.exp(theta - theta.max())
        return e / e.sum()

    def objective(theta):
        return log_loss(y, P @ softmax(theta))

    best_w, best_value = uniform, objective(np.zeros(k))
    try:
        from scipy.optimize import minimize
        starts = [np.zeros(k)] + [np.eye(k)[i] * 2.0 for i in range(k)]
        for start in starts:
            result = minimize(objective, start, method="Nelder-Mead",
                              options={"maxiter": 2000, "xatol": 1e-4,
                                       "fatol": 1e-7})
            if result.fun < best_value:
                best_value, best_w = float(result.fun), softmax(result.x)
    except ImportError:
        pass

    return (1.0 - shrink) * best_w + shrink * uniform
