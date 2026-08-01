"""
pooling.py — combinazione di probabilità con pool logaritmico a pesi vincolati.

Il meta-modello ovvio sarebbe una logistica sulle log-probabilità dei modelli
base. Sui dati veri va male: con tre modelli e tre esiti sono 27 coefficienti
liberi stimati su predizioni out-of-fold di qualità disomogenea (i primi fold
sono addestrati su poco storico). Il risultato è un ensemble peggiore dei
modelli che combina — esattamente quello che si vuole evitare.

Il pool logaritmico impone invece la struttura giusta:

    p(esito k) ∝ exp( Σ_i w_i · log p_i(k) + b_k )

cioè una media geometrica pesata, con un solo peso per modello (non uno per
modello-esito) e un intercetta per esito che assorbe eventuali bias di base.
Con tre modelli sono 5 parametri invece di 27, e i pesi sono vincolati a
essere non negativi: un modello può essere ignorato, non invertito.

I pesi sono regolarizzati verso la media uniforme: senza evidenza contraria
l'ensemble resta una media geometrica dei modelli.
"""

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-6


def to_log_probs(p, n_classes):
    """Da probabilità (n,) o (n,k) a log-probabilità (n,k) normalizzate."""
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    p = np.clip(p, _EPS, None)
    p = p / p.sum(axis=1, keepdims=True)
    if p.shape[1] != n_classes:
        raise ValueError(f"attese {n_classes} classi, ricevute {p.shape[1]}")
    return np.log(p)


def stack_components(prob_list, n_classes):
    """Impila le predizioni dei modelli in un tensore (n, n_modelli, n_classi)."""
    logs = [to_log_probs(p, n_classes) for p in prob_list]
    return np.stack(logs, axis=1)


def _softmax(s):
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


class LogPoolMeta:
    """
    Pool logaritmico con pesi non negativi e intercette per esito.

        LP  tensore (n, M, K) di log-probabilità dei modelli base
        y   classe osservata (0..K-1)
    """

    def __init__(self, l2=0.5, max_weight=4.0, fit_intercept=True):
        self.l2 = l2
        self.max_weight = max_weight
        self.fit_intercept = fit_intercept
        self.w_ = None
        self.b_ = None
        self.n_models_ = 0
        self.n_classes_ = 0
        self.fitted_ = False

    # ── ottimizzazione ────────────────────────────────────────────────────────
    def _unpack(self, theta, M, K):
        w = theta[:M]
        b = np.zeros(K)
        if self.fit_intercept and K > 1:
            b[1:] = theta[M:M + K - 1]
        return w, b

    def _objective(self, theta, LP, y, sw, M, K, target_w):
        w, b = self._unpack(theta, M, K)
        s = np.tensordot(LP, w, axes=([1], [0])) + b          # (n, K)
        p = _softmax(s)
        idx = np.arange(len(y))

        wsum = sw.sum()
        nll = -np.sum(sw * np.log(np.clip(p[idx, y], _EPS, None))) / wsum
        pen = self.l2 * np.sum((w - target_w) ** 2)

        g = p.copy()
        g[idx, y] -= 1.0
        g = g * (sw[:, None] / wsum)                          # dNLL/ds

        grad_w = np.einsum("nk,nmk->m", g, LP) + 2.0 * self.l2 * (w - target_w)
        grad = [grad_w]
        if self.fit_intercept and K > 1:
            grad.append(g.sum(axis=0)[1:])
        return nll + pen, np.concatenate(grad)

    def fit(self, LP, y, sample_weight=None):
        LP = np.asarray(LP, dtype=float)
        y = np.asarray(y, dtype=int)
        n, M, K = LP.shape
        sw = (np.ones(n) if sample_weight is None
              else np.asarray(sample_weight, dtype=float))

        mask = np.isfinite(LP).all(axis=(1, 2)) & (y >= 0) & (y < K) & np.isfinite(sw)
        LP, y, sw = LP[mask], y[mask], sw[mask]
        if len(y) < 50 or len(np.unique(y)) < 2:
            self.fitted_ = False
            return self

        target_w = np.full(M, 1.0 / M)
        theta0 = np.concatenate([target_w, np.zeros(K - 1)]) if self.fit_intercept \
            else target_w.copy()
        bounds = [(0.0, self.max_weight)] * M
        if self.fit_intercept:
            bounds += [(-3.0, 3.0)] * (K - 1)

        res = minimize(self._objective, theta0, jac=True, method="L-BFGS-B",
                       bounds=bounds, args=(LP, y, sw, M, K, target_w),
                       options={"maxiter": 500})

        theta = res.x if res.success or np.all(np.isfinite(res.x)) else theta0
        self.w_, self.b_ = self._unpack(theta, M, K)
        self.n_models_, self.n_classes_ = M, K
        self.fitted_ = True
        return self

    # ── inferenza ─────────────────────────────────────────────────────────────
    def predict_proba(self, LP):
        LP = np.asarray(LP, dtype=float)
        if not self.fitted_:
            # media geometrica semplice
            return _softmax(LP.mean(axis=1))
        s = np.tensordot(LP, self.w_, axes=([1], [0])) + self.b_
        return _softmax(s)

    @property
    def weights(self):
        """Pesi normalizzati, leggibili come 'quanto conta ciascun modello'."""
        if self.w_ is None:
            return None
        s = self.w_.sum()
        return self.w_ / s if s > 0 else self.w_

    def __repr__(self):
        if not self.fitted_:
            return "<LogPoolMeta non addestrato>"
        w = ", ".join(f"{x:.2f}" for x in self.w_)
        return f"<LogPoolMeta pesi=[{w}] b={np.round(self.b_, 3).tolist()}>"
