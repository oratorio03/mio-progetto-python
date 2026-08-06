"""
hmm.py — Modello di Markov a stati nascosti sulla forma delle squadre.

La forma di una squadra è una variabile latente che evolve lentamente e che
non si osserva: si osservano solo gli esiti delle partite. È letteralmente la
struttura di un HMM.

    stato latente   z_t ∈ {0..K-1}     transizioni  A[z_{t-1}, z_t]
    osservazione    x_t ∈ {0..4}       emissione    B[z_t, campo_t, x_t]

L'emissione è condizionata al campo: lo stesso stato di forma rende
diversamente in casa e in trasferta, ed è precisamente il segnale che serve
per il mercato "vittoria in casa".

Osservazioni, dal punto di vista della squadra:
    0 sconfitta con 2+ gol di scarto   3 vittoria di misura
    1 sconfitta di misura              4 vittoria con 2+ gol di scarto
    2 pareggio

I parametri si stimano con Baum-Welch in pooling su tutte le squadre: A e B
descrivono come si muove la forma nel calcio, non una singola squadra. Lo
stato della singola squadra si ottiene poi filtrando in avanti la sua
sequenza, fermandosi rigorosamente alla partita precedente.

Implementato solo con numpy.
"""

import numpy as np
import pandas as pd

N_SYMBOLS  = 5
N_VENUES   = 2          # 0 = casa, 1 = trasferta
SYM_LOSS   = [0, 1]
SYM_WIN    = [3, 4]
SYM_LABELS = ["KO 2+", "KO 1", "X", "W 1", "W 2+"]


def encode(scored, conceded):
    diff = scored - conceded
    if diff <= -2:
        return 0
    if diff == -1:
        return 1
    if diff == 0:
        return 2
    if diff == 1:
        return 3
    return 4


def build_sequences(played):
    """
    Da un DataFrame di partite giocate alle sequenze per squadra.
    {squadra: {"dates": …, "venues": …, "symbols": …}} in ordine cronologico.
    """
    acc = {}
    # match_id come secondo criterio: l'ordine dev'essere lo stesso qualunque
    # sia il pezzo di storico che stiamo guardando, altrimenti lo stato filtrato
    # di una squadra cambierebbe a seconda di quante partite ci sono dopo
    ordered = played.sort_values(["date", "match_id"], kind="mergesort")
    for m in ordered.itertuples(index=False):
        hg, ag = float(m.home_goals), float(m.away_goals)
        for team, venue, gf, ga in ((m.home, 0, hg, ag), (m.away, 1, ag, hg)):
            slot = acc.setdefault(team, {"dates": [], "venues": [], "symbols": []})
            slot["dates"].append(m.date)
            slot["venues"].append(venue)
            slot["symbols"].append(encode(gf, ga))
    return {team: {"dates":   np.array(v["dates"], dtype="datetime64[ns]"),
                   "venues":  np.array(v["venues"], dtype=int),
                   "symbols": np.array(v["symbols"], dtype=int)}
            for team, v in acc.items()}


# ── HMM discreto ──────────────────────────────────────────────────────────────

class FormHMM:
    """HMM discreto con emissioni per campo, stimato con Baum-Welch."""

    def __init__(self, n_states=3, n_iter=40, n_restarts=4, tol=1e-4,
                 smoothing=1.0, random_state=42):
        self.n_states   = n_states
        self.n_iter     = n_iter
        self.n_restarts = n_restarts
        self.tol        = tol
        self.smoothing  = smoothing
        self.random_state = random_state
        self.pi = self.A = self.B = None
        self.loglik_ = -np.inf

    # -- inizializzazione ------------------------------------------------------

    def _init(self, rng):
        K = self.n_states
        pi = rng.dirichlet(np.ones(K) * 5.0)
        A = rng.dirichlet(np.ones(K) * 2.0, size=K) + np.eye(K) * 2.0
        A /= A.sum(axis=1, keepdims=True)
        B = rng.dirichlet(np.ones(N_SYMBOLS) * 3.0, size=(K, N_VENUES))
        # rompe la simmetria: gli stati partono già ordinati da debole a forte,
        # altrimenti EM può assestarsi su permutazioni diverse a ogni restart
        tilt = np.linspace(-1.0, 1.0, K)
        grid = np.arange(N_SYMBOLS) - (N_SYMBOLS - 1) / 2.0
        B *= np.exp(0.6 * tilt[:, None, None] * grid[None, None, :])
        B /= B.sum(axis=2, keepdims=True)
        return pi, A, B

    # -- forward-backward ------------------------------------------------------

    def _emission(self, venues, symbols):
        """(T, K) con b_t(k) = B[k, campo_t, simbolo_t]."""
        return self.B[:, venues, symbols].T

    def _forward(self, obs):
        T, K = obs.shape
        alpha = np.zeros((T, K))
        scale = np.zeros(T)
        current = self.pi * obs[0]
        for t in range(T):
            if t:
                current = (alpha[t - 1] @ self.A) * obs[t]
            total = current.sum()
            if total <= 0:
                current, total = np.full(K, 1.0 / K), 1e-300
            alpha[t] = current / current.sum()
            scale[t] = total
        return alpha, scale

    def _backward(self, obs, scale):
        T, K = obs.shape
        beta = np.zeros((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (obs[t + 1] * beta[t + 1])) / max(scale[t + 1], 1e-300)
        return beta

    # -- stima -----------------------------------------------------------------

    def fit(self, sequences):
        """sequences: lista di (venues, symbols), array interi di pari lunghezza."""
        sequences = [(np.asarray(v, dtype=int), np.asarray(s, dtype=int))
                     for v, s in sequences if len(v) >= 2]
        if not sequences:
            raise ValueError("nessuna sequenza utilizzabile per l'HMM")

        best = None
        for restart in range(self.n_restarts):
            rng = np.random.default_rng(self.random_state + restart)
            self.pi, self.A, self.B = self._init(rng)
            previous = -np.inf
            for _ in range(self.n_iter):
                loglik = self._em_step(sequences)
                if abs(loglik - previous) < self.tol * max(1.0, abs(previous)):
                    previous = loglik
                    break
                previous = loglik
            if best is None or previous > best[0]:
                best = (previous, self.pi.copy(), self.A.copy(), self.B.copy())

        self.loglik_, self.pi, self.A, self.B = best
        self._sort_states()
        return self

    def _em_step(self, sequences):
        K = self.n_states
        pi_acc = np.zeros(K)
        A_num  = np.zeros((K, K))
        B_num  = np.zeros((K, N_VENUES, N_SYMBOLS))
        loglik = 0.0

        for venues, symbols in sequences:
            obs = self._emission(venues, symbols)
            alpha, scale = self._forward(obs)
            beta = self._backward(obs, scale)
            loglik += float(np.log(np.maximum(scale, 1e-300)).sum())

            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
            pi_acc += gamma[0]

            for t in range(len(symbols) - 1):
                xi = alpha[t][:, None] * self.A * (obs[t + 1] * beta[t + 1])[None, :]
                total = xi.sum()
                if total > 0:
                    A_num += xi / total

            for venue in range(N_VENUES):
                at_venue = venues == venue
                if not at_venue.any():
                    continue
                for symbol in range(N_SYMBOLS):
                    mask = at_venue & (symbols == symbol)
                    if mask.any():
                        B_num[:, venue, symbol] += gamma[mask].sum(axis=0)

        eps = self.smoothing
        self.pi = (pi_acc + eps) / (pi_acc.sum() + eps * K)
        self.A  = (A_num + eps) / (A_num.sum(axis=1, keepdims=True) + eps * K)
        self.B  = (B_num + eps) / (B_num.sum(axis=2, keepdims=True) + eps * N_SYMBOLS)
        return loglik

    def _sort_states(self):
        """Stati ordinati da 'in crisi' a 'in fiducia': stabilità e leggibilità."""
        grid = np.arange(N_SYMBOLS)
        strength = (self.B[:, 0, :] @ grid + self.B[:, 1, :] @ grid) / 2.0
        order = np.argsort(strength)
        self.pi = self.pi[order]
        self.A  = self.A[np.ix_(order, order)]
        self.B  = self.B[order]

    # -- inferenza -------------------------------------------------------------

    def filter_path(self, venues, symbols):
        """
        (T+1, K): la riga t è la distribuzione di stato PRIMA di osservare la
        partita t (riga 0 = prior). Usando la riga t si predice la partita t
        senza averne mai visto l'esito.
        """
        symbols = np.asarray(symbols, dtype=int)
        venues = np.asarray(venues, dtype=int)
        path = np.zeros((len(symbols) + 1, self.n_states))
        path[0] = self.pi
        if len(symbols):
            alpha, _ = self._forward(self._emission(venues, symbols))
            path[1:] = alpha @ self.A
        return path

    def symbol_dist(self, state_dist, venue):
        return state_dist @ self.B[:, venue, :]


# ── classificatore ────────────────────────────────────────────────────────────

class HMMHomeWin:
    """
    Due letture indipendenti della stessa partita:

        p_casa   = P(la squadra di casa vinca)  dalla sua sequenza, emissione in casa
        p_ospite = P(l'ospite perda)            dalla sua sequenza, emissione fuori

    fuse da una logistica a due input stimata sui dati (pesi non imposti).
    """

    def __init__(self, n_states=3, random_state=42, **kwargs):
        self.hmm = FormHMM(n_states=n_states, random_state=random_state, **kwargs)
        self.coef_ = np.zeros(2)
        self.intercept_ = 0.0
        self.sequences = {}
        self._paths = {}

    @staticmethod
    def _logit(p, eps=1e-4):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def set_history(self, played):
        """Sequenze note per il filtraggio (tutte le partite già giocate)."""
        self.sequences = build_sequences(played)
        self._paths = {}
        return self

    def _state_before(self, team, date):
        seq = self.sequences.get(team)
        if seq is None or len(seq["dates"]) == 0:
            return self.hmm.pi
        if team not in self._paths:
            self._paths[team] = self.hmm.filter_path(seq["venues"], seq["symbols"])
        path = self._paths[team]
        # searchsorted 'left' esclude la partita stessa e tutto ciò che segue
        index = int(np.searchsorted(seq["dates"], np.datetime64(date), side="left"))
        return path[min(index, len(path) - 1)]

    def signals(self, table):
        """(n, 2) con [P(casa vince), P(ospite perde)] dalle rispettive sequenze."""
        out = np.zeros((len(table), 2))
        for i, row in enumerate(table.itertuples(index=False)):
            home_dist = self.hmm.symbol_dist(self._state_before(row.home, row.date), 0)
            away_dist = self.hmm.symbol_dist(self._state_before(row.away, row.date), 1)
            out[i, 0] = home_dist[SYM_WIN].sum()
            out[i, 1] = away_dist[SYM_LOSS].sum()
        return out

    def fit(self, played_train, table_train, y, history=None):
        sequences = build_sequences(played_train)
        self.hmm.fit([(v["venues"], v["symbols"]) for v in sequences.values()])
        self.set_history(history if history is not None else played_train)
        self._fit_logistic(self._logit(self.signals(table_train)),
                           np.asarray(y, dtype=float))
        return self

    def _fit_logistic(self, X, y, l2=1.0, iters=200):
        """Newton-IRLS su due feature: piccolo, senza dipendenze, sufficiente."""
        Xb = np.hstack([np.ones((len(X), 1)), X])
        w = np.zeros(Xb.shape[1])
        penalty = np.diag([0.0] + [1.0] * X.shape[1])
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -35, 35)))
            grad = Xb.T @ (p - y) + l2 * np.r_[0.0, w[1:]]
            hess = (Xb * np.clip(p * (1 - p), 1e-6, None)[:, None]).T @ Xb + l2 * penalty
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                break
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.intercept_, self.coef_ = float(w[0]), w[1:]
        return self

    def predict_proba(self, table):
        z = self.intercept_ + self._logit(self.signals(table)) @ self.coef_
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    # -- lettura ---------------------------------------------------------------

    def emissions_table(self):
        rows = []
        for k in range(self.hmm.n_states):
            for venue, name in enumerate(("casa", "trasferta")):
                row = {"stato": k, "campo": name}
                row.update({SYM_LABELS[s]: round(float(self.hmm.B[k, venue, s]), 3)
                            for s in range(N_SYMBOLS)})
                rows.append(row)
        return pd.DataFrame(rows)

    def transitions_table(self):
        return pd.DataFrame(np.round(self.hmm.A, 3),
                            index=[f"da s{k}" for k in range(self.hmm.n_states)],
                            columns=[f"a s{k}" for k in range(self.hmm.n_states)])
