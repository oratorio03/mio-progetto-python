"""
hmm.py — Modello di Markov a stati nascosti sulla forma delle squadre.

Idea: la forma di una squadra è una variabile latente che evolve lentamente
(stato "in crisi" / "regolare" / "in fiducia"). Non la osserviamo: osserviamo
solo l'esito delle partite. È esattamente la struttura di un HMM.

    stato latente   z_t ∈ {0..K-1}      transizioni A[z_{t-1}, z_t]
    osservazione    x_t ∈ {0..4}        emissione   B[z_t, venue_t, x_t]

Le emissioni sono condizionate al campo (casa/trasferta): lo stesso stato di
forma produce esiti diversi in casa e fuori, e questo è precisamente il segnale
che ci interessa per il mercato Home Win.

Simboli osservati (dal punto di vista della squadra):
    0 = sconfitta con 2+ gol di scarto
    1 = sconfitta di misura
    2 = pareggio
    3 = vittoria di misura
    4 = vittoria con 2+ gol di scarto

Addestramento: Baum-Welch (EM) con forward-backward scalato, in pooling su
tutte le sequenze di tutte le squadre — i parametri descrivono "come si muove
la forma nel calcio", non una singola squadra. Lo stato specifico della
squadra viene poi ricavato per filtraggio in avanti sulla sua sequenza.

Nessuna dipendenza esterna oltre numpy.
"""

import numpy as np
import pandas as pd

N_SYMBOLS = 5
N_VENUES  = 2      # 0 = casa, 1 = trasferta
SYM_LOSS  = (0, 1)
SYM_WIN   = (3, 4)


def encode_symbol(goals_for, goals_against):
    diff = goals_for - goals_against
    if diff <= -2:
        return 0
    if diff == -1:
        return 1
    if diff == 0:
        return 2
    if diff == 1:
        return 3
    return 4


# ── HMM discreto ──────────────────────────────────────────────────────────────

class FormHMM:
    """HMM discreto con emissioni condizionate al campo, stimato con Baum-Welch."""

    def __init__(self, n_states=3, n_iter=40, n_restarts=4, tol=1e-4,
                 smoothing=1.0, random_state=42):
        self.n_states   = n_states
        self.n_iter     = n_iter
        self.n_restarts = n_restarts
        self.tol        = tol
        self.smoothing  = smoothing
        self.random_state = random_state
        self.pi = None
        self.A  = None
        self.B  = None
        self.loglik_ = -np.inf

    # -- inizializzazione ------------------------------------------------------

    def _init_params(self, rng):
        K = self.n_states
        pi = rng.dirichlet(np.ones(K) * 5.0)
        # transizioni persistenti: la forma cambia lentamente
        A = rng.dirichlet(np.ones(K) * 2.0, size=K) + np.eye(K) * 2.0
        A /= A.sum(axis=1, keepdims=True)
        B = rng.dirichlet(np.ones(N_SYMBOLS) * 3.0, size=(K, N_VENUES))
        # rompi la simmetria ordinando gli stati da "debole" a "forte"
        tilt = np.linspace(-1.0, 1.0, K)
        grid = np.arange(N_SYMBOLS) - (N_SYMBOLS - 1) / 2.0
        B *= np.exp(0.6 * tilt[:, None, None] * grid[None, None, :])
        B /= B.sum(axis=2, keepdims=True)
        return pi, A, B

    # -- forward-backward ------------------------------------------------------

    def _emission(self, venues, symbols):
        """Matrice (T, K) con b_t(k) = B[k, venue_t, symbol_t]."""
        return self.B[:, venues, symbols].T

    def _forward(self, obs):
        T, K = obs.shape
        alpha = np.zeros((T, K))
        scale = np.zeros(T)
        a = self.pi * obs[0]
        s = a.sum()
        if s <= 0:
            a = np.full(K, 1.0 / K)
            s = 1e-300
        alpha[0] = a / a.sum()
        scale[0] = s
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * obs[t]
            s = a.sum()
            if s <= 0:
                a = np.full(K, 1.0 / K)
                s = 1e-300
            alpha[t] = a / a.sum()
            scale[t] = s
        return alpha, scale

    def _backward(self, obs, scale):
        T, K = obs.shape
        beta = np.zeros((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            b = self.A @ (obs[t + 1] * beta[t + 1])
            beta[t] = b / max(scale[t + 1], 1e-300)
        return beta

    # -- addestramento ---------------------------------------------------------

    def fit(self, sequences):
        """sequences: lista di (venues, symbols), array int della stessa lunghezza."""
        sequences = [(np.asarray(v, dtype=int), np.asarray(s, dtype=int))
                     for v, s in sequences if len(v) >= 2]
        if not sequences:
            raise ValueError("nessuna sequenza utilizzabile per l'HMM")

        best = None
        for restart in range(self.n_restarts):
            rng = np.random.default_rng(self.random_state + restart)
            self.pi, self.A, self.B = self._init_params(rng)
            prev_ll = -np.inf
            for _ in range(self.n_iter):
                ll = self._em_step(sequences)
                if abs(ll - prev_ll) < self.tol * max(1.0, abs(prev_ll)):
                    prev_ll = ll
                    break
                prev_ll = ll
            if best is None or prev_ll > best[0]:
                best = (prev_ll, self.pi.copy(), self.A.copy(), self.B.copy())

        self.loglik_, self.pi, self.A, self.B = best
        self._sort_states()
        return self

    def _em_step(self, sequences):
        K = self.n_states
        pi_acc = np.zeros(K)
        A_num  = np.zeros((K, K))
        A_den  = np.zeros(K)
        B_num  = np.zeros((K, N_VENUES, N_SYMBOLS))
        total_ll = 0.0

        for venues, symbols in sequences:
            obs = self._emission(venues, symbols)
            alpha, scale = self._forward(obs)
            beta = self._backward(obs, scale)
            total_ll += float(np.log(np.maximum(scale, 1e-300)).sum())

            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

            pi_acc += gamma[0]
            T = len(symbols)
            for t in range(T - 1):
                xi = (alpha[t][:, None] * self.A *
                      (obs[t + 1] * beta[t + 1])[None, :])
                tot = xi.sum()
                if tot > 0:
                    A_num += xi / tot
            A_den += gamma[:-1].sum(axis=0)

            for v in range(N_VENUES):
                mask_v = venues == v
                if not mask_v.any():
                    continue
                for s in range(N_SYMBOLS):
                    mask = mask_v & (symbols == s)
                    if mask.any():
                        B_num[:, v, s] += gamma[mask].sum(axis=0)

        eps = self.smoothing
        self.pi = (pi_acc + eps) / (pi_acc.sum() + eps * K)
        self.A  = (A_num + eps) / (A_num.sum(axis=1, keepdims=True) + eps * K)
        self.B  = (B_num + eps) / (B_num.sum(axis=2, keepdims=True) + eps * N_SYMBOLS)
        return total_ll

    def _sort_states(self):
        """Ordina gli stati per forza crescente: interpretabilità e stabilità."""
        grid = np.arange(N_SYMBOLS)
        strength = (self.B[:, 0, :] @ grid + self.B[:, 1, :] @ grid) / 2.0
        order = np.argsort(strength)
        self.pi = self.pi[order]
        self.A  = self.A[np.ix_(order, order)]
        self.B  = self.B[order]

    # -- inferenza -------------------------------------------------------------

    def filter_path(self, venues, symbols):
        """
        Ritorna un array (T+1, K): la riga t è la distribuzione sullo stato
        PRIMA di osservare la partita t (riga 0 = prior). Usando la riga t si
        predice la partita t senza mai guardarne l'esito.
        """
        venues  = np.asarray(venues, dtype=int)
        symbols = np.asarray(symbols, dtype=int)
        T = len(symbols)
        out = np.zeros((T + 1, self.n_states))
        out[0] = self.pi
        if T == 0:
            return out
        obs = self._emission(venues, symbols)
        alpha, _ = self._forward(obs)
        for t in range(T):
            out[t + 1] = alpha[t] @ self.A
        return out

    def symbol_dist(self, state_dist, venue):
        return state_dist @ self.B[:, venue, :]


# ── Modello Home Win basato su HMM ────────────────────────────────────────────

def build_sequences(results):
    """
    Da all_results.csv alle sequenze per squadra.
    Ritorna {team_id: {"dates": np.datetime64[], "venues": int[], "symbols": int[]}}
    ordinate cronologicamente.
    """
    df = results.sort_values("date")
    acc = {}
    for r in df.itertuples(index=False):
        hg, ag = float(r.home_goals), float(r.away_goals)
        for tid, venue, gf, ga in ((r.home_id, 0, hg, ag), (r.away_id, 1, ag, hg)):
            slot = acc.setdefault(tid, {"dates": [], "venues": [], "symbols": []})
            slot["dates"].append(r.date)
            slot["venues"].append(venue)
            slot["symbols"].append(encode_symbol(gf, ga))
    return {tid: {"dates":   np.array(v["dates"], dtype="datetime64[ns]"),
                  "venues":  np.array(v["venues"], dtype=int),
                  "symbols": np.array(v["symbols"], dtype=int)}
            for tid, v in acc.items()}


class HMMHomeModel:
    """
    Classificatore Home Win derivato dall'HMM.

    Due letture indipendenti della stessa partita:
      p_home = P(la squadra di casa vinca) dalla sua sequenza, emissione venue=casa
      p_away = P(la squadra ospite perda) dalla sua sequenza, emissione venue=trasferta

    Le due letture vengono fuse da una regressione logistica a due input sui
    rispettivi logit: pesi e taratura sono stimati sui dati, non imposti.
    """

    def __init__(self, n_states=3, random_state=42, **hmm_kwargs):
        self.hmm = FormHMM(n_states=n_states, random_state=random_state, **hmm_kwargs)
        self.coef_ = None
        self.intercept_ = 0.0
        self.sequences = {}
        self._paths = {}

    # -- utilità ---------------------------------------------------------------

    @staticmethod
    def _logit(p, eps=1e-4):
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    def set_history(self, results):
        """Sequenze usate per il filtraggio (tutte le partite note al momento t)."""
        self.sequences = build_sequences(results)
        self._paths = {tid: self.hmm.filter_path(v["venues"], v["symbols"])
                       for tid, v in self.sequences.items()} if self.hmm.pi is not None else {}
        return self

    def _state_before(self, team_id, date):
        """Distribuzione di stato usando solo le partite precedenti a `date`."""
        seq = self.sequences.get(team_id)
        if seq is None or len(seq["dates"]) == 0:
            return self.hmm.pi
        k = int(np.searchsorted(seq["dates"], np.datetime64(date), side="left"))
        path = self._paths.get(team_id)
        if path is None:
            path = self.hmm.filter_path(seq["venues"], seq["symbols"])
            self._paths[team_id] = path
        return path[min(k, len(path) - 1)]

    def raw_signals(self, meta):
        """
        meta: DataFrame con home_id, away_id, date.
        Ritorna array (n, 2): [P(casa vince | sequenza casa), P(ospite perde | sequenza ospite)]
        """
        out = np.zeros((len(meta), 2))
        for i, r in enumerate(meta.itertuples(index=False)):
            sh = self._state_before(r.home_id, r.date)
            sa = self._state_before(r.away_id, r.date)
            dh = self.hmm.symbol_dist(sh, 0)     # squadra di casa, in casa
            da = self.hmm.symbol_dist(sa, 1)     # ospite, in trasferta
            out[i, 0] = dh[list(SYM_WIN)].sum()
            out[i, 1] = da[list(SYM_LOSS)].sum()
        return out

    # -- API modello -----------------------------------------------------------

    def fit(self, results_train, meta_train, y_train, history=None):
        """
        results_train : partite del periodo di training (stima parametri HMM)
        meta_train    : righe (home_id, away_id, date) su cui tarare la fusione
        history       : partite note per il filtraggio (default: results_train)
        """
        seqs = build_sequences(results_train)
        self.hmm.fit([(v["venues"], v["symbols"]) for v in seqs.values()])
        self.set_history(history if history is not None else results_train)

        X = self._logit(self.raw_signals(meta_train))
        self._fit_logistic(X, np.asarray(y_train, dtype=float))
        return self

    def _fit_logistic(self, X, y, l2=1.0, iters=200):
        """Newton-IRLS su 2 feature + intercetta: piccolo e senza dipendenze."""
        n, d = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])
        w = np.zeros(d + 1)
        for _ in range(iters):
            z = Xb @ w
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
            g = Xb.T @ (p - y) + l2 * np.r_[0.0, w[1:]]
            W = np.clip(p * (1.0 - p), 1e-6, None)
            H = (Xb * W[:, None]).T @ Xb + l2 * np.diag([0.0] + [1.0] * d)
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.intercept_ = float(w[0])
        self.coef_ = w[1:]
        return self

    def predict_proba(self, meta):
        X = self._logit(self.raw_signals(meta))
        z = self.intercept_ + X @ self.coef_
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    # -- diagnostica -----------------------------------------------------------

    def describe_states(self):
        """Tabella leggibile: cosa emette ogni stato latente, per campo."""
        labels = ["KO 2+", "KO 1", "X", "W 1", "W 2+"]
        rows = []
        for k in range(self.hmm.n_states):
            for v, vname in enumerate(("casa", "trasferta")):
                row = {"stato": k, "campo": vname}
                row.update({labels[s]: round(float(self.hmm.B[k, v, s]), 3)
                            for s in range(N_SYMBOLS)})
                rows.append(row)
        return pd.DataFrame(rows)

    def transition_matrix(self):
        return pd.DataFrame(np.round(self.hmm.A, 3),
                            index=[f"da s{k}" for k in range(self.hmm.n_states)],
                            columns=[f"a s{k}" for k in range(self.hmm.n_states)])
