"""
models.py — I tre modelli base e la loro combinazione.

    1. Regressione logistica   — segnale lineare, liscio, ben calibrato
    2. HMM di forma            — dinamica temporale latente (vedi hmm.py)
    3. Albero decisionale      — interazioni e soglie non lineari

Combinazione: calibrazione e pesi si stimano su predizioni OUT-OF-FOLD generate
a finestra espansiva (walk-forward interno al training). Poi i tre output
calibrati vengono fusi con pesi che minimizzano la log-loss, e il risultato
passa per un ultimo calibratore. Nessun peso deciso a mano.

Perché out-of-fold e non una singola finestra di validazione: con una sola
finestra il calibratore viene stimato su modelli addestrati su meno dati di
quelli poi messi in produzione, e il risultato è sistematicamente sovra-sicuro
(misurato: pendenza di ricalibrazione ~0.45 invece di 1.0, log-loss peggiore
di una costante nonostante un AUC chiaramente sopra 0.5). Le previsioni
out-of-fold arrivano invece da modelli addestrati su volumi confrontabili con
quello finale, e sono molte di più.

Disciplina anti-leakage:
  - split cronologico, mai casuale
  - iperparametri scelti con TimeSeriesSplit dentro il primo blocco di training
  - calibratori e pesi stimati solo su predizioni out-of-fold
  - rifit finale dei modelli base su tutto il training, pesi/calibratori invariati
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import features as F
from .hmm import HMMHomeModel

EPS = 1e-6
BASE_NAMES = ("logistic", "hmm", "tree")


# ── utilità ───────────────────────────────────────────────────────────────────

def logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=float), -35, 35)))


def log_loss_safe(y, p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class PlattCalibrator:
    """Ricalibrazione monodimensionale: sigmoide su a*logit(p)+b."""

    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, p, y):
        y = np.asarray(y, dtype=float)
        if len(np.unique(y)) < 2 or len(y) < 20:
            return self
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(logit(p).reshape(-1, 1), y)
        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
        return self

    def transform(self, p):
        return sigmoid(self.a * logit(p) + self.b)


# ── modelli base ──────────────────────────────────────────────────────────────

class LogisticBase:
    """Logistica L2 su feature standardizzate. C scelto per log-loss in CV temporale."""

    GRID_C = (0.02, 0.05, 0.1, 0.3, 1.0, 3.0)

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = None
        self.clf = None
        self.C = None

    def _make(self, C):
        return LogisticRegression(C=C, solver="lbfgs", max_iter=2000,
                                  random_state=self.random_state)

    def select(self, X, y, n_splits=3):
        self.C = _select_by_cv(self._make, self.GRID_C, X, y, n_splits, scale=True)
        return self

    def fit(self, X, y):
        if self.C is None:
            self.select(X, y)
        self.scaler = StandardScaler().fit(X)
        self.clf = self._make(self.C).fit(self.scaler.transform(X), y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]

    def coefficients(self):
        return pd.Series(self.clf.coef_[0], index=F.FEATURE_COLS).sort_values(
            key=np.abs, ascending=False)


class TreeBase:
    """
    Albero singolo, potato. Volutamente poco profondo: serve a catturare
    interazioni a soglia (es. "Elo alto E ospite in crisi fuori casa"),
    non a memorizzare il campionato.
    """

    GRID = tuple({"max_depth": d, "min_samples_leaf": leaf, "ccp_alpha": a}
                 for d in (3, 4, 5, 6)
                 for leaf in (30, 60, 120)
                 for a in (0.0, 0.0005))

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.clf = None
        self.params = None

    def _make(self, params):
        return DecisionTreeClassifier(random_state=self.random_state,
                                      criterion="entropy", **params)

    def select(self, X, y, n_splits=3):
        self.params = _select_by_cv(self._make, self.GRID, X, y, n_splits, scale=False)
        return self

    def fit(self, X, y):
        if self.params is None:
            self.select(X, y)
        self.clf = self._make(self.params).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def importances(self):
        return pd.Series(self.clf.feature_importances_,
                         index=F.FEATURE_COLS).sort_values(ascending=False)


def _select_by_cv(make_fn, grid, X, y, n_splits, scale):
    """Seleziona l'iperparametro con la log-loss media su TimeSeriesSplit."""
    n = len(y)
    if n < 200 or len(np.unique(y)) < 2:
        return grid[len(grid) // 2]
    n_splits = max(2, min(n_splits, n // 100))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best, best_ll = grid[0], np.inf
    for cand in grid:
        losses = []
        for tr, va in tscv.split(X):
            if len(np.unique(y[tr])) < 2:
                continue
            Xtr, Xva = X[tr], X[va]
            if scale:
                sc = StandardScaler().fit(Xtr)
                Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
            try:
                m = make_fn(cand).fit(Xtr, y[tr])
                losses.append(log_loss_safe(y[va], m.predict_proba(Xva)[:, 1]))
            except Exception:
                continue
        if losses:
            ll = float(np.mean(losses))
            if ll < best_ll:
                best_ll, best = ll, cand
    return best


# ── ensemble ──────────────────────────────────────────────────────────────────

def optimise_weights(P, y, shrink=0.0):
    """
    Pesi non negativi a somma 1 che minimizzano la log-loss della media pesata.
    Parametrizzazione softmax: nessun vincolo esplicito da gestire.

    `shrink` mescola i pesi ottimi con quelli uniformi. Su poche centinaia di
    partite l'ottimo può azzerare un modello per puro rumore campionario: un
    filo di shrinkage tiene tutti e tre in gioco senza rinunciare al segnale.
    """
    P = np.clip(np.asarray(P, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    k = P.shape[1]
    if len(y) < 30 or len(np.unique(y)) < 2:
        return np.full(k, 1.0 / k)

    def unpack(theta):
        e = np.exp(theta - theta.max())
        return e / e.sum()

    def obj(theta):
        return log_loss_safe(y, P @ unpack(theta))

    best_w, best_v = np.full(k, 1.0 / k), obj(np.zeros(k))
    try:
        from scipy.optimize import minimize
        for start in (np.zeros(k), np.array([1.0] + [0.0] * (k - 1))):
            r = minimize(obj, start, method="Nelder-Mead",
                         options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-7})
            if r.fun < best_v:
                best_v, best_w = float(r.fun), unpack(r.x)
    except Exception:
        pass
    if shrink > 0:
        best_w = (1.0 - shrink) * best_w + shrink / k
    return best_w


class HomeWinEnsemble:
    """
    Ensemble Home Win. Uso tipico:

        ens = HomeWinEnsemble().fit(results_train)
        ens.prepare_history(results_noti)          # per il filtraggio HMM
        p = ens.predict_proba(feature_rows)
    """

    def __init__(self, n_states=3, oof_frac=0.5, n_folds=3, weight_shrink=0.15,
                 random_state=42, verbose=True):
        self.n_states    = n_states
        self.oof_frac    = oof_frac     # coda del training usata come out-of-fold
        self.n_folds     = n_folds
        self.weight_shrink = weight_shrink
        self.random_state = random_state
        self.verbose     = verbose

        self.imputer     = None
        self.logistic    = None
        self.tree        = None
        self.hmm_model   = None
        self.calibrators = {}
        self.final_calibrator = None
        self.weights     = None
        self.train_end   = None
        self.n_train     = 0
        self.n_oof       = 0
        self.base_valid_scores = {}
        self.oof_scores  = {}

    # -- helper interni --------------------------------------------------------

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _base_probs(self, X, meta):
        raw = {
            "logistic": self.logistic.predict_proba(X),
            "hmm":      self.hmm_model.predict_proba(meta),
            "tree":     self.tree.predict_proba(X),
        }
        cal = {k: self.calibrators[k].transform(v) if k in self.calibrators else v
               for k, v in raw.items()}
        return raw, cal

    def _fit_bases(self, results, tab, X, y, hparams=None, history=None):
        """Addestra i tre modelli base su un blocco di training."""
        lr = LogisticBase(self.random_state)
        tr = TreeBase(self.random_state)
        if hparams:
            lr.C, tr.params = hparams
        lr.fit(X, y)
        tr.fit(X, y)
        hm = HMMHomeModel(n_states=self.n_states,
                          random_state=self.random_state).fit(
            results, tab[["home_id", "away_id", "date"]], y,
            history=history if history is not None else results)
        return lr, tr, hm

    # -- fit -------------------------------------------------------------------

    def fit(self, results, table=None):
        """
        results : DataFrame delle partite giocate del periodo di training
        table   : tabella feature già costruita (facoltativa, per non rifarla)
        """
        if table is None:
            table = F.build_feature_table(results)
        tab = table[table["played"] == 1].sort_values("date").reset_index(drop=True)
        tab = tab[tab["target"].notna()]
        if len(tab) < 100:
            raise ValueError(f"servono almeno 100 partite per l'addestramento, trovate {len(tab)}")

        # validazione: né troppo piccola (pesi rumorosi) né troppo grande
        # (base models affamati); tetto a 400 partite, che bastano e avanzano
        res = results.copy()
        res["date"] = pd.to_datetime(res["date"])

        n = len(tab)
        # confini dei fold: il primo blocco serve solo ad addestrare, la coda
        # (oof_frac del training) viene predetta a pezzi, sempre in avanti
        first = max(150, int(n * (1.0 - self.oof_frac)))
        n_folds = max(1, min(self.n_folds, (n - first) // 60))
        bounds = np.linspace(first, n, n_folds + 1).astype(int)
        self._log(f"  {n} partite | {n_folds} fold out-of-fold da "
                  f"{tab['date'].iloc[bounds[0]].date()}")

        # ---- iperparametri: scelti una volta sul primo blocco ----------------
        self.imputer = F.Imputer().fit(tab.iloc[:bounds[0]])
        X0 = self.imputer.transform(tab.iloc[:bounds[0]])
        y0 = tab["target"].to_numpy(dtype=float)[:bounds[0]]
        C_sel = LogisticBase(self.random_state).select(X0, y0).C
        tree_sel = TreeBase(self.random_state).select(X0, y0).params
        hparams = (C_sel, tree_sel)

        # ---- predizioni out-of-fold ------------------------------------------
        oof = {name: [] for name in BASE_NAMES}
        oof_y = []
        for i in range(n_folds):
            lo, hi = bounds[i], bounds[i + 1]
            tr_tab, te_tab = tab.iloc[:lo], tab.iloc[lo:hi]
            if len(te_tab) == 0:
                continue
            split_date = te_tab["date"].iloc[0]
            res_tr = res[res["date"] < split_date]

            imp = F.Imputer().fit(tr_tab)
            X_tr, X_te = imp.transform(tr_tab), imp.transform(te_tab)
            y_tr = tr_tab["target"].to_numpy(dtype=float)

            lr, trm, hm = self._fit_bases(res_tr, tr_tab, X_tr, y_tr,
                                          hparams=hparams, history=res_tr)
            # per predire il fold servono i risultati precedenti a ogni partita:
            # il filtraggio HMM taglia per data, quindi passare res è corretto
            hm.set_history(res)

            oof["logistic"].append(lr.predict_proba(X_te))
            oof["tree"].append(trm.predict_proba(X_te))
            oof["hmm"].append(hm.predict_proba(te_tab[["home_id", "away_id", "date"]]))
            oof_y.append(te_tab["target"].to_numpy(dtype=float))

        y_oof = np.concatenate(oof_y)
        p_oof = {k: np.concatenate(v) for k, v in oof.items()}
        self.n_oof = len(y_oof)

        # ---- calibrazione per modello + pesi, tutto su out-of-fold -----------
        self.calibrators = {}
        cal_oof = {}
        for name in BASE_NAMES:
            cal = PlattCalibrator().fit(p_oof[name], y_oof)
            self.calibrators[name] = cal
            cal_oof[name] = cal.transform(p_oof[name])
            self.oof_scores[name] = {
                "log_loss":     log_loss_safe(y_oof, p_oof[name]),
                "log_loss_cal": log_loss_safe(y_oof, cal_oof[name]),
                "slope":        round(cal.a, 3),
            }
        self.base_valid_scores = self.oof_scores      # compatibilità

        P = np.column_stack([cal_oof[n] for n in BASE_NAMES])
        self.weights = optimise_weights(P, y_oof, shrink=self.weight_shrink)
        blend = P @ self.weights
        # ultima ricalibrazione sulla miscela: assorbe la sicurezza in eccesso
        # che nasce dal fondere modelli correlati
        self.final_calibrator = PlattCalibrator().fit(blend, y_oof)

        self._log("  pesi: " + " ".join(f"{n}={w:.3f}"
                                        for n, w in zip(BASE_NAMES, self.weights)))
        self._log(f"  log-loss OOF ensemble: {log_loss_safe(y_oof, blend):.4f} "
                  f"→ ricalibrata "
                  f"{log_loss_safe(y_oof, self.final_calibrator.transform(blend)):.4f} "
                  f"(prior {log_loss_safe(y_oof, np.full(len(y_oof), y_oof.mean())):.4f})")

        # ---- rifit dei base su tutto il training -----------------------------
        self.imputer = F.Imputer().fit(tab)
        X_all = self.imputer.transform(tab)
        y_all = tab["target"].to_numpy(dtype=float)
        self.logistic, self.tree, self.hmm_model = self._fit_bases(
            res, tab, X_all, y_all, hparams=hparams, history=res)

        self.train_end = tab["date"].max()
        self.n_train = len(tab)
        return self

    # -- predizione ------------------------------------------------------------

    def prepare_history(self, results):
        """Aggiorna le sequenze note all'HMM (partite giocate fino ad oggi)."""
        res = results.copy()
        res["date"] = pd.to_datetime(res["date"])
        self.hmm_model.set_history(res)
        return self

    def predict_proba(self, table, detail=False):
        X = self.imputer.transform(table)
        meta = table[["home_id", "away_id", "date"]]
        raw, cal = self._base_probs(X, meta)
        P = np.column_stack([cal[n] for n in BASE_NAMES])
        p = P @ self.weights
        if self.final_calibrator is not None:
            p = self.final_calibrator.transform(p)
        if not detail:
            return p
        out = pd.DataFrame({f"p_{n}": cal[n] for n in BASE_NAMES})
        for n in BASE_NAMES:
            out[f"p_{n}_raw"] = raw[n]
        out["p_ensemble"] = p
        out["spread"] = out[[f"p_{n}" for n in BASE_NAMES]].max(axis=1) - \
                        out[[f"p_{n}" for n in BASE_NAMES]].min(axis=1)
        return out

    # -- diagnostica -----------------------------------------------------------

    def summary(self):
        lines = [f"Ensemble Home Win — {self.n_train} partite di training, "
                 f"fino a {pd.Timestamp(self.train_end).date()}"]
        lines.append("Pesi: " + ", ".join(f"{n} {w:.1%}" for n, w
                                          in zip(BASE_NAMES, self.weights)))
        lines.append(f"Calibrazione stimata su {self.n_oof} predizioni out-of-fold")
        for n in BASE_NAMES:
            s = self.oof_scores.get(n, {})
            if s:
                lines.append(f"  {n:9s} log-loss OOF {s['log_loss']:.4f} "
                             f"→ calibrato {s['log_loss_cal']:.4f} "
                             f"(pendenza {s.get('slope', float('nan'))})")
        if self.final_calibrator is not None:
            lines.append(f"  miscela: pendenza finale "
                         f"{self.final_calibrator.a:.3f}, "
                         f"intercetta {self.final_calibrator.b:+.3f}")
        lines.append(f"Logistica C={self.logistic.C}, albero {self.tree.params}")
        return "\n".join(lines)

    # -- persistenza -----------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        # le sequenze si ricostruiscono da all_results.csv: non vanno nel pickle
        hm = state.get("hmm_model")
        if hm is not None:
            hm = hm.__class__.__new__(hm.__class__)
            hm.__dict__.update(self.hmm_model.__dict__)
            hm.sequences = {}
            hm._paths = {}
            state["hmm_model"] = hm
        return state
