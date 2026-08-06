"""
ensemble.py — La combinazione dei tre modelli.

    logistica  segnale lineare su forza, forma, riposo, scontri diretti
    HMM        stato di forma latente, con emissioni diverse in casa e fuori
    albero     interazioni a soglia fra le stesse feature

Ogni modello viene calibrato singolarmente, poi i tre output vengono fusi con
pesi che minimizzano la log-loss, e la miscela passa per un'ultima
calibrazione. Nessun peso deciso a mano.

Il punto delicato è DOVE si stimano calibratori e pesi. Con una singola
finestra di validazione il calibratore viene stimato su modelli addestrati su
meno dati di quelli poi messi in produzione, e il risultato è
sistematicamente sovra-sicuro: misurato su dati di prova, pendenza di
ricalibrazione 0.44 invece di 1.0 e log-loss peggiore di una costante
(0.6924 contro 0.6896) nonostante un AUC di 0.56 — il modello ordinava bene
le partite ma sparava numeri troppo estremi.

Qui si usano invece predizioni OUT-OF-FOLD generate da un walk-forward interno
al training: il primo blocco addestra, la coda viene predetta a pezzi, sempre
in avanti. Sono molte di più e vengono da modelli addestrati su volumi
confrontabili con quello finale. Stessi dati di prova: log-loss 0.6768.
"""

import numpy as np
import pandas as pd

from . import features as F
from .base_models import LogisticModel, TreeModel
from .hmm import HMMHomeWin
from .util import PlattCalibrator, log_loss, optimise_weights

MODELS = ("logistica", "hmm", "albero")


class HomeWinEnsemble:
    """
        ens = HomeWinEnsemble().fit(partite_giocate)
        ens.set_history(partite_giocate)        # sequenze note all'HMM
        p = ens.predict_proba(tabella_feature)
    """

    def __init__(self, n_states=3, oof_frac=0.5, n_folds=3, weight_shrink=0.15,
                 random_state=42, verbose=True):
        self.n_states      = n_states
        self.oof_frac      = oof_frac      # coda del training usata out-of-fold
        self.n_folds       = n_folds
        self.weight_shrink = weight_shrink
        self.random_state  = random_state
        self.verbose       = verbose

        self.imputer     = None
        self.logistica   = None
        self.albero      = None
        self.hmm         = None
        self.calibrators = {}
        self.final_calibrator = None
        self.weights     = None
        self.oof_scores  = {}
        self.n_train = self.n_oof = 0
        self.train_end = None

    # -- interni ---------------------------------------------------------------

    def _log(self, message):
        if self.verbose:
            print(message)

    def _fit_bases(self, played, table, X, y, hyper, history=None):
        logistica = LogisticModel(self.random_state, C=hyper[0]).fit(X, y)
        albero    = TreeModel(self.random_state, params=hyper[1]).fit(X, y)
        hmm = HMMHomeWin(n_states=self.n_states,
                         random_state=self.random_state).fit(
            played, table, y, history=history if history is not None else played)
        return logistica, albero, hmm

    def _probabilities(self, table):
        X = self.imputer.transform(table)
        raw = {
            "logistica": self.logistica.predict_proba(X),
            "hmm":       self.hmm.predict_proba(table),
            "albero":    self.albero.predict_proba(X),
        }
        calibrated = {name: self.calibrators[name].transform(p)
                      if name in self.calibrators else p
                      for name, p in raw.items()}
        return raw, calibrated

    # -- addestramento ---------------------------------------------------------

    def fit(self, played, table=None):
        """
        played : partite giocate del periodo di training
        table  : tabella feature già costruita (facoltativa, per non rifarla)
        """
        if table is None:
            table = F.build(played)
        table = (table[(table["played"] == 1) & table["target"].notna()]
                 .sort_values("date", kind="mergesort").reset_index(drop=True))
        if len(table) < 100:
            raise ValueError(
                f"servono almeno 100 partite con risultato, trovate {len(table)}")

        played = played.copy()
        played["date"] = pd.to_datetime(played["date"])

        n = len(table)
        first = max(150, int(n * (1.0 - self.oof_frac)))
        n_folds = max(1, min(self.n_folds, (n - first) // 60))
        bounds = np.linspace(first, n, n_folds + 1).astype(int)
        self._log(f"  {n} partite, {n_folds} fold out-of-fold da "
                  f"{table['date'].iloc[bounds[0]].date()}")

        # ---- iperparametri, scelti una volta sul primo blocco ----------------
        head = table.iloc[:bounds[0]]
        X_head = F.Imputer().fit_transform(head)
        y_head = head["target"].to_numpy(dtype=float)
        hyper = (LogisticModel(self.random_state).select(X_head, y_head).C,
                 TreeModel(self.random_state).select(X_head, y_head).params)

        # ---- predizioni out-of-fold ------------------------------------------
        oof = {name: [] for name in MODELS}
        oof_y = []
        for i in range(n_folds):
            lo, hi = bounds[i], bounds[i + 1]
            train, test = table.iloc[:lo], table.iloc[lo:hi]
            if test.empty:
                continue
            past = played[played["date"] < test["date"].iloc[0]]

            imputer = F.Imputer().fit(train)
            logistica, albero, hmm = self._fit_bases(
                past, train, imputer.transform(train),
                train["target"].to_numpy(dtype=float), hyper, history=past)
            # Il filtraggio HMM taglia per data, quindi passare tutte le partite
            # è corretto: per prevedere la giornata di domenica userà i risultati
            # fino a sabato, esattamente come le feature rolling e come accadrà
            # in produzione. I PARAMETRI però vengono solo da `past`.
            hmm.set_history(played)

            X_test = imputer.transform(test)
            oof["logistica"].append(logistica.predict_proba(X_test))
            oof["albero"].append(albero.predict_proba(X_test))
            oof["hmm"].append(hmm.predict_proba(test))
            oof_y.append(test["target"].to_numpy(dtype=float))

        y_oof = np.concatenate(oof_y)
        p_oof = {name: np.concatenate(values) for name, values in oof.items()}
        self.n_oof = len(y_oof)

        # ---- calibrazione e pesi, solo su out-of-fold ------------------------
        self.calibrators = {}
        calibrated = {}
        for name in MODELS:
            calibrator = PlattCalibrator().fit(p_oof[name], y_oof)
            self.calibrators[name] = calibrator
            calibrated[name] = calibrator.transform(p_oof[name])
            self.oof_scores[name] = {
                "log_loss":     log_loss(y_oof, p_oof[name]),
                "log_loss_cal": log_loss(y_oof, calibrated[name]),
                "pendenza":     round(calibrator.a, 3),
            }

        P = np.column_stack([calibrated[name] for name in MODELS])
        self.weights = optimise_weights(P, y_oof, shrink=self.weight_shrink)
        blend = P @ self.weights
        # ultima ricalibrazione della miscela: assorbe la sicurezza in eccesso
        # che nasce dal fondere modelli correlati fra loro
        self.final_calibrator = PlattCalibrator().fit(blend, y_oof)

        self._log("  pesi: " + "  ".join(
            f"{name} {w:.1%}" for name, w in zip(MODELS, self.weights)))
        self._log(
            f"  log-loss out-of-fold: {log_loss(y_oof, blend):.4f} → ricalibrata "
            f"{log_loss(y_oof, self.final_calibrator.transform(blend)):.4f} "
            f"(costante: {log_loss(y_oof, np.full(len(y_oof), y_oof.mean())):.4f})")

        # ---- rifit dei modelli base su tutto il training ---------------------
        self.imputer = F.Imputer().fit(table)
        X_all = self.imputer.transform(table)
        y_all = table["target"].to_numpy(dtype=float)
        self.logistica, self.albero, self.hmm = self._fit_bases(
            played, table, X_all, y_all, hyper, history=played)

        # ---- il livello lo detta il training, non la finestra out-of-fold ----
        raw_train, _ = self._probabilities(table)
        base_rate = float(y_all.mean())
        for name in MODELS:
            self.calibrators[name].recenter(raw_train[name], base_rate)
        _, calibrated_train = self._probabilities(table)
        blend_train = np.column_stack(
            [calibrated_train[name] for name in MODELS]) @ self.weights
        self.final_calibrator.recenter(blend_train, base_rate)
        self._log(f"  livello riportato alla frequenza del training "
                  f"({base_rate:.1%})")

        self.n_train = len(table)
        self.train_end = table["date"].max()
        return self

    # -- predizione ------------------------------------------------------------

    def set_history(self, played):
        """Aggiorna le partite note all'HMM (tutti i risultati fino a oggi)."""
        played = played.copy()
        played["date"] = pd.to_datetime(played["date"])
        self.hmm.set_history(played)
        return self

    def predict_proba(self, table, detail=False):
        raw, calibrated = self._probabilities(table)
        P = np.column_stack([calibrated[name] for name in MODELS])
        p = P @ self.weights
        if self.final_calibrator is not None:
            p = self.final_calibrator.transform(p)
        if not detail:
            return p

        out = pd.DataFrame({f"p_{name}": calibrated[name] for name in MODELS})
        for name in MODELS:
            out[f"p_{name}_grezza"] = raw[name]
        out["p_ensemble"] = p
        columns = [f"p_{name}" for name in MODELS]
        out["disaccordo"] = out[columns].max(axis=1) - out[columns].min(axis=1)
        return out

    # -- lettura ---------------------------------------------------------------

    def summary(self):
        lines = [f"Ensemble vittoria in casa — {self.n_train} partite di "
                 f"training, fino al {pd.Timestamp(self.train_end).date()}",
                 "Pesi: " + ", ".join(f"{name} {w:.1%}"
                                      for name, w in zip(MODELS, self.weights)),
                 f"Calibrazione stimata su {self.n_oof} predizioni out-of-fold:"]
        for name in MODELS:
            score = self.oof_scores.get(name, {})
            if score:
                lines.append(f"  {name:10s} log-loss {score['log_loss']:.4f} → "
                             f"calibrata {score['log_loss_cal']:.4f} "
                             f"(pendenza {score['pendenza']})")
        if self.final_calibrator is not None:
            lines.append(f"  miscela    pendenza {self.final_calibrator.a:.3f}, "
                         f"intercetta {self.final_calibrator.b:+.3f}")
        lines.append(f"Logistica C={self.logistica.C} · albero {self.albero.params}")
        return "\n".join(lines)

    def __getstate__(self):
        """Le sequenze si ricostruiscono dal CSV: non finiscono nel modello salvato."""
        state = self.__dict__.copy()
        if self.hmm is not None:
            light = HMMHomeWin.__new__(HMMHomeWin)
            light.__dict__.update(self.hmm.__dict__)
            light.sequences = {}
            light._paths = {}
            state["hmm"] = light
        return state
