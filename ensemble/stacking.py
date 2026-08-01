"""
stacking.py — combinazione dei modelli base, un mercato alla volta.

Perché combinare e non scegliere il migliore: i tre modelli sbagliano in modo
diverso. M2 è forte dove il modello di punteggio è ben specificato, M1 dove
contano interazioni e non linearità, M3 dove i dati sono pochi e serve poca
varianza. Il peso ottimale cambia da mercato a mercato, e va stimato.

Come si evita di ingannarsi da soli:

  - le probabilità su cui si stima la combinazione sono out-of-fold: ogni
    blocco temporale è predetto da modelli base addestrati solo sui blocchi
    precedenti, mai su sé stesso;
  - il calibratore isotonico è addestrato sull'ultima parte delle predizioni
    out-of-fold, che il pool non ha usato per stimare i pesi;
  - i modelli base finali vengono riaddestrati su tutto lo storico solo dopo,
    per essere usati sulle partite future.

La combinazione è un pool logaritmico a pesi vincolati (pooling.py), non una
logistica libera: con predizioni out-of-fold di qualità disomogenea una
logistica libera peggiora l'ensemble invece di migliorarlo.
"""

import numpy as np
import pandas as pd

from .calibration import make_calibrator
from .config import MARKETS, MARKET_TARGET_COLS, EnsembleConfig
from .models import default_models
from .models.base import time_decay_weights
from .pooling import LogPoolMeta, stack_components

_EPS = 1e-6

N_CLASSES = {"1x2": 3, "over25": 2, "btts": 2}


def market_probs(pred, market):
    """Estrae dalla predizione di un modello la matrice del mercato richiesto."""
    p = pred.get(market)
    if p is None:
        return None
    p = np.asarray(p, dtype=float)
    if market == "1x2":
        return p if p.ndim == 2 else None
    return p.reshape(-1)


def _component_probs(preds_by_model, market, model_names):
    out = []
    for name in model_names:
        p = market_probs(preds_by_model.get(name, {}), market)
        if p is None:
            return None, []
        out.append(p)
    return out, list(model_names)


def average_blend(preds_by_model, market):
    """Fallback: media geometrica dei modelli disponibili."""
    ps = [market_probs(preds_by_model[n], market) for n in sorted(preds_by_model)]
    ps = [p for p in ps if p is not None]
    if not ps:
        return None
    K = N_CLASSES[market]
    LP = stack_components(ps, K)
    p = np.exp(LP.mean(axis=1))
    p = p / p.sum(axis=1, keepdims=True)
    return p if market == "1x2" else p[:, 1]


class StackedEnsemble:
    """
    Ensemble a due livelli: modelli base + pool per mercato.

        ens = StackedEnsemble(cfg)
        ens.fit(df_train, feats_train)
        probs = ens.predict_proba(df_test, feats_test)   # {mercato: probabilità}
    """

    def __init__(self, cfg: EnsembleConfig = None, models=None, seed=None,
                 verbose=True):
        self.cfg = cfg or EnsembleConfig()
        self.seed = seed if seed is not None else self.cfg.seed
        self._model_factory = models
        self.models_ = {}
        self.meta_ = {}
        self.calibrators_ = {}
        self.model_names_ = []
        self.verbose = verbose
        self.fitted_ = False
        self.oof_ = None

    # ── modelli base ──────────────────────────────────────────────────────────
    def _new_models(self):
        models = (self._model_factory() if callable(self._model_factory)
                  else default_models(self.cfg, seed=self.seed))
        return {m.name: m for m in models}

    def _fit_base(self, df, feats):
        models = self._new_models()
        for m in models.values():
            m.fit(df, feats)
        return models

    @staticmethod
    def _predict_base(models, df, feats):
        return {name: m.predict_proba(df, feats) for name, m in models.items()}

    # ── predizioni out-of-fold ────────────────────────────────────────────────
    def _oof_predictions(self, df, feats):
        """
        Blocchi temporali contigui: il blocco i è predetto da modelli
        addestrati sui blocchi 0..i-1.

        I primi blocchi vengono saltati: un modello addestrato su poche
        centinaia di partite produce probabilità che non somigliano a quelle
        del modello finale, e stimare i pesi su quelle è fuorviante.
        """
        n = len(df)
        folds = max(int(self.cfg.stack.oof_folds), 2)
        min_train = max(self.cfg.stack.min_train_matches, int(0.35 * n))
        if n < min_train + 300:
            return None

        edges = np.linspace(0, n, folds + 1).astype(int)
        rows = []
        for i in range(1, folds):
            tr_end, te_start, te_end = edges[i], edges[i], edges[i + 1]
            if tr_end < min_train or te_end <= te_start:
                continue

            df_tr = df.iloc[:tr_end]
            if self.cfg.stack.embargo_days > 0:
                limit = (df.iloc[te_start]["kickoff"]
                         - pd.Timedelta(days=self.cfg.stack.embargo_days))
                df_tr = df_tr[df_tr["kickoff"] <= limit]
                if len(df_tr) < self.cfg.stack.min_train_matches:
                    continue

            df_te = df.iloc[te_start:te_end]
            if self.verbose:
                print(f"    fold {i}: train {len(df_tr):>6} → test {len(df_te):>5}",
                      flush=True)

            models = self._fit_base(df_tr, feats.loc[df_tr.index])
            preds = self._predict_base(models, df_te, feats.loc[df_te.index])
            rows.append((df_te.index.to_numpy(), preds))

        if not rows:
            return None

        names = sorted(rows[0][1])
        index = np.concatenate([idx for idx, _ in rows])
        probs = {}
        for name in names:
            for market in MARKETS:
                parts = []
                for _, preds in rows:
                    p = market_probs(preds.get(name, {}), market)
                    if p is None:
                        parts = None
                        break
                    parts.append(p)
                if parts:
                    probs[(name, market)] = np.concatenate(parts, axis=0)
        return {"index": index, "probs": probs, "models": names}

    # ── pool ──────────────────────────────────────────────────────────────────
    def _market_component(self, feats, index, market):
        """Le probabilità de-viggate del mercato come componente aggiuntiva del pool."""
        if not self.cfg.stack.use_market_features:
            return None
        cols = {"1x2": ["mkt_p_h", "mkt_p_d", "mkt_p_a"],
                "over25": ["mkt_p_o25"], "btts": ["mkt_p_btts"]}[market]
        sub = feats.reindex(index=index, columns=cols).astype(float)
        if sub.isna().all().any() or sub.notna().all(axis=1).mean() < 0.8:
            return None      # quote troppo incomplete per essere una componente stabile
        if market == "1x2":
            p = sub.to_numpy()
            filled = np.where(np.isfinite(p).all(axis=1, keepdims=True), p,
                              np.array([[0.44, 0.26, 0.30]]))
            return filled / filled.sum(axis=1, keepdims=True)
        p = sub.to_numpy().reshape(-1)
        return np.where(np.isfinite(p), p, np.nanmedian(p))

    def _components(self, preds, market, feats, index):
        comps, names = _component_probs(preds, market, self.model_names_)
        if comps is None:
            return None, []
        mkt = self._market_component(feats, index, market)
        if mkt is not None:
            comps.append(mkt)
            names = names + ["MERCATO"]
        return stack_components(comps, N_CLASSES[market]), names

    def _validated_calibrator(self, P, y, market):
        """
        Calibra solo se serve davvero.

        L'isotonica non è gratis: se il pool è già ben calibrato aggiunge
        rumore, e la mappa ha comunque una scadenza (il livello di gol e la
        quota di pareggi di una lega si spostano nel tempo, quindi una
        correzione stimata sul passato può essere sbagliata sul futuro).

        Si stima la mappa sulla prima metà del blocco di calibrazione e la si
        verifica sulla seconda. Viene accettata solo se la log loss migliora
        di almeno `calibration_min_gain` in termini relativi: un guadagno
        marginale non è distinguibile dal rumore e non giustifica il rischio.
        """
        kind = self.cfg.stack.calibration
        if kind in (None, "none"):
            return make_calibrator("none")

        half = len(y) // 2
        if half < 100 or len(np.unique(y[:half])) < 2:
            return make_calibrator("none")

        cal = make_calibrator(kind)
        cal.fit(P[:half], y[:half])
        p_cal = cal.transform(P[half:])

        idx = np.arange(len(y) - half)
        base_ll = -np.mean(np.log(np.clip(P[half:][idx, y[half:]], _EPS, None)))
        cal_ll = -np.mean(np.log(np.clip(p_cal[idx, y[half:]], _EPS, None)))
        gain = (base_ll - cal_ll) / base_ll if base_ll > 0 else 0.0

        if not np.isfinite(cal_ll) or gain < self.cfg.stack.calibration_min_gain:
            if self.verbose:
                print(f"    calibrazione {market}: guadagno {gain * 100:+.1f}% "
                      f"sotto soglia → disattivata")
            return make_calibrator("none")

        final = make_calibrator(kind)
        final.fit(P, y)
        if self.verbose:
            print(f"    calibrazione {market}: log loss {base_ll:.4f} → {cal_ll:.4f} "
                  f"({gain * 100:+.1f}%) → attiva")
        return final

    def _fit_meta(self, df, feats, oof):
        index = oof["index"]
        df_oof = df.loc[index]
        preds = {name: {m: oof["probs"][(name, m)]
                        for m in MARKETS if (name, m) in oof["probs"]}
                 for name in oof["models"]}

        for market in MARKETS:
            y = df_oof[MARKET_TARGET_COLS[market]].to_numpy(dtype=float)
            LP, names = self._components(preds, market, feats, index)
            if LP is None:
                continue
            mask = np.isfinite(y) & np.isfinite(LP).all(axis=(1, 2))
            if mask.sum() < 300 or len(np.unique(y[mask])) < 2:
                continue

            LPf, yf = LP[mask], y[mask].astype(int)
            w = time_decay_weights(df_oof["kickoff"].to_numpy()[mask],
                                   half_life_days=720.0)

            # coda finale riservata alla calibrazione: il pool non la vede
            cut = int(len(yf) * 0.75)
            calibrator = make_calibrator("none")
            if len(yf) - cut >= 200:
                pool_cal = LogPoolMeta(l2=self.cfg.stack.pool_l2)
                pool_cal.fit(LPf[:cut], yf[:cut], sample_weight=w[:cut])
                calibrator = self._validated_calibrator(
                    pool_cal.predict_proba(LPf[cut:]), yf[cut:], market)

            pool = LogPoolMeta(l2=self.cfg.stack.pool_l2)
            pool.fit(LPf, yf, sample_weight=w)
            if not pool.fitted_:
                continue

            self.meta_[market] = {"pool": pool, "names": names}
            self.calibrators_[market] = calibrator
            if self.verbose:
                pesi = ", ".join(f"{n}={v:.2f}" for n, v in zip(names, pool.weights))
                print(f"    pesi {market:<7} {pesi}")

    # ── API ───────────────────────────────────────────────────────────────────
    def fit(self, df, feats):
        """Addestra modelli base, pool e calibratori su `df` (solo passato)."""
        d = df[df[MARKET_TARGET_COLS["1x2"]].notna()]
        f = feats.loc[d.index]

        if self.verbose:
            print(f"  Stacking: {len(d):,} partite di training")

        # i nomi dei modelli servono a mantenere l'ordine delle componenti
        self.model_names_ = sorted(self._new_models())

        oof = self._oof_predictions(d, f)
        self.oof_ = oof
        if oof is not None:
            self._fit_meta(d, f, oof)
        elif self.verbose:
            print("  [stacking] storico insufficiente per stimare i pesi → "
                  "media geometrica")

        self.models_ = self._fit_base(d, f)
        self.fitted_ = True
        return self

    def predict_base(self, df, feats):
        if not self.fitted_:
            raise RuntimeError("StackedEnsemble non addestrato")
        return self._predict_base(self.models_, df, feats)

    def predict_proba(self, df, feats, return_base=False):
        """Probabilità finali calibrate per i tre mercati."""
        preds = self.predict_base(df, feats)

        out = {}
        for market in MARKETS:
            meta = self.meta_.get(market)
            if meta is None:
                out[market] = average_blend(preds, market)
                continue
            LP, _ = self._components(preds, market, feats, df.index)
            if LP is None:
                out[market] = average_blend(preds, market)
                continue
            p = meta["pool"].predict_proba(LP)
            cal = self.calibrators_.get(market)
            if cal is not None:
                p = cal.transform(p)
            out[market] = p if market == "1x2" else p[:, 1]

        if return_base:
            return out, preds
        return out

    def weights_report(self):
        """Pesi stimati per mercato — la diagnostica più utile dell'ensemble."""
        rows = []
        for market, meta in self.meta_.items():
            for name, w in zip(meta["names"], meta["pool"].weights):
                rows.append({"mercato": market, "componente": name, "peso": round(float(w), 3)})
        return pd.DataFrame(rows)
