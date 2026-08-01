"""
config.py — parametri e costanti dell'ensemble.

Tutti i valori sono raccolti qui per essere modificabili senza toccare la logica.
I default sono quelli documentati in docs/ENSEMBLE.md.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path

# ── Path ──────────────────────────────────────────────────────────────────────
# La root del progetto è la cartella che contiene il pacchetto ensemble/
ROOT_DIR   = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR  = ROOT_DIR / "models_ensemble"

# ── Mercati ───────────────────────────────────────────────────────────────────
# "1x2" è multiclasse (H/D/A), gli altri sono binari.
MARKETS      = ["1x2", "over25", "btts"]
OUTCOME_1X2  = ["H", "D", "A"]

# Colonna quota di riferimento per ogni mercato (nel file odds.csv del progetto)
MARKET_ODDS_COLS = {
    "1x2":    ["q1", "qx", "q2"],
    "over25": ["odd_o25"],
    "btts":   ["odd_btts"],
}

# Colonna target per ogni mercato (creata da data.add_targets)
MARKET_TARGET_COLS = {
    "1x2":    "y_1x2",
    "over25": "y_over25",
    "btts":   "y_btts",
}


@dataclass
class RatingConfig:
    """Iperparametri dei rating dinamici (aggiornati online, sempre pre-match)."""
    # Elo
    elo_start:        float = 1500.0
    elo_k:            float = 20.0
    elo_hfa:          float = 60.0      # vantaggio casa iniziale, in punti Elo
    elo_hfa_lr:       float = 0.02      # aggiornamento online del HFA per lega
    elo_season_regress: float = 0.25    # ritorno verso la media a inizio stagione
    elo_tier_step:    float = 60.0      # penalità Elo iniziale per tier di lega

    # pi-ratings (Constantinou & Fenton, 2013)
    pi_lambda:        float = 0.035     # learning rate rating del venue giocato
    pi_gamma:         float = 0.7       # propagazione sull'altro venue
    pi_c:             float = 3.0       # costante della funzione psi

    # GAP ratings (attack/defence su gol, alla Wheatcroft)
    gap_lr:           float = 0.06      # learning rate
    gap_season_regress: float = 0.30    # ritorno verso la media lega a inizio stagione
    gap_xg_weight:    float = 0.60      # peso dell'xG (se presente) nell'aggiornamento

    # Forma / rolling
    form_window:      int   = 6         # ultime N partite per la forma
    congestion_days:  int   = 14        # finestra per il conteggio partite recenti

    # Priors di lega
    league_prior_matches: int = 40      # peso dello shrinkage verso la media globale


@dataclass
class ModelConfig:
    """Iperparametri dei tre modelli base."""
    # M1 — gradient boosting
    m1_backend:        str   = "auto"   # auto | catboost | xgboost | lightgbm | sklearn
    m1_n_estimators:   int   = 400
    m1_learning_rate:  float = 0.05
    m1_max_depth:      int   = 4
    m1_min_samples:    int   = 40
    m1_subsample:      float = 0.85
    m1_l2:             float = 1.0

    # M2 — Poisson gerarchico con shrinkage bayesiano + correzione Dixon-Coles
    m2_half_life_days: float = 240.0    # emivita del decadimento temporale
    m2_alpha:          float = 0.35     # forza del prior gaussiano (ridge) su attacco/difesa
    m2_max_goals:      int   = 12
    m2_rho_grid:       tuple = (-0.20, -0.15, -0.10, -0.05, 0.0, 0.05)
    m2_min_matches:    int   = 200      # sotto questa soglia usa solo i rating

    # M3 — GAP/xG + logistica calibrata
    m3_c:              float = 1.0      # inverso della regolarizzazione L2
    m3_half_life_days: float = 540.0


@dataclass
class StackConfig:
    """Stacking per-mercato e calibrazione."""
    pool_l2:           float = 0.02     # attrazione dei pesi verso la media uniforme
    use_market_features: bool = False   # se True il pool include anche le quote
    calibration:       str   = "isotonic"   # isotonic | sigmoid | none
    calibration_min_gain: float = 0.03  # guadagno relativo minimo di log loss per accettarla
    min_train_matches: int   = 800      # partite minime prima di poter addestrare
    embargo_days:      int   = 0        # giorni di stacco fra train e test (anti-leak)
    refit_days:        int   = 28       # ogni quanti giorni si riaddestra (walk-forward)
    oof_folds:         int   = 5        # fold temporali per le predizioni out-of-fold


@dataclass
class BettingConfig:
    """Filtro value, staking e gestione bankroll."""
    devig_method:      str   = "shin"   # shin | multiplicative | power
    min_edge:          float = 0.03     # edge minimo sulla quota lorda
    min_prob:          float = 0.05     # probabilità minima del modello
    max_odds:          float = 8.0      # quote oltre le quali non si gioca
    min_odds:          float = 1.35
    kelly_fraction:    float = 0.25     # Kelly frazionario
    max_stake_pct:     float = 0.02     # cap per singola giocata (% bankroll)
    bankroll:          float = 1000.0
    max_bets_per_day:  int   = 25       # cap operativo sul numero di giocate


@dataclass
class EnsembleConfig:
    """Configurazione completa dell'ensemble."""
    ratings:  RatingConfig  = field(default_factory=RatingConfig)
    models:   ModelConfig   = field(default_factory=ModelConfig)
    stack:    StackConfig   = field(default_factory=StackConfig)
    betting:  BettingConfig = field(default_factory=BettingConfig)
    seed:     int           = 42

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        return cls(
            ratings=RatingConfig(**d.get("ratings", {})),
            models=ModelConfig(**d.get("models", {})),
            stack=StackConfig(**d.get("stack", {})),
            betting=BettingConfig(**d.get("betting", {})),
            seed=d.get("seed", 42),
        )
