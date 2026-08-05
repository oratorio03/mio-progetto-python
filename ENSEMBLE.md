# Ensemble Home Win — Logistica + HMM + Albero

Step 9 della pipeline. Stima la probabilità di **vittoria in casa** combinando
tre modelli che guardano la stessa partita da angoli diversi, e li fonde con
pesi stimati sui dati.

Nessuna chiamata API: legge gli stessi CSV già prodotti dagli step 1–7.

```
data/{nation}/processed/all_results.csv     partite giocate    (obbligatorio)
data/{nation}/processed/all_fixtures.csv    partite future     (per predict)
data/{nation}/processed/odds.csv            quote 1X2          (facoltativo)
```

## Uso

```bash
python 9_ensemble_home.py train    --all              # addestra e salva i modelli
python 9_ensemble_home.py predict  italy              # probabilità sulle fixture
python 9_ensemble_home.py backtest --all              # walk-forward + metriche
python 9_ensemble_home.py inspect  italy              # cosa ha imparato
```

Opzioni principali:

| opzione | effetto |
|---|---|
| `--cutoff=2026-03-01` | ignora tutto da quella data in poi (simulazioni "come sarei arrivato a…") |
| `--states=3` | stati latenti dell'HMM |
| `--min-train=600` | partite minime prima di iniziare a prevedere nel backtest |
| `--step-days=30` | ampiezza della finestra prevista a ogni passo del walk-forward |
| `--min-edge=0.05` | edge minimo (`p × quota − 1`) per marcare `VALUE` |
| `--retrain` | forza il riaddestramento anche se esiste un modello salvato |

Output:

```
models/{nation}_ensemble.pkl                  modello serializzato
data/{nation}/processed/ensemble_home.csv     previsioni per nazione
output/ensemble_home.csv                      aggregato cross-nazione
output/ensemble_backtest_{nation}.csv         predizioni out-of-sample
```

## I tre modelli

**1. Regressione logistica** (L2, feature standardizzate, `C` scelto in CV
temporale). Segnale lineare e liscio su 25 feature: differenza Elo, forma in
casa/trasferta, gol fatti e subiti, riposo, scontri diretti, prior di lega.
È il modello che sbaglia "poco per volta": raramente il migliore, quasi mai
il peggiore.

**2. HMM sugli stati di forma** (`ensemble/hmm.py`). La forma di una squadra è
una variabile latente che evolve lentamente e non si osserva: si osservano solo
gli esiti. È letteralmente la struttura di un HMM.

```
stato latente   z_t ∈ {0..K-1}      transizioni  A[z_{t-1}, z_t]
osservazione    x_t ∈ {0..4}        emissione    B[z_t, campo_t, x_t]
```

L'osservazione codifica margine e segno (`KO 2+`, `KO 1`, `X`, `W 1`, `W 2+`) e
l'emissione è condizionata al campo: lo stesso stato di forma rende diversamente
in casa e fuori, che è precisamente il segnale utile per il mercato Home Win.

I parametri si stimano con **Baum-Welch** in pooling su tutte le sequenze di
tutte le squadre — descrivono *come si muove la forma nel calcio*, non una
singola squadra. Lo stato della singola squadra si ottiene poi per filtraggio
in avanti sulla sua sequenza, fermandosi rigorosamente alla partita precedente.

Per ogni partita l'HMM produce due letture indipendenti, fuse da una logistica
a due input stimata sui dati:

```
p_casa   = P(la squadra di casa vinca)   dalla sua sequenza, emissione "in casa"
p_ospite = P(l'ospite perda)             dalla sua sequenza, emissione "in trasferta"
```

Implementazione autonoma, solo numpy: forward-backward scalato, restart multipli,
smoothing di Dirichlet, stati riordinati per forza crescente (così `s0` è sempre
"in crisi" e `sK-1` "in fiducia", e `inspect` resta leggibile).

**3. Albero decisionale** (entropia, potato, `max_depth` e `min_samples_leaf` in
CV temporale). Volutamente poco profondo: serve a catturare interazioni a soglia
— *"Elo alto **e** ospite che perde sempre fuori casa"* — non a memorizzare il
campionato.

## Come vengono fusi

1. Ogni modello base viene calibrato singolarmente (Platt) su predizioni
   **out-of-fold**.
2. I pesi minimizzano la log-loss della media pesata (parametrizzazione softmax,
   Nelder-Mead), con un filo di shrinkage verso l'uniforme.
3. La miscela passa per un ultimo calibratore.

Le predizioni out-of-fold arrivano da un walk-forward interno al training: il
primo blocco addestra, la coda (metà del training, in `n_folds` pezzi) viene
predetta sempre in avanti. Poi i modelli base vengono riaddestrati su tutto il
training, mentre pesi e calibratori restano quelli stimati out-of-fold.

Questo punto non è cosmesi. Con una **singola** finestra di validazione il
calibratore viene stimato su modelli addestrati su meno dati di quelli poi messi
in produzione, e il risultato è sistematicamente sovra-sicuro: sui dati di test
la pendenza di ricalibrazione misurata era **0.44** invece di 1.0, con log-loss
*peggiore di una costante* (0.6924 contro 0.6896) nonostante un AUC di 0.56 —
il modello sapeva ordinare le partite ma sparava probabilità troppo estreme.
Passando a calibrazione out-of-fold: log-loss **0.6768**, AUC 0.588.

## Disciplina anti-leakage

Il rischio numero uno in questo tipo di modelli è far entrare il futuro nelle
feature. Le regole applicate:

- **Passata cronologica unica**: per ogni partita si emettono prima le feature
  dallo stato corrente, *poi* si aggiorna lo stato con il risultato.
- **Le fixture future non aggiornano mai lo stato**, quindi non contaminano il
  passato né si contaminano fra loro.
- **Ordinamento stabile** (`kind="mergesort"`): la riga *i* è identica che il
  file finisca a *i* o mille partite dopo. Il test lo verifica ricostruendo la
  tabella su dati troncati e confrontando riga per riga.
- **Split sempre cronologici**, mai casuali, anche nella CV degli iperparametri
  (`TimeSeriesSplit`).
- **Mediane di imputazione** stimate solo sul training e riusate identiche in
  predizione.
- **HMM**: i parametri vengono solo dal periodo di training; il filtraggio taglia
  per data (`searchsorted` su `date < data partita`), quindi per prevedere la
  giornata di domenica usa i risultati fino a sabato — come in produzione, senza
  mai vedere la partita che sta prevedendo.

## Lettura dell'output

```
nation  date        home_name  away_name  p_ensemble  p_logistic  p_hmm  p_tree  spread  q1    edge  value
italy   2026-03-14  Inter      Empoli          0.712       0.698  0.681   0.744   0.063  1.55  0.104 VALUE
```

- `p_ensemble` — la probabilità da usare.
- `p_logistic`, `p_hmm`, `p_tree` — i tre pareri, già calibrati.
- `spread` — quanto sono in disaccordo. Uno spread largo su una probabilità alta
  è il caso in cui vale la pena guardare la partita a mano.
- `edge` — `p_ensemble × q1 − 1`. `VALUE` quando supera `--min-edge`.

`backtest` stampa metriche per l'ensemble **e per ogni modello base**, così si
vede subito chi sta contribuendo, più:

- `p_prior` — la frequenza di vittorie casalinghe nota al momento
  dell'addestramento. È il confronto onesto: batterla è il minimo sindacale.
- `log_loss_baseline` — la base rate del campione di test *stesso*. È un
  **oracolo** (quel numero non lo si conosce prima delle partite): va letto come
  tetto pessimistico, non come avversario.
- tabella delle soglie, tabella di calibrazione, e ROI a puntata piatta sulle
  value bet se `odds.csv` è disponibile.

## Test

```bash
python tests/test_ensemble.py
```

Genera un campionato sintetico con forza di squadra **e** uno stato di forma
latente markoviano — cioè esattamente la struttura che l'HMM dovrebbe
recuperare — e ci fa girare sopra tutta la pipeline. Verifica fra le altre cose
che le feature non guardino il futuro, che l'HMM ritrovi stati ordinati per
forza crescente e apprenda il vantaggio del campo, che l'ensemble batta il
prior out-of-sample, e che il modello ricaricato da disco predica identico.

Non servono i CSV reali: scrive tutto in una cartella temporanea via
`BETPRO_ROOT`.

## Note pratiche

- Servono almeno **200 partite** per addestrare; sotto le ~1000 le stime di peso
  e calibrazione restano rumorose e conviene guardare il backtest prima di
  fidarsi.
- Un `train` richiede una decina di secondi per nazione; il `backtest`
  walk-forward riaddestra a ogni passo, quindi va a minuti.
- `BETPRO_ROOT` forza la root del progetto se il layout non viene rilevato da
  solo (di norma la root è la prima cartella risalendo che contiene `data/` o
  `config/`).
