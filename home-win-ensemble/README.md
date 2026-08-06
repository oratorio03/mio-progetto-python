# Home Win Ensemble

Probabilità di **vittoria in casa** nel calcio, stimata combinando tre modelli:

1. **Regressione logistica** — segnale lineare su forza, forma, riposo, scontri diretti
2. **HMM sugli stati di forma** — la dinamica temporale che gli altri due non vedono
3. **Albero decisionale** — interazioni a soglia fra le stesse feature

Progetto autonomo: dipende solo da numpy, pandas, scikit-learn e scipy, prende
in ingresso un CSV di partite qualunque e funziona anche se non hai ancora dati.

## Prova subito, senza dati

```bash
pip install -r requirements.txt
python -m hwe demo
```

`demo` genera un campionato sintetico, mostra come legge il file, addestra,
stampa cosa ha imparato l'HMM, prevede le partite da giocare e chiude con un
backtest walk-forward. È il modo più rapido per capire cosa fa il progetto.

## Con i tuoi dati

```bash
python -m hwe schema   --data partite.csv     # come viene letto il file
python -m hwe train    --data partite.csv
python -m hwe predict  --data partite.csv
python -m hwe backtest --data partite.csv
python -m hwe inspect                          # cosa ha imparato
```

### Il CSV

Servono tre colonne: **data, squadra di casa, squadra ospite**. Se ci sono
anche i gol, la riga è un risultato; se i gol sono vuoti, è una partita da
prevedere. Un unico file può contenere entrambe le cose.

I nomi delle colonne vengono riconosciuti da soli: funzionano i formati di
football-data.co.uk (`Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `B365H`),
gli export di API (`date`, `home_team`, `away_score`…) e i nomi italiani
(`Data`, `Squadra casa`, `Gol trasferta`…). Con `schema` vedi come è stato
interpretato il file:

```
Schema riconosciuto in partite.csv:
  date        ← Date
  home        ← HomeTeam
  away        ← AwayTeam
  home_goals  ← FTHG
  away_goals  ← FTAG
  odds_home   ← B365H
```

Se un nome non viene riconosciuto lo si forza a mano, e l'errore dice già come:

```bash
python -m hwe predict --data partite.csv \
    --map date=Quando --map home=Ospitante --map away=Ospite
```

Colonne facoltative ma utili: `league` (i rating Elo sono separati per
campionato), `season`, e le quote 1X2 — con quelle il backtest calcola anche
edge e ROI.

### Cosa esce

```
date        league  home       away      p_ensemble  p_logistica  p_hmm  p_albero  disaccordo  odds_home  edge   value
2026-03-14  E0      Inter      Empoli         0.712        0.698  0.681     0.744       0.063       1.55  0.104  VALUE
```

- `p_ensemble` — la probabilità da usare
- `p_logistica`, `p_hmm`, `p_albero` — i tre pareri, già calibrati
- `disaccordo` — quanto i tre sono in disaccordo; se è largo su una probabilità
  alta, è la partita che vale la pena guardare a mano
- `edge` — `p_ensemble × quota − 1`, marcato `VALUE` sopra `--min-edge`

## Come funziona

### L'HMM, in breve

La forma di una squadra evolve lentamente e non si osserva: si osservano solo
gli esiti. È letteralmente la struttura di un modello di Markov a stati nascosti.

```
stato latente   z_t ∈ {0..K-1}     transizioni  A[z_{t-1}, z_t]
osservazione    x_t ∈ {0..4}       emissione    B[z_t, campo_t, x_t]
```

L'osservazione tiene conto del margine (`KO 2+`, `KO 1`, `X`, `W 1`, `W 2+`) e
l'emissione è condizionata al campo: lo stesso stato di forma rende
diversamente in casa e fuori, ed è precisamente il segnale che serve per questo
mercato. I parametri si stimano con **Baum-Welch** in pooling su tutte le
squadre — descrivono come si muove la forma nel calcio, non una singola
squadra — e lo stato della singola squadra si ricava filtrando in avanti la
sua sequenza, fermandosi alla partita precedente.

Ogni partita riceve due letture indipendenti, fuse da una logistica a due
input stimata sui dati: *quanto vince in casa questa squadra di casa* e
*quanto perde in trasferta questo ospite*.

`inspect` mostra cosa ha imparato:

```
HMM — cosa emette ogni stato, per campo:
 stato     campo  KO 2+  KO 1      X   W 1  W 2+
     0      casa  0.194 0.186  0.286 0.200 0.134
     0 trasferta  0.402 0.232  0.197 0.081 0.088
     2      casa  0.152 0.093  0.073 0.343 0.340
     2 trasferta  0.207 0.091  0.128 0.281 0.293
```

### La combinazione

Ogni modello viene calibrato singolarmente, i tre output vengono fusi con pesi
che minimizzano la log-loss, e la miscela passa per un'ultima calibrazione.
Nessun peso deciso a mano.

Il punto delicato è **dove** si stimano calibratori e pesi, e ci sono due
trappole distinte, entrambe trovate misurando:

**La pendenza va stimata fuori campione.** Con una singola finestra di
validazione il calibratore viene stimato su modelli addestrati su meno dati di
quelli poi usati davvero, e il risultato è sistematicamente sovra-sicuro:
pendenza di ricalibrazione 0.44 invece di 1.0, log-loss *peggiore di una
costante* (0.6924 contro 0.6896) nonostante un AUC di 0.56 — il modello
ordinava bene le partite ma sparava numeri troppo estremi. Con predizioni
out-of-fold (un walk-forward interno al training): 0.6768.

**Il livello no.** Un modello con intercetta prevede già, sul proprio training,
una media pari alla frequenza osservata. Se la finestra out-of-fold ha avuto
una frequenza di vittorie diversa dal training nel suo complesso, il
calibratore si porta dietro quello scarto: media prevista 0.512 dove il modello
grezzo diceva 0.467, con log-loss peggiore del semplice prior. Perciò la
pendenza viene dall'out-of-fold e **l'intercetta viene riportata sulla
frequenza del training completo**. Sullo stesso hold-out: da 0.6920 a 0.6792,
meglio sia del prior sia della costante.

### Niente informazione dal futuro

È il rischio numero uno in questo tipo di modelli, e qui è trattato come tale:

- **Una sola passata cronologica**: per ogni partita si emettono prima le
  feature dallo stato corrente, *poi* si aggiorna lo stato con il risultato.
- **Le partite da giocare non aggiornano mai lo stato**, quindi non
  contaminano il passato né si contaminano fra loro.
- **Ordinamento stabile e deterministico** ovunque: la riga *i* è identica che
  il file finisca a *i* o mille partite dopo — c'è un test che lo verifica
  ricostruendo la tabella su dati troncati e confrontandola riga per riga.
- **Split sempre cronologici**, mai casuali, anche nella scelta degli
  iperparametri (`TimeSeriesSplit`).
- **Mediane di imputazione** stimate sul solo training.
- **Feature limitate**: niente conteggi che crescono all'infinito, che sarebbero
  trend temporali travestiti (le partite viste finora sono saturate a 40).
- **HMM**: i parametri vengono solo dal periodo di training, e il filtraggio
  taglia per data — per prevedere la domenica usa i risultati fino al sabato.

### Backtest

`backtest` è walk-forward puro: a ogni passo il modello viene **riaddestrato**
solo sulle partite precedenti alla finestra da prevedere. Stampa le metriche
per l'ensemble e per ogni modello base separatamente, così si vede chi sta
contribuendo, più due riferimenti:

- `p_prior` — la frequenza di vittorie casalinghe nota al momento
  dell'addestramento. È il confronto onesto: batterla è il minimo sindacale.
- `log_loss_cost` — la costante calcolata sul campione di test *stesso*. È un
  **oracolo**, quel numero non lo si conosce prima delle partite: va letto come
  tetto pessimistico, non come avversario.

Poi tabella delle soglie, tabella di calibrazione, e se ci sono le quote il ROI
a puntata piatta sulle value bet.

## Test

```bash
pip install -r requirements-dev.txt
pytest -q                  # la demo completa è marcata slow
```

I test girano su un campionato sintetico che ha una struttura nota: forza di
attacco e difesa stabili **più** uno stato di forma latente che evolve come
catena di Markov. È esattamente ciò che l'HMM dovrebbe recuperare, quindi si
può verificare che lo recuperi davvero — stati ordinati da "in crisi" a "in
fiducia", transizioni persistenti, vantaggio del campo appreso da solo — oltre
alle proprietà che contano davvero: feature che non guardano il futuro,
monotonia di EM, filtraggio che ignora la partita che sta prevedendo,
probabilità che battono il prior fuori campione.

## Struttura

```
hwe/
  schema.py       riconoscimento delle colonne del CSV
  data.py         caricamento, pulizia, partite giocate/da giocare
  features.py     passata cronologica, Elo, forma, riposo, scontri diretti
  hmm.py          HMM discreto: Baum-Welch, filtraggio, classificatore
  base_models.py  logistica e albero
  ensemble.py     calibrazione out-of-fold, pesi, combinazione
  evaluation.py   metriche, walk-forward, value bet
  synthetic.py    generatore di campionati finti
  cli.py          riga di comando
tests/            un file per modulo
```

## Limiti da tenere presenti

- Servono almeno **100 partite** per addestrare, ma sotto le ~1000 le stime di
  pesi e calibrazione restano rumorose: guarda il backtest prima di fidarti.
- Il modello dà probabilità, non certezze. Con dati sintetici realistici batte
  il prior di poco, come è normale in questo dominio: le quote dei bookmaker
  sono un avversario duro, e il `backtest` le mette in tabella apposta perché
  il confronto sia sotto gli occhi.
- Un `train` richiede una decina di secondi; il `backtest` riaddestra a ogni
  passo, quindi va a minuti.
