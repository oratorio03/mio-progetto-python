# Ensemble M1/M2/M3 — alternativa a Poisson/Dixon-Coles

Documento di riferimento del pacchetto `ensemble/`. Spiega **cosa** è stato
implementato, **perché** quelle scelte e **cosa non è stato verificato**.

Le percentuali e i risultati citati come "letteratura" vengono dalla ricerca
che ha motivato questo lavoro: sono il motivo per cui l'architettura è questa,
non risultati misurati su questo repository. I numeri misurati qui — quelli
prodotti da `demo`, `backtest` e dai test — sono sempre indicati come tali.

---

## 1. Perché non basta Poisson/Dixon-Coles

La pipeline attuale (`8_predict.py`) stima due lambda e ne deriva tutti i
mercati. È un buon modello, con limiti noti e documentati:

| Limite | Effetto pratico |
|---|---|
| Indipendenza fra i due punteggi | Sottostima i pareggi bassi (0-0, 1-1) e la correlazione fra le due squadre. È il problema che Dixon-Coles corregge con `tau`, ma solo sui quattro risultati più bassi. |
| Varianza vincolata (media = varianza) | Il calcio è leggermente sotto-disperso rispetto al Poisson: le code (4-0, 5-1) sono sovrastimate, e i mercati Over alti ne risentono. |
| Un solo canale informativo | Tutto passa dai gol segnati. Tiri, xG, possesso, riposo, congestione di calendario, forza degli avversari affrontati non entrano se non indirettamente. |
| Squadre nuove | Neopromosse e primo turno di stagione non hanno parametri stimabili: si ricade su medie di lega. |
| Coerenza forzata fra mercati | 1X2, Over 2.5 e BTTS derivano dalla stessa matrice dei punteggi. Se i lambda sbagliano, sbagliano tutti e tre nello stesso verso. |
| Struttura fissa | La relazione fra forza relativa e probabilità è imposta dalla forma funzionale, non appresa. |

Le alternative puramente statistiche che affrontano il primo e il secondo
limite sono note e valide:

- **PARX** (Poisson Autoregressive with eXogenous covariates) — l'intensità
  segue un processo autoregressivo alimentato da covariate esterne. Coglie la
  dinamica della forma meglio del decadimento esponenziale.
- **Weibull count / COM-Poisson** — distribuzioni di conteggio con parametro
  di dispersione libero; risolvono il vincolo media = varianza e migliorano
  soprattutto i mercati sui totali.
- **Modelli state-space (Koopman & Lit)** — attacco e difesa come variabili
  latenti che evolvono nel tempo, stimate con filtro di Kalman. È l'approccio
  più elegante e restituisce intervalli di incertezza, al costo di una stima
  molto più pesante.

Tutte restano però modelli di conteggio a canale singolo. Il salto di qualità
misurato in letteratura arriva dal cambiare famiglia di modelli e dal
combinarli — che è quello che fa questo pacchetto.

---

## 2. I tre componenti

### M1 — Gradient boosting (`ensemble/models/m1_gbm.py`)

Alberi in boosting su rating (pi-ratings, Elo, GAP), forma, riposo,
congestione e contesto di lega. Tre teste separate: 1X2 multiclasse,
Over 2.5 e BTTS binarie.

- **Perché**: è l'approccio con il miglior RPS pubblicato sul 1X2 (≈0.1925,
  accuratezza ≈55.8% nella Soccer Prediction Challenge). Cattura interazioni
  che un modello di conteggio non può esprimere: per esempio che il valore di
  una differenza di rating cambia con il livello di gol atteso della lega.
- **Backend**: CatBoost → XGBoost → LightGBM → scikit-learn, il primo
  disponibile. Nessuna dipendenza obbligatoria oltre a scikit-learn.
- **Numero di alberi**: scelto con early stopping su una coda temporale del
  training, poi il modello viene riaddestrato su tutti i dati con quel numero.
  Senza early stopping, in test su ~6.000 partite l'RPS peggiorava di ~0.02
  (0.2338 → 0.2126): con storici piccoli il boosting sovradatta in fretta.

### M2 — Bayesiano ibrido con rating dinamici (`ensemble/models/m2_bayes.py`)

Regressione di Poisson su effetti attacco/difesa per squadra **più** covariate
di rating continue, con penalizzazione L2 (equivalente a un prior gaussiano
centrato sulla media di lega) e correzione Dixon-Coles sui punteggi bassi con
`rho` stimato per massima verosimiglianza pesata.

- **Perché**: è la parte "Dolores" dell'architettura. Lo shrinkage bayesiano
  evita stime rumorose per le squadre con pochi dati; le covariate di rating
  fanno da ponte per le squadre **mai viste** — senza dummy attive il modello
  ricade sui rating e resta sensato, mentre un Dixon-Coles puro non ha nulla
  da dire su una neopromossa.
- Restituisce anche i lambda, direttamente confrontabili con quelli della
  pipeline attuale (`expected_goals`).

### M3 — GAP/xG + logistica calibrata (`ensemble/models/m3_gap.py`)

Rating GAP (attacco/difesa per venue, aggiornati sui gol o sull'xG quando
disponibile) dati in pasto a una regressione logistica che stima
**direttamente** l'evento, senza passare da una distribuzione di punteggi.
Calibrazione isotonica interna, addestrata su una coda temporale del training
che la logistica non ha visto.

- **Perché**: è l'approccio con la traccia di profitto di lungo periodo più
  solida in letteratura su Over 2.5 (≈ +0.8% per giocata su 12 anni) e Home
  Win, proprio perché stima l'evento e non il punteggio. È anche il componente
  a varianza più bassa: quando il boosting sovradatta su leghe con pochi dati,
  M3 tiene.

### Riferimento — Dixon-Coles (`ensemble/models/baseline.py`)

Lo stesso Poisson gerarchico di M2 senza covariate di rating e con shrinkage
minimo: è il modello della pipeline attuale, incluso nel backtest per
rispondere alla domanda "l'ensemble guadagna davvero qualcosa?".

---

## 3. Rating dinamici (`ensemble/ratings.py`)

Tre sistemi aggiornati in un unico passaggio cronologico:

| Sistema | Cosa cattura |
|---|---|
| **Elo** | Forza complessiva, con vantaggio casa stimato online **per lega** (non un valore globale) e ritorno parziale verso la media a inizio stagione. |
| **pi-ratings** | Rating separati casa/trasferta aggiornati sull'errore di differenza reti (Constantinou & Fenton). Coglie le squadre strutturalmente diverse in casa e fuori. |
| **GAP** | Attacco e difesa per venue, in unità di gol attesi. È la base di M3 e l'input più informativo per i mercati sui totali. |

**Regola non negoziabile**: per ogni partita le feature sono lo stato dei
rating *prima* del calcio d'inizio; l'aggiornamento avviene dopo aver emesso
la riga, e solo per le partite effettivamente giocate. Le fixture future
producono feature senza aggiornare nulla, quindi lo stesso codice serve
backtest e produzione. Il test `test_ratings_sono_causali` verifica che le
feature calcolate su tutto lo storico coincidano esattamente con quelle
calcolate sul solo prefisso.

**xG**: se i CSV contengono `home_xg`/`away_xg` i rating GAP li usano
(peso 0.60 sull'xG, 0.40 sui gol). Se mancano — come oggi, visto che
API-Football non li fornisce nel piano usato — si ricade sui gol senza
modifiche al codice.

---

## 4. Combinazione: pool logaritmico, non stacking libero

La versione ovvia dello stacking (una logistica sulle log-probabilità dei tre
modelli) **è stata provata e scartata**: sui dati sintetici produceva un
ensemble peggiore di tutti i suoi componenti (RPS 0.2255 contro 0.2098 del
miglior modello base, log loss 1.40 contro 1.03). Con tre modelli e tre esiti
sono 27 coefficienti liberi stimati su predizioni out-of-fold di qualità
disomogenea: i primi fold sono addestrati su poco storico e il meta-modello
impara a correggere un errore che nel modello finale non c'è più.

La versione implementata (`ensemble/pooling.py`) impone la struttura giusta:

```
p(esito k) ∝ exp( Σ_i w_i · log p_i(k) + b_k )
```

cioè una media geometrica pesata con **un solo peso per modello**, vincolato a
essere non negativo (un modello può essere ignorato, non invertito), più
un'intercetta per esito. Cinque parametri invece di ventisette, regolarizzati
verso la media uniforme.

Con questa struttura, sui dati sintetici l'ensemble batte il miglior modello
base sul 1X2 su tutti i semi provati, e resta entro lo 0.3% su Over 2.5 e BTTS
(dove il generatore sintetico è per costruzione favorevole ai modelli
parametrici).

**Anti-leak**, in tre punti:
1. le probabilità su cui si stimano i pesi sono out-of-fold, per blocchi
   temporali contigui;
2. i primi blocchi vengono scartati (training troppo piccolo per essere
   rappresentativo del modello finale);
3. il calibratore è stimato sull'ultima parte delle predizioni out-of-fold,
   che il pool non ha usato.

---

## 5. Calibrazione (`ensemble/calibration.py`)

Isotonica one-vs-rest con rinormalizzazione, più due protezioni che non sono
dettagli:

- **floor** — nessuna probabilità viene mai portata a zero. L'isotonica
  assegna 0 a un intero blocco se in quel blocco l'evento non si è mai
  verificato; sul betting significa quota infinita, e una sola occorrenza
  manda la log loss a infinito.
- **shrinkage** — l'output è una media fra probabilità calibrata e originale
  con peso `n/(n+400)`: con pochi dati si corregge poco.

E soprattutto: **la calibrazione viene accettata solo se serve.** Si stima la
mappa sulla prima metà del blocco di calibrazione, si verifica sulla seconda,
e la si tiene solo se la log loss migliora di almeno il 3% relativo.

Il motivo è misurato, non teorico: in test una calibrazione BTTS che migliorava
dell'1.3% sul blocco di verifica peggiorava l'RPS sul periodo successivo da
0.2186 a 0.2348 (ECE da 0.065 a 0.138). Una mappa di calibrazione ha una
scadenza: il livello di gol e la quota di pareggi di una lega si spostano nel
tempo, e una correzione stimata sul passato può essere sbagliata sul futuro.

---

## 6. Dal modello alla giocata (`ensemble/value.py`)

Tre passaggi separati.

**De-vig.** Tre metodi: proporzionale, power, Shin (default). Shin modella il
margine come protezione del book contro gli scommettitori informati e corregge
il *favourite-longshot bias* meglio della normalizzazione proporzionale, che
sistematicamente sovrastima gli outsider. Per i mercati a due esiti dove si ha
una sola quota (Over 2.5, BTTS nel formato `odds.csv` attuale) il margine viene
stimato da quello del 1X2 della stessa partita, scalato di 2/3.

**Filtro value.** Si gioca solo con `edge = p·quota − 1` sopra soglia
(default 3%), quota fra 1.35 e 8.00, probabilità sopra il 5%. La soglia non è
estetica: sotto il 3% l'edge è nell'ordine dell'errore di stima del modello.

**Staking.** Kelly frazionario (default 1/4) con cap per singola giocata
(default 2% del bankroll) e cap giornaliero sul numero di giocate. Kelly pieno
è ottimale solo se le probabilità sono esatte; non lo sono mai, e il
drawdown di Kelly pieno con probabilità stimate è insostenibile.

**CLV.** Se esiste `data/{nazione}/processed/odds_closing.csv` (stesso formato
di `odds.csv`, quote rilevate a ridosso del calcio d'inizio) il backtest calcola
il Closing Line Value in quota e in probabilità. È il KPI di processo: su poche
centinaia di giocate il ROI è quasi solo rumore, mentre un CLV positivo e
stabile dice che il modello vede qualcosa prima del mercato. Senza quel file il
CLV non è calcolabile e viene dichiarato mancante invece di essere stimato.

---

## 7. Evidenze per mercato

**1X2.** È il mercato più efficiente e con il margine più basso. La letteratura
colloca il miglior RPS pubblicato intorno a 0.1925 con accuratezza ~55.8%, e la
quota de-viggata resta un avversario durissimo: nei test sintetici di questo
repo il mercato è secondo solo alle probabilità vere del generatore. L'edge, se
c'è, sta nelle leghe minori e nei momenti in cui il mercato non ha ancora
incorporato un'informazione (formazioni, infortuni, riposo).

**Over 2.5.** È il mercato dove la letteratura riporta il profitto di lungo
periodo più solido con i rating GAP/xG (≈ +0.8% per giocata su 12 anni). Ha
senso: il totale gol dipende meno dalle notizie dell'ultimo minuto e più da
caratteristiche strutturali (stile di gioco, difese, lega), che i rating
catturano bene.

**BTTS.** Il mercato con il margine più alto dei tre e il più sensibile alla
correlazione fra i punteggi — proprio dove il Poisson indipendente sbaglia. Un
modello diretto (M3) ha qui il vantaggio maggiore rispetto a un modello di
punteggio, ed è quello che si osserva anche nei test sintetici (M2 è
sistematicamente il peggiore sul BTTS).

**Efficienza e margini.** L'ordine di efficienza è 1X2 > Over 2.5 > BTTS, e
l'ordine dei margini è l'inverso. Nelle leghe minori i margini sono più alti ma
i prezzi sono peggio informati: si compensano solo in parte, e conviene
verificarlo lega per lega con il breakdown del backtest anziché assumerlo.

---

## 8. Uso

```bash
pip install -r requirements-ensemble.txt

# autotest completo su campionati sintetici: non serve alcun dato reale
python 9_ensemble.py demo

# backtest walk-forward sui dati veri
python 9_ensemble.py backtest --start=2025-08-01 --end=2026-03-01

# solo alcune nazioni, più rapido
python 9_ensemble.py backtest --start=2025-08-01 --end=2026-03-01 \
    --nations=italy,england --fast

# previsioni e giocate sulle fixture future
python 9_ensemble.py predict --from=2026-03-01 --to=2026-03-08

# fotografia dei rating correnti
python 9_ensemble.py ratings --nations=italy

# test
python tests/test_ensemble.py
```

**Tempi.** Ogni finestra di backtest riaddestra i tre modelli base più i fold
out-of-fold: il costo cresce con la dimensione dell'archivio e col numero di
finestre. Su ~3.300 partite e un anno di test con `--fast` sono pochi minuti;
su decine di migliaia di partite e riaddestramento ogni 28 giorni si va sulle
ore. Per un primo giro conviene `--fast --refit=56 --nations=...`.

Opzioni: `--refit=28` (giorni fra riaddestramenti), `--min-edge=0.03`,
`--kelly=0.25`, `--bankroll=1000`, `--devig=shin|power|multiplicative`,
`--market-features` (include le quote fra le componenti dell'ensemble),
`--no-calibration`, `--fast`, `--root=/path`.

Output in `output/`:
`ensemble_backtest_preds_*.csv` (una riga per partita e per modello),
`ensemble_backtest_bets_*.csv` (giocate con edge, stake, esito, CLV),
`ensemble_backtest_bankroll_*.csv` (curva del bankroll),
`ensemble_predictions_*.csv` e `ensemble_bets_*.csv` per la produzione.

Il pacchetto **non tocca `8_predict.py`**: i due sistemi girano in parallelo
sugli stessi CSV e possono essere confrontati sullo stesso periodo.

---

## 9. Dati, validazione, rischi

**Dati.** Servono solo i CSV che la pipeline già produce
(`all_results.csv`, `all_fixtures.csv`, `odds.csv`). Due aggiunte pagherebbero
molto: `odds_closing.csv` per il CLV, e le colonne `home_xg`/`away_xg` in
`all_results.csv`, che i rating GAP userebbero automaticamente.

**Validazione.** Il backtest riaddestra ogni `refit_days` giorni e predice solo
partite successive al training. È l'unica forma di validazione che significa
qualcosa in questo dominio: una cross-validation casuale su partite di calcio
sovrastima le prestazioni in modo grossolano, perché mette nel training partite
successive a quelle di test della stessa squadra e stagione.

**Cosa è stato verificato qui.** Che tutta la catena gira; che i rating sono
causali; che ogni modello base batte il prior; che M2 gestisce squadre mai
viste; che il pool pesa i modelli e la calibrazione riduce l'ECE su
sbilanciamenti reali; che il filtro value e Kelly rispettano i vincoli. Il tutto
su campionati sintetici (`ensemble/synthetic.py`), dove le probabilità vere sono
note.

**Cosa non è stato verificato.** Le prestazioni sui dati reali del progetto: i
CSV non sono nel repository, quindi RPS, hit rate, ROI e CLV su leghe vere sono
tutti da misurare con `9_ensemble.py backtest`. I dati sintetici sono generati
da un processo di Poisson con effetti squadra: **favoriscono per costruzione i
modelli parametrici (M2, M3) e penalizzano M1**, che sui dati veri è il
componente più forte in letteratura. Non si tragga dalla demo alcuna conclusione
sui pesi relativi dei tre modelli.

**Rischi noti, in ordine di importanza.**

1. *Il mercato è forte.* Il riferimento non è il 50% di accuratezza, è la
   quota de-viggata. Se l'ensemble non batte quella colonna nel backtest, non
   c'è edge, e il filtro value sta solo selezionando errori di stima.
2. *Sovradattamento delle soglie.* Ogni soglia scelta guardando il backtest
   (edge minimo, quota massima, leghe da giocare) consuma parte del margine.
   La pipeline attuale ne ha già molte, calibrate su 22 settimane; vale la
   pena tenerne poche e robuste.
3. *Costo del margine.* Con margine del 5-7% servono ~3-4 punti di edge reale
   solo per andare in pari. Un modello "un po' meglio del mercato" perde soldi.
4. *Limiti di puntata e movimento della linea.* Le quote raccolte non sono
   quelle che si ottengono davvero al momento della giocata; il CLV misura
   esattamente questo scarto.
5. *Deriva.* Rating, pesi del pool e calibrazione vanno riaddestrati
   periodicamente. Il backtest walk-forward simula questo; una previsione
   prodotta con un modello di sei mesi fa no.
