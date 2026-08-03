# Simulatore SAR multifloor - v38 completa

Questa distribuzione completa deriva dalla v37. Mantiene
la vista 3D Panda3D persistente, Max Turbo, l'occupancy grid fine, il planner,
le statistiche con effect size e tutti gli elementi grafici già presenti. La
v38 aggiunge l'analisi della relazione tra tempo effettivo di esplorazione e
area totale esplorata, con classificazione lineare/non lineare e identificazione
della forma della relazione.


## Correzione v36: blocco dopo l'episodio 144 con densita 3x

Con tre valori di `Fw` e due valori di `Opt`, ogni edificio produce sei
episodi. L'episodio 144 termina quindi la Run 24; la Run 25 usa il seed 24.
Con densita oggetti `3x`, quel seed portava il packing degli arredi vicino alla
saturazione. Il vecchio fallback riprovava migliaia di volte gli stessi
footprint non validi e ricalcolava ogni volta connettivita, distance transform
e raggiungibilita di tutte le stanze. Il lavoro avveniva prima del watchdog
della diagnostica start, percio sembrava un blocco completo.

La v36 memorizza i footprint gia rifiutati, esaurisce una sola volta i candidati
`2x2`, usa controlli di connettivita piu efficienti e conserva in cache il
template deterministico dell'edificio. Ogni episodio riceve comunque una copia
profonda indipendente: occupancy grid, mappe semantiche e stato del planner non
sono condivisi.

Test specifico:

```powershell
python verify_density_three_episode_144_transition.py
```

## Installazione

```powershell
python -m pip install -r requirements.txt
```

Panda3D richiede una scheda grafica e driver con supporto OpenGL. Su Windows la
ruota Python installata da pip include il motore e non richiede un compilatore.

## Avvio

```powershell
python main.py
```

Nella finestra principale premere **Abilita vista 3D**. La seconda finestra può
essere chiusa con `Esc`, con la `X`, oppure premendo nuovamente il pulsante.

## Architettura

Il processo principale continua a gestire la simulazione. Il processo Panda3D
riceve una sola volta muri, scale, arredi, macerie e persone. Durante la run
riceve soltanto piano, posizione e orientamento del robot. La scena resta quindi
coerente con la ground truth senza essere rigenerata a ogni fotogramma.

## Test headless

```powershell
python verify_camera3d.py
python verify_rich_office_3d.py
```

I test headless verificano serializzazione, footprint e contenuto della scena.
La finestra Panda3D deve essere provata su una macchina con display e GPU.

## Correzioni v24

La telecamera Panda3D usa ora una trasformazione esplicita dal sistema di coordinate della mappa 2D al sistema destrorso di Panda3D. La posizione e il punto osservato sono calcolati direttamente dal vettore frontale del robot, evitando inversioni destra/sinistra o viste rivolte all'indietro.

La camera e collocata 0.28 m davanti al centro del robot e orientata mediante `lookAt()`.

Le decorazioni murali sono state ripristinate e rese piu affidabili. Il renderer individua il lato della parete rivolto verso uno spazio interno e vi colloca deterministicamente:

- quadri con cornice;
- bacheche con fogli;
- librerie e scaffali;
- libri separati;
- battiscopa.

Questi elementi sono puramente visivi e non alterano collisioni, LiDAR, occupancy grid o planner.

Test aggiuntivo:

```powershell
python verify_camera_orientation_and_decor.py
```

## Novità v25 — ufficio SAR e movimento della videocamera

La vista Panda3D include ora variazioni visuali coerenti con uno scenario di
search and rescue, senza modificare la ground truth usata dalla simulazione:

- una quota deterministica di sedie e tavoli è ribaltata;
- le vittime sono distribuite fra pose in piedi, sedute e sdraiate;
- piante da ufficio, cestini e piccoli accessori sono collocati vicino ai muri;
- soltanto i muri perimetrali possono ricevere finestre con cielo azzurro;
- la videocamera frontale compie una lieve oscillazione verticale durante la
  traslazione, inferiore a 2 cm, per suggerire il passo del quadrupede.

Piante, finestre e accessori sono decorazioni e non vengono inseriti nella
mappa delle collisioni. Tavoli e sedie ribaltati mantengono invece lo stesso
footprint fisico già usato dal simulatore: cambia solo la loro posa 3D.

## Miglioramenti visivi v26

Le finestre della vista 3D sono ora integrate nelle pareti perimetrali e includono montante centrale e maniglia. Una finestra viene creata soltanto quando la parete e la fascia interna davanti ad essa sono libere da arredi, vittime, porte e scale. Anche le decorazioni non fisiche rispettano questa fascia libera.

Il soffitto include lampade da ufficio; alcune sono spente in modo deterministico per simulare un guasto conseguente all'evento catastrofico. Vicino alle librerie sono inoltre presenti libri caduti sul pavimento. Lampade, libri e dettagli delle finestre sono esclusivamente visuali e non modificano la simulazione.

## Novità v27

- Ogni tavolo presenta sempre almeno uno tra monitor spento, portapenne con penne e faldoni/documenti; sono possibili combinazioni multiple.
- Le scale in discesa sono rappresentate come aperture scavate nel solaio, con vano scuro, pareti interne e gradini che scendono sotto quota pavimento.
- Le scale in salita mantengono la resa precedente.
- Sono stati aggiunti attaccapanni a parete, talvolta con vestiti; una quota deterministica è caduta sul pavimento.
- Tutti questi dettagli sono esclusivamente visivi e non modificano LiDAR, collisioni, occupancy grid o planner.

## Aggiornamento v28

I monitor sono sempre rivolti verso un lato lungo del tavolo, anche quando il footprint e ruotato. Gli attaccapanni sono stati rimossi. Le scale in discesa sono colorate in azzurro e composte da veri gradini che scendono progressivamente sotto la quota del pavimento.

## Aggiornamento v29

Le scale SU e GIU sono entrambe simboli 3D rialzati sul pavimento. Le prime crescono nella direzione della freccia; le seconde sono azzurre e presentano la successione speculare, quindi la freccia punta verso il gradino piu basso. Gli accessori sui tavoli sono distribuiti in zone distinte: monitor sul bordo posteriore, faldoni sul lato anteriore sinistro e portapenne sul lato anteriore destro.

## Miglioramento v30: librerie

Le librerie mostrano ora file dense di libri realmente appoggiati sui ripiani.
Dalla stanza sono visibili i dorsi affiancati, con larghezze, altezze e colori
leggermente differenti. La disposizione è deterministica e puramente visiva.

## Aggiornamento v31: librerie aperte e fogli sparsi

Gli armadi alti già presenti sono stati conservati. È stata aggiunta una
seconda tipologia di arredo chiaramente riconoscibile come libreria aperta, con
ripiani orizzontali visibili e file dense di libri colorati. Dalla stanza si
vedono soltanto i dorsi; i libri sono appoggiati sul ripiano sottostante e sono
affiancati con piccole variazioni di larghezza e altezza.

Quando la geometria lo consente, ogni piano riserva pareti distinte a un
armadio esistente e a una nuova libreria. I libri caduti sul pavimento sono
collocati in prossimità delle librerie aperte.

Inoltre, numerosi fogli formato A4 sono sparsi in stanze e corridoi. Alcuni
sono isolati, altri sono parzialmente sovrapposti. La distribuzione è
procedurale ma deterministica e copre l'intero piano. Fogli e libri caduti sono
solo decorazioni: non modificano collisioni, LiDAR, occupancy grid o planner.

Test specifico:

```powershell
python verify_open_bookshelves_and_papers.py
```


## Aggiornamento v32: Max Turbo

Accanto al pulsante **Abilita vista 3D** è disponibile il nuovo pulsante
**Max Turbo**. Quando viene attivato:

- l'ultimo fotogramma completo della finestra principale resta congelato;
- viene aggiornata soltanto la fascia superiore con il numero di Run e di
  episodio corrente;
- il rendering delle mappe, della legenda, dei raggi e della traiettoria viene
  completamente sospeso;
- la finestra Panda3D viene temporaneamente chiusa per non consumare CPU/GPU;
- la simulazione esegue direttamente i passi fisici da 10 ms in cicli stretti,
  senza limite a 60 FPS, moltiplicatore temporale o attese real-time.

La modalità resta attiva anche quando termina un episodio e comincia quello
successivo. Il pulsante **Ferma Max Turbo** ripristina la visualizzazione e il
moltiplicatore precedente (`1x`, `2x`, `4x` oppure `8x`). Se la vista 3D era
aperta prima dell'attivazione, viene riaperta sulla scena e sulla posa correnti.

Il loop Turbo continua a controllare gli eventi a intervalli brevi, così il
pulsante rimane utilizzabile anche mentre il processore esegue la simulazione
alla massima velocità disponibile.

Test specifico:

```powershell
python verify_max_turbo.py
```

## Correzione v33: pausa apparente tra gli edifici in Max Turbo

La v32 poteva sembrare bloccata immediatamente dopo l'ultimo episodio di un
edificio. Prima di iniziare l'edificio successivo, il programma esegue infatti
la diagnostica invisibile della posizione iniziale per tutte le condizioni
`Fw x Opt`. Con tre valori di `Fw` e due valori di `Opt`, vengono eseguite sei
brevi simulazioni di prova. In quel tratto il terminale non stampava alcun
avanzamento e la finestra Pygame, mantenuta aperta da Max Turbo, non elaborava
gli eventi del sistema operativo.

La v33 mantiene esattamente la stessa diagnostica, ma:

- mostra `Preparazione Run ...`, tentativo, condizione e percentuale;
- elabora periodicamente gli eventi Pygame anche tra due edifici;
- consente di disattivare Max Turbo o chiudere la finestra durante la verifica;
- applica un watchdog di 30 secondi reali a ogni singola prova di start;
- libera esplicitamente le grandi mappe e le cache al termine di ogni edificio;
- stampa nel terminale l'inizio di ogni condizione diagnostica.

Il watchdog non modifica i risultati: se una prova anomala supera il limite,
quella posizione iniziale viene scartata e viene selezionato un nuovo
`start_attempt`, come già previsto dal simulatore.

Test specifico:

```powershell
python verify_max_turbo_transition_watchdog.py
```

## Correzione v34: sei piani e densità oggetti elevata

Con molti piani il vecchio generatore cercava le scale dopo aver creato tutte
le porte. La posizione doveva essere contemporaneamente libera da ogni fascia
di accesso alle porte di tutti i piani; con sei piani l'intersezione poteva
diventare vuota e produrre `No adjacent stair pair available in corridors`.

La v34 riserva invece i nuclei scala nel template comune dei corridoi prima di
generare le stanze. Le porte di ciascun piano vengono poi collocate evitando
quelle regioni. Eventuali suddivisioni incompatibili vengono rigenerate in modo
deterministico a partire dallo stesso seed. La procedura funziona anche con
oggetti a densità `4x` e mantiene lo stesso edificio per tutte le condizioni
`Fw x Opt` della run.

Un errore strutturale non viene più confuso con uno start bloccato: il programma
non ripete inutilmente lo stesso edificio cambiando soltanto la posizione del
robot.

Test specifico:

```powershell
python verify_six_floor_dense_stairs.py
```

## Aggiornamento v35 (derivato direttamente dalla v34)

Questa versione riparte dalla v34 e introduce soltanto due modifiche.

### Preferenza della vista 3D persistente

Quando il pulsante **Abilita vista 3D** viene premuto, la scelta resta attiva
per tutti gli episodi successivi e anche quando viene generato un nuovo
edificio. Il processo Panda3D viene comunque chiuso alla fine dell'episodio e
ricreato con la geometria esatta del nuovo `RunState`; in questo modo non viene
riutilizzata per errore la scena dell'edificio precedente.

Premendo **Disabilita vista 3D**, la preferenza viene spenta per tutte le run
successive. Durante Max Turbo la finestra 3D è sospesa, ma la preferenza resta
memorizzata e la vista viene ripristinata sulla scena corrente quando Max Turbo
viene disattivato.

### Effect size nei test statistici

Tutti i test finali riportano ora anche una dimensione dell'effetto e una
valutazione qualitativa:

- ANOVA su condizioni `(Fw, Opt)` e ANOVA sui gruppi `Fw`: eta-quadrato
  (`eta^2`), classificato come trascurabile, basso, medio o alto;
- confronti Tukey-Kramer fra coppie di gruppi: Hedges' `g` con segno;
- Welch t-test `Opt=ON` contro `Opt=OFF`: Hedges' `g` con segno.

Le soglie qualitative usate sono:

- `eta^2`: trascurabile `<0.01`, basso `<0.06`, medio `<0.14`, alto `>=0.14`;
- `|g|`: trascurabile `<0.20`, basso `<0.50`, medio `<0.80`, alto `>=0.80`.

Effect size e interpretazione vengono stampati nel terminale e aggiunti ai
fogli `ANOVA conditions`, `Tukey conditions`, `Welch Opt`, `ANOVA Fw` e
`Tukey Fw` del file `sar_simulation_analysis.xlsx`.

Test specifico:

```powershell
python verify_persistent_3d_and_effect_sizes.py
```

## Aggiornamento v38: relazione tempo effettivo-area esplorata

Al termine del batch viene ora analizzata la relazione fra:

- `texpl`, cioe il tempo simulato effettivamente trascorso nell'esplorazione;
- `Atot`, cioe la percentuale dell'area complessivamente esplorabile che e stata
  esplorata.

`texpl` esclude gia il tempo convenzionale speso nei cambi di piano: quando il
robot usa una scala, il costo `Ts` viene sottratto dal budget residuo, ma non
incrementa `texpl`.

L'analisi riporta:

- correlazione di Pearson e correlazione di Spearman;
- regressione lineare, pendenza, p-value, `R^2` e `R^2` corretto;
- confronto AICc tra modelli lineare, quadratico, logaritmico, a radice
  quadrata ed esponenziale saturante;
- classificazione automatica come `approssimativamente lineare` oppure
  `non lineare`;
- in caso di non linearita, descrizione della forma: saturante/rendimenti
  decrescenti, quadratica concava oppure quadratica convessa;
- forza della relazione e livello di evidenza della non linearita.

Per evitare di interpretare come non linearita miglioramenti trascurabili, un
modello non lineare viene preferito solo se migliora l'AICc del modello lineare
di almeno 2 punti. L'analisi viene calcolata complessivamente e anche separata
per `Opt`, per `Fw` e per ogni condizione `(Fw, Opt)`, quando vi sono abbastanza
osservazioni e tempi distinti.

I risultati sono disponibili:

- nel terminale, come nuova sezione 4 dei test statistici;
- nel grafico finale `Relazione tempo effettivo - area totale esplorata`;
- nei fogli Excel `Time-area relation` e `Time-area models`;
- nel file `sar_simulation_time_area_relationship.csv`.

Test specifico:

```powershell
python verify_time_area_relationship.py
```
