# Performance- und Speicherplan (CUPS-Filter)

Stand: 2026-09-23 — gemessen auf dem Printserver; Maßnahmen 1 und 2 umgesetzt (noch nicht committet/installiert).

## Ausgangslage

Zielsystem: Raspberry Pi 3 Model B+ (4 Kerne, 905 MB RAM), Debian 13,
gs 10.05.1, cups-filters 1.28.17, Python 3.13.

CUPS-Filterkette für einen PDF-Job (per `cupsfilter --list-filters` geprüft):

```
PDF → pdftopdf → pdftops (Poppler) → brhl4150cdn-filter
        └→ Tempfile → gs -sDEVICE=ppmraw -r600 → Pipe → read_ppm (ganze Seite)
           → crop_page (Kopie) → _render_page (zeilenweise) → XL2HB auf stdout
```

`pdftops` nutzt für Brother-Drucker absichtlich Poppler statt gs
(„work around bugs in the printer's PS interpreters“). Es läuft also **kein**
doppelter gs-Lauf.

### Messung auf dem Pi (5 Seiten eines LuaTeX-PDFs, A4, 600 dpi)

| Schritt | Zeit |
|---|---|
| pdftopdf + pdftops (5 Seiten) | 1,5 s |
| gs PS → ppmraw, allein | 2,1 s (≈ 0,43 s/Seite) |
| gs PDF → ppmraw direkt | 2,5 s (langsamer als aus PS) |
| gs mit `-dNumRenderingThreads=4` | 2,3 s (kaum Gewinn) |
| pdftoppm (Poppler) 600 dpi | 3,5 s |
| **Unser Filter komplett** | **10,5–11,5 s wall**, 7,4 s user, **4,5 s sys** |
| Python-Start (Imports) | 1,1 s pro Job |
| Inverse-LUT laden (64 MB, `np.load`) | 1,1 s warm, **2,8 s kalt** pro Job |
| Spitzen-RSS | **566 MB** (von 905 MB) |
| CPU-Auslastung laut `vmstat` | ~25 % → genau ein Kern arbeitet |

cProfile des Filters (9,7 s für 5 Seiten):

| Posten | Zeit | Anteil |
|---|---|---|
| `read()` aus der gs-Pipe (Warten auf gs + Kopieren) | 2,1 s | 21 % |
| `crop_page` (`np.full` 0,84 s + `tobytes` 1,05 s + Slicing 0,49 s) | 2,4 s | 25 % |
| `_render_page` gesamt | 3,8 s | 39 % |
| — davon eigene Zeit (Zeilen-Slicing, Weiß-Vergleich, Schleife) | 1,3 s | |
| — Dither | 1,0 s | |
| — Farbe (Inverse-LUT) | 0,9 s | |
| — Encode + Buffer | 0,5 s | |
| Start, Imports, LUT | ~1,3 s | 13 % |

Die Inverse-LUT ist installiert; der ~20× langsamere Fallback greift nicht.

### Folgerungen

- **Speicherkopien sind der größte einzelne Posten**, nicht gs: `crop_page`
  plus die Page-Fault-Last der 100-MB-Puffer (4,5 s Kernel-Zeit) kosten mehr
  als das Rastern.
- Alles läuft seriell auf einem Kern; drei Kerne sind im Leerlauf.
- Der Filter schafft ~2 s/Seite. Der Drucker druckt 24 Seiten/min (2,5 s/Seite).
  Bei langen Jobs ist also der Drucker der Engpass; spürbar ist vor allem die
  **Zeit bis zur ersten Seite** (≈ 1,5 s pdftops + 1,1 s Start + 1–3 s LUT +
  ~2 s erste Seite) und das **Speicherrisiko**.
- 566 MB RSS auf 905 MB RAM: Long-Edge-Duplex (`_flip_vertical`, weitere
  Kopien) liegt noch darüber. `settings.reverse` hält **alle** Seiten im
  Speicher (~100 MB/Seite) — bei mehr als ~3 Seiten droht OOM.

### Warum `asyncio` nicht das richtige Werkzeug ist

`asyncio` würde nur das Lesen der Pipe nicht-blockierend machen. Das Rendern ist
CPU-Arbeit und blockiert den Event-Loop — man müsste es ohnehin per
`run_in_executor` in einen Thread auslagern. Direkter ist ein Reader-Thread;
blockierende `read()`-Aufrufe geben den GIL frei.

Warum Python heute auf gs wartet: `read_ppm` blockiert, bis gs die komplette
Seite geschrieben hat; gs rastert im Banding-Modus erst beim Ausgeben. Danach
blockiert gs nach 64 KB Pipe-Puffer, bis Python die Seite fertig hat. Effektiv:
`Σ (gs + python)` statt `Σ max(gs, python)`.

## Ergebnis Maßnahmen 1 + 2 (Pi, gleiche 5 Seiten, alt = installierter HEAD)

| | alt | neu |
|---|---|---|
| wall, warm (3 Läufe) | 9,3–12,0 s | **6,5–6,8 s** |
| wall, kalter Page-Cache | 14,0 s | **8,7 s** |
| user / sys | 7,4 s / 4,5 s | 5,9 s / 3,0 s |
| Spitzen-RSS | 566 MB | **330 MB** |
| Duplex Long-Edge, wall | 10,2 s | 6,7 s |

Ausgabe in allen Läufen byte-identisch (Normal, Duplex Long-Edge, kalter
Cache); lokal zusätzlich Duplex Short-Edge + `BRReverse`, `BRSkipBlank` +
`BRBrightness` gegen den alten Stand verglichen. 1041 Tests grün.

Umsetzung:

- `crop_page` liefert einen View auf den gs-Puffer, wenn der Render den
  Druckbereich abdeckt (Normalfall); nur kleinere Renders werden weiß
  aufgefüllt kopiert.
- `_render_page` arbeitet auf Zeilen-Views (`_page_rows`); Weißzeilen per
  einer vektorisierten Maske pro Seite; `_flip_vertical` liefert einen
  umgekehrten View statt `np.full` + `tobytes`.
- `is_blank_page` ersetzt die `bytes.count`-Prüfung von `skip_blank`.
- Inverse-LUT per `np.load(mmap_mode="r")`; Fallback auf die Interpolation
  loggt einmal pro Prozess eine Warnung.

Verbleibende 330 MB: der 105-MB-`bytes`-Puffer aus `read_ppm` plus
transiente Kopien beim Lesen aus der Pipe, Numpy/LUT-Mapping. Das adressiert
Maßnahme 3 (Streaming).

## Maßnahmen (nach Nutzen/Aufwand sortiert)

### 1. `crop_page`-Kopien entfernen (≈ −25 %, klein, sicher) — ✅ umgesetzt

- [x] Kein `np.full` + `tobytes` mehr: `_render_page` bekommt den
      ungecroppten Puffer plus Offset (x=100, y=100) und Quellbreite und
      schneidet jede Zeile per `memoryview` direkt aus.
- [x] Fehlende Zeilen/Spalten (Quelle kleiner als Druckbereich) als Weiß
      behandeln — das leistet heute `np.full(..., 255)`.
- [x] Im Render-Loop `pixel_data[a:b]` (kopiert jede Zeile als `bytes`) durch
      `memoryview`-Slices ersetzen; der Weiß-Vergleich braucht dann einen
      Weg ohne Kopie (z. B. vorab per numpy eine Maske „Zeile komplett weiß“
      für die ganze Seite).
- Byte-Identität unverändert (gleiche Pixel, nur ohne Kopie).

### 2. Inverse-LUT per mmap (≈ −1 bis −2,8 s pro Job, klein, sicher) — ✅ umgesetzt

- [x] `np.load(path, mmap_mode="r")`: nur tatsächlich benutzte Farben werden von
      der SD-Karte gelesen, Page-Cache wird zwischen Jobs geteilt, RSS sinkt um
      bis zu 64 MB.
- [x] Prüfen, ob `gather_kcmy` einen read-only-Puffer akzeptiert
      (`const unsigned char[:]` im `.pyx`).
- [x] Fallback auf `_rgb_to_cmyk_interp_arr` mit `WARNING` loggen — ein
      fehlgeschlagener Install-Schritt macht den Filter sonst still ~20×
      langsamer.

### 3. Zeilen-Streaming + Reader-Thread (Speicher und Überlappung mit gs)

- [ ] `read_ppm` in Header-Parser + blockweisen Zeilenleser aufteilen
      (z. B. 256 Zeilen ≈ 3,8 MB pro Block).
- [ ] Reader-Thread liest gs-Ausgabe in eine `queue.Queue(maxsize=N)` von
      Zeilenblöcken; Hauptthread rendert. gs rastert weiter, während Python
      rendert. `N` regelt Speicher gegen Entkopplung.
- [ ] Cropping entfällt dabei komplett (Offsets beim Lesen anwenden).
- [ ] Exceptions/EOF per Sentinel an den Hauptthread durchreichen.
- Effekt: Spitzen-RSS von 566 MB auf < 100 MB; sys-Zeit sinkt (keine
  100-MB-Allokationen mehr); gs-Zeit (~0,43 s/Seite) überlappt mit Rendern.
- Ausnahmen, die weiter die ganze Seite brauchen:
  - Long-Edge-Duplex-Rückseiten (`_flip_vertical`): Seite puffern, aber ohne
    `np.full` + `[::-1].tobytes()` — Zeilen rückwärts aus dem Puffer lesen.
  - ~~`settings.reverse`~~ — ✅ umgesetzt: Seiten werden vorwärts gerendert,
    fertige XL2HB-Seiten in eine temporäre Datei geschrieben und rückwärts
    ausgegeben. Long-Edge-Duplex braucht die Seitenzahl vorab
    (`count_ps_pages`: DSC `%%Pages:` sofort, sonst gs-`nullpage`-Zählung,
    ~16 s für 131 Seiten). Pi, 131 Seiten, Reverse + Long-Edge: 280 s,
    347 MB RSS (vorher OOM); byte-identisch zum alten In-RAM-Umdrehen.
  - `skip_blank` (heute unkritisch, prüft je eine Seite): beim Streaming die
    Seite erst nach dem Rendern verwerfen (gerenderte XL2HB-Seite puffern,
    Leerseite erkennen) statt vorab das Raster zu prüfen.

### 4. Parallelisierung über Kerne (3 Kerne liegen brach)

Variante A — Seiten-Pipeline über Prozesse: gs, Farbe/Dither, Encode laufen
bereits teilweise getrennt; mit Maßnahme 3 überlappen gs und Python. Das
nutzt 2 Kerne.

Variante B — Bänder innerhalb einer Seite:

- [ ] Band-Kernel in Cython `render_band(rgb, first_line, n_lines, ...)`:
      Farbe → Dither → RLE für z. B. 64 Zeilen in einem `nogil`-Aufruf.
      Beseitigt nebenbei den Python-Overhead pro Zeile (1,3 s eigene Zeit von
      `_render_page` für 5 Seiten).
- [ ] Bänder per `ThreadPoolExecutor` rendern, Ergebnisse **in Reihenfolge**
      an den `PlaneBuffer`-Flush (bleibt sequentiell).
- Voraussetzungen: `DitherChannel._tiled_cache` vorwärmen; kein
  zeilenübergreifender Zustand in Saturation/Vivid/Input-Remap/Tone-Curve
  (prüfen); diese Schritte müssen in den Kernel oder bandweise davor.
- Erwartung: Render-Anteil (≈ 0,76 s/Seite) auf ein Drittel bis Viertel.
- Byte-Identität per `TestSettingVariants` absichern.

### 5. Startzeit (jeder Job)

- [ ] `python -X importtime` auswerten (heute 1,1 s); `fine_encoder`,
      `color_lut_gen` usw. nur bei Bedarf importieren.
- [ ] `.pyc` bei der Installation vorkompilieren (`compileall`); prüfen, ob
      `__pycache__` in `/usr/local/lib/brhl4150cdn/` für den CUPS-User aktuell
      ist.
- [ ] Tempfile vermeiden: gs liest PostScript direkt von stdin (`-`).

### 6. Verworfen bzw. niedrige Priorität (nach Messung)

- `-dNumRenderingThreads`: 2,46 → 2,30 s, lohnt kaum.
- gs direkt aus PDF statt aus PS: langsamer (2,5 vs. 2,1 s).
- `pdftoppm`/Poppler als Rasterizer: langsamer (3,5 s).
- CUPS-Raster als Eingabe (`application/vnd.cups-raster` über
  `gstoraster`/`pdftoraster`): Es gibt keinen doppelten gs-Lauf, der Gewinn wäre
  nur der Wegfall von pdftops (läuft ohnehin parallel in der CUPS-Kette).
  Bleibt interessant, weil CUPS dann Streaming, `reverse`, Seitenbereiche und
  druckbaren Bereich übernimmt — aber größerer Umbau, später.

## Reihenfolge

1. ~~`crop_page` entfernen (1) und LUT per mmap (2)~~ — erledigt:
   −30 bis −45 % Laufzeit, −236 MB RSS
2. ~~`reverse` ohne Raster-Puffer~~ — erledigt, 131 Seiten mit 347 MB
3. Streaming + Reader-Thread (3)
4. Startzeit (5)
5. Band-Kernel + Threads (4B), nur falls die Zeit bis zur ersten Seite danach
   noch stört

Jede Stufe gegen `uv run pytest tests/ -q` und die Byte-Identitätstests
(`TestSettingVariants`) absichern und auf dem Pi nachmessen (wall, user, sys,
maxrss wie oben).
