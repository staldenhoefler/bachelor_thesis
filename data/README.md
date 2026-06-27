# data – Datensätze & Struktur

Kurzübersicht über die verwendeten Bilddaten, Begriffe, Splits und Ordner.

---

## Begriffe
- **Referenz-Bild (Before):** Zeigt die Kleberaupen im ursprünglichen Zustand. Dient zur Differenzbildung zum Nachher-Bild (klassische Methode) und als Zusatzinformation für DL-Modelle.
- **Nachher-Bild (After / Post-Peel):** Gleiche Stelle nach dem Bead-Peel-Test. Kleberückstände dienen zur Beurteilung der Haftung.

---

## Datensätze

### 1) Post-Peel-Datensatz (nur Nachher-Bilder)
- **Umfang:** 1008 Bilder (iPhone SE, 12 MP, HEIF; manuell beschnitten auf eine Raupe/Bild).  
- **Labels:** Qualitätsgüteklassen 1 (beste) bis 5 (schlechteste) + Metadaten (Produkt, Lagerung, Substrat, Primer, Zusatzfehler).
- **Split & Metrik:** 80/20 Train/Test, **stratifiziert**; Optimierung und Reporting mit **Macro-F1** (gerecht für seltene Klassen).

### 2) Referenz-Datensatz (Before/After-Paare)
- **Umfang:** 75 Bildpaare → **219** segmentierte Klebestreifen.
- **Labels:** Keine Klassenlabels; Fokus auf Paar-Bezug, Segmentierung/Bounding-Boxes.
- **Erhebung:** Erst ohne, dann **mit Fotobox** (konstante Kameraposition). 

---

## Ordnerstruktur

| Pfad | Inhalt |
|---|---|
| `data/train/img` | Post-Peel Trainingsbilder |
| `data/train/labels` | Labels/Metadaten für Train |
| `data/test/img` | Post-Peel Testbilder |
| `data/test/labels` | Labels/Metadaten für Test |
| `data/img`, `data/labels` | (Optional) Gesamtablagen ohne Split |
| `data/before_after_with_box/before_after_img_raw` | Rohbilder der **Paare** (Before/After, Fotobox) |
| `data/before_after_with_box/bbox_annotated_img_raw` | Bilder mit **BBox-Annotationen** der Klebestreifen |
| `data/before_after_with_box/train/train_labels_before_after_with_box.xlsx` | Metadaten/Zuordnungen für **Train** (Paare, BBoxes, IDs …) |
| `data/before_after_with_box/train/train_stripes.pkl` | Vorverarbeitete Streifen/Annotationen (Train) |
| `data/before_after_with_box/test/test_labels_before_after_with_box.xlsx` | Metadaten/Zuordnungen für **Test** |
| `data/before_after_with_box/test/test_stripes.pkl` | Vorverarbeitete Streifen/Annotationen (Test) |

> Hinweis: Die `*.pkl`-Dateien enthalten vorverarbeitete Objekte/Strukturen (z. B. Streifen-Koordinaten, Indizes).