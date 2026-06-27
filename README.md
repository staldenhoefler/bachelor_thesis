# Digitale Bildauswertung zur Hafteigenschaftsanalyse von Klebstoffen

Dieses Repository enthält den Code und begleitende Notebooks zur Bachelorarbeit **„Digitale Bildauswertung zur Hafteigenschaftsanalyse von Klebstoffen“** (FS25).

Ziel ist es, Bilder von Klebstoffproben (u.a. **Vorher/Nachher**-Aufnahmen mit/ohne Photobox) automatisiert auszuwerten, Features zu extrahieren und Modelle zu trainieren, die Rückschlüsse auf Hafteigenschaften erlauben.

---

## Inhaltsverzeichnis

* [Projektüberblick](#projektüberblick)
* [Repository-Struktur](#repository-struktur)
* [Daten & Ordnerkonventionen](#daten--ordnerkonventionen)
* [Konfiguration](#konfiguration)
* [Training](#training)
* [Notebooks](#notebooks)
* [Baselines & Utilities](#baselines--utilities)
* [Coding Style](#coding-style)

---

## Projektüberblick

* **Problemstellung:** Quantitative Analyse der Hafteigenschaften von Klebstoffen anhand digitaler Bilddaten.
* **Ansatz:** Kombination aus klassischer Bildverarbeitung (Detektion/Segmentierung von Klebestreifen) und Deep-Learning-Modellen (CNNs, ResNets, Vision Transformer) auf Vorher/Nachher-Bildpaaren.
* **Ergebnisartefakte:** Trainierte Modelle, Metriken, Plots und Tabellen werden reproduzierbar in `artifacts/` bzw. über Weights & Biases (optional) abgelegt.

---

## Repository-Struktur

```text
.
├─ data/                      
├─ notebooks/
│  └─ archive/
│     ├─ 00_eda_images.ipynb
│     ├─ 00_eda_labels.ipynb
│     ├─ 02_traintestsplit.ipynb
│     ├─ 03_traintestsplit_before_after_with_box.ipynb
│     ├─ 04_before_after_without_photobox.ipynb
│     ├─ 05_before_after_with_photobox.ipynb
│     ├─ 06_additional_rating.ipynb
│     ├─ 07_prediction_classic.ipynb
│     └─ 08_tests_on_testset.ipynb
├─ src/
│  ├─ data/
│  │  ├─ after_image_dataset.py
│  │  └─ before_after_image_with_box_dataset.py
│  ├─ model/
│  │  ├─ base_cnn.py
│  │  ├─ advanced_cnn.py
│  │  ├─ before_after_many_feature_maps_cnn.py
│  │  ├─ CNN.py
│  │  ├─ resnet_18.py
│  │  ├─ resnet_50.py
│  │  ├─ Vision_Transformer.py
│  │  └─ lightning_base_model.py
│  └─ utils/
│     ├─ adhesive_stripe_detector.py
│     ├─ login_data.yaml             # Zugangsdaten (W&B)
│     ├─ params.yaml                 # zentrale Projekt-/Trainings-Parameter
│     └─ train.py                    # Trainings-Entry-Point
├─ coding_guidelines.md       # Style Guide für dieses Repo
├─ pyproject.toml             # Poetry-Projektdefinition
└─ README.md
```

---

## Konfiguration

Alle wichtigen Hyperparameter und Projektpfade werden in `src/utils/params.yaml` gepflegt, z.B.:

* Datenpfade & Splits
* Bildvorverarbeitung/Transforms
* Trainingsparameter (Batchgröße, Epochen, Lernraten, Seeds)
* Modellwahl (z.B. `base_cnn`, `resnet_18`, `vit`)

---

## Training

```bash
python -m src.utils.train
```

---



## Baselines & Utilities

* **Klassische Baseline:** `src/utils/adhesive_stripe_detector.py` – Klassische Verfahren zur Detektion/Segmentierung von Klebestreifen.
* **Datasets:**

  * `after_image_dataset.py` – Datensatz für Einzelbilder (Nachher)
  * `before_after_image_with_box_dataset.py` – Datensatz für Vorher/Nachher-Bildpaare mit optionalen Boxen/Masken
* **Modelle:**

  * CNN-Varianten (`base_cnn.py`, `advanced_cnn.py`, `before_after_many_feature_maps_cnn.py`, `CNN.py`)
  * ResNets (`resnet_18.py`, `resnet_50.py`)
  * Vision Transformer (`Vision_Transformer.py`)
  * Lightning-Basisklasse (`lightning_base_model.py`) zur Vereinheitlichung von Training/Logging

---

## Coding Style

* **Coding-Guidelines:** Bitte `coding_guidelines.md` beachten (Konventionen, Linting, Typisierungen, Struktur).

