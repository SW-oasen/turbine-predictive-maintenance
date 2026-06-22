# Turbofan Predictive Maintenance – Vorhersage der Remaining Useful Life (RUL)

Ein End-to-End Machine-Learning-Projekt zur Vorhersage der **Remaining Useful Life (RUL)** von Turbofan-Triebwerken auf Basis des **NASA CMAPSS**-Datensatzes.

Das Projekt demonstriert den kompletten Workflow von der Datenaufbereitung über Feature Engineering bis hin zur Modelloptimierung und interaktiven Visualisierung mit Streamlit.

---

## Projektübersicht

Predictive Maintenance ermöglicht es, den verbleibenden Lebenszyklus einer Maschine abzuschätzen und Wartungen rechtzeitig durchzuführen.

In diesem Projekt werden Sensordaten simulierte Flugzeugtriebwerke verwendet, um die Anzahl der verbleibenden Betriebszyklen (**Remaining Useful Life, RUL**) vorherzusagen.

Besonderer Wert wurde auf eine **realistische Evaluierung ohne Data Leakage** gelegt. Im Gegensatz zu zufälligen Train-Test-Splits erfolgt die Bewertung ausschließlich anhand der offiziellen Testdaten.

---

## Datensatz

Verwendet wird der **NASA CMAPSS Turbofan Engine Degradation Simulation Dataset**.

Enthalten sind vier unterschiedliche Szenarien:

| Datensatz | Betriebsbedingungen | Fehlermodi |
|-----------|--------------------|------------|
| FD001 | Eine | HPC |
| FD002 | Mehrere | HPC |
| FD003 | Eine | HPC + Fan |
| FD004 | Mehrere | HPC + Fan |

Jeder Zyklus enthält:

- 3 Operational Settings
- 21 Sensorsignale
- Zyklusnummer (`time_cycles`)
- Remaining Useful Life (`RUL`)

---

## Projektstruktur

```
.
├── data/          # Rohdaten (nicht versioniert)
├── db/            # SQLite-Datenbank
├── models/        # Trainierte Modelle und Auswertungen
├── notebooks/     # Experimente
├── src/           # ETL, Datenbank, Modelltraining und Dashboard
└── README.md
```

---

## Datenpipeline

- Zusammenführen aller vier CMAPSS-Datensätze
- Berechnung der RUL für Trainingsdaten
- Speicherung in einer SQLite-Datenbank
- Separate Views für Training und Test
- Saubere Trennung von Trainings- und Evaluierungsdaten

---

## Feature Engineering

Untersucht wurden unter anderem:

- Sensorwerte
- Differenz-Features (`sensor_diff`)
- Rolling Features
- Betriebsbedingungen (`condition`)
- Fehlermodi (`fault_mode`)
- RUL-Capping

Die größten Verbesserungen wurden erzielt durch:

- **Sensor-Differenzmerkmale**
- **RUL-Capping auf maximal 150 Zyklen**

Rolling Features sowie `condition` und `fault_mode` lieferten dagegen nur geringe Verbesserungen.

---

## Modelle

Verglichen wurden verschiedene Regressionsverfahren.

Die besten Ergebnisse erzielten:

- LightGBM
- XGBoost

Anschließend wurden beide Modelle mit **Bayesian Optimization** weiter optimiert.

---

## Ergebnisse

| Modell | MAE | RMSE | R² |
|---------|----:|------:|---:|
| Tuned LightGBM | **17.82** | **24.33** | **0.77** |
| Tuned XGBoost | **17.98** | **24.40** | **0.77** |

Gegenüber der ursprünglichen Baseline konnte die Modellleistung deutlich verbessert werden.

---

## Interaktives Streamlit-Dashboard

Das Projekt enthält ein Dashboard zur Visualisierung der Ergebnisse.

Funktionen:

- Modellvergleich
- Analyse verschiedener RUL-Caps
- Feature-Importance
- Engine Explorer
- Visualisierung normalisierter Sensorverläufe

Starten mit:

```bash
uv add streamlit
uv run streamlit run src/streamlit_dashboard.py
```

---

## Verwendete Technologien

- Python
- pandas
- NumPy
- scikit-learn
- LightGBM
- XGBoost
- SQLAlchemy
- SQLite
- scikit-optimize
- Streamlit
- Matplotlib

---

## Installation

```bash
uv sync
```

oder

```bash
pip install -r requirements.txt
```

Anschließend den NASA-CMAPSS-Datensatz in `data/` ablegen und die ETL-Pipeline ausführen.

---

## Lizenz

Dieses Repository dient Lern-, Demonstrations- und Portfoliozwecken.

---

**Autor:** Yuchuan Liu