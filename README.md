# South Africa Electricity System Analysis
### Demand Forecasting & Supply Optimisation

**Supervised by Nicolas Maisonneuve-Bonteil, Deloitte | Université Paris 1 Panthéon-Sorbonne | M2 Sustainable Development Economics**

---

## Overview

This project develops a quantitative analytical framework for South Africa's electricity system, combining **electricity demand forecasting** with **supply-side economic dispatch optimisation**. The analysis evaluates trade-offs between cost, reliability, and decarbonisation under alternative policy scenarios.

The case study is motivated by South Africa's ongoing energy crisis: chronic load shedding, heavy coal dependence (~80% of generation), and a structural transition toward renewables under the Just Energy Transition Partnership (JETP).

---

## Project Structure

```
south-africa-electricity-analysis/
├── notebooks/
│   ├── 01_demand_forecasting.ipynb         # Demand forecasting (ARIMA/SARIMAX/LSTM/TFT/Hybrid)
│   └── 02_supply_optimization.ipynb        # Pyomo dispatch model & scenarios
├── data/
│   └── ESK17390.csv                        # Eskom hourly system data (2021–2025)
├── reports/
│   ├── south-africa-electricity-analysis.pdf   # Full academic report (PDF)
│   ├── presentation_slides.pdf                 # Presentation slides 
│   └── report_source.md                     # Report source (Markdown)
├── figures/
├── requirements.txt
└── README.md
```

---

## Analytical Components

### 1. Demand Forecasting (`01_demand_forecasting.ipynb`)

Forecasts daily electricity demand over a 90-day out-of-sample horizon using six modelling approaches:

| Model | Type | Key feature |
|-------|------|-------------|
| ARIMA(1,0,1) | Statistical | Baseline univariate model |
| SARIMA(1,0,1)(1,1,1,7) | Statistical | Weekly seasonality |
| SARIMAX | Statistical | Exogenous regressors (outages, load shedding, pumped storage) |
| LSTM | Deep learning | Nonlinear dynamics, 30-day lookback window |
| Temporal Fusion Transformer (TFT) | Deep learning | Multi-horizon, attention mechanism |
| SARIMA + LSTM Hybrid | Ensemble | Combines parametric and neural components |

**Best model**: SARIMA-LSTM Hybrid — MAE ≈ 474 MW (1.95% of mean demand)

**Risk-aware metrics**: Pinball loss (α = 0.95), P95 tail error

---

### 2. Supply Optimisation (`02_supply_optimization.ipynb`)

Economic dispatch model implemented in **Python/Pyomo** (linear programming). Minimises total system cost subject to technical, contractual, and policy constraints.

**Generation technologies**: Nuclear · Coal · Gas · Solar · Wind

**Objective function**:

```
min Z = Σ(i,t) cᵢ · Pᵢₜ  +  c_shed · Σₜ Uₜ  +  p_CO₂ · Σ(i,t) eᵢ · Pᵢₜ  +  ToP_penalty · Short
```

**Constraints**: Demand balance (with 10% transmission loss) · Capacity limits · Renewable intermittency · Take-or-Pay gas contract · Carbon emissions

---

## Key Results

### Baseline (FY2023-24)

| Metric | Value |
|--------|-------|
| Total system cost | R 350.1 billion |
| Average SMP | R 1,780 /MWh |
| Load shedding | 712 hours (8.1% of year) |
| Avg. emissions | 19,578 tCO₂/hour |
| RF pricing model R² | 0.949 |

### Scenario Comparison

| Scenario | Avg SMP (R/MWh) | Load Shedding (h) | Emissions (tCO₂/h) |
|----------|-----------------|-------------------|---------------------|
| Baseline | 1,780 | 712 (8.1%) | 19,578 |
| +5% Renewables | 1,762 | 118 (1.3%) | 19,407 |
| +25% Renewables | 1,692 | 90 (1.0%) | 18,717 |

**Key finding**: A 5% increase in renewable capacity reduces load shedding by **84%**, demonstrating a highly non-linear reliability dividend at the current capacity margin.

### Carbon Price Sensitivity

| Carbon Price (R/tCO₂) | Avg SMP (R/MWh) | Emissions change |
|-----------------------|-----------------|-----------------|
| R 0 (Phase I) | 1,780 | — |
| R 120 (Phase II) | 1,891 | None |
| R 250 | 2,010 | None |
| R 500 | 2,240 | None |

Carbon pricing raises system prices but does not reduce emissions under current dispatch constraints — the take-or-pay gas obligation and absence of low-carbon alternatives prevent fuel switching. This suggests direct capacity investment is needed alongside carbon pricing.

---

## Data

**Source**: [Eskom Data Portal](https://www.eskom.co.za/dataportal/) — hourly national system data, April 2021 to March 2025 (35,064 observations).

Key variables: RSA Contracted Demand · Residual Demand · Generation by technology (coal, gas, nuclear, hydro, wind, PV, CSP) · Outage factors (PCLF, UCLF, OCLF) · Load shedding proxy (MLR) · Renewable installed capacity

**Cost assumptions**: CSIR Energy Research Centre 2024 · EPRI IRP 2023-24 · IEA · IRENA

---

## Setup

```bash
git clone https://github.com/Siyaovo/south-africa-electricity-analysis.git
cd south-africa-electricity-analysis
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/01_demand_forecasting.ipynb` first, then `02_supply_optimization.ipynb`.

**Solver requirement**: The supply optimisation notebook requires [GLPK](https://www.gnu.org/software/glpk/) or [CBC](https://github.com/coin-or/Cbc):
```bash
# macOS
brew install glpk
```

---

## Technical Stack

- **Python 3.10+**
- **Optimisation**: Pyomo, GLPK/CBC solver
- **Statistical models**: statsmodels (ARIMA/SARIMA/SARIMAX)
- **Machine learning**: scikit-learn (Random Forest)
- **Deep learning**: TensorFlow/Keras (LSTM), PyTorch Lightning + pytorch-forecasting (TFT)
- **Data**: pandas, NumPy
- **Visualisation**: matplotlib, seaborn

---

## Authors

**Siyao Zhang** — M2 Sustainable Development Economics, Université Paris 1 Panthéon-Sorbonne

**Bérangère Tichet** — M2 Sustainable Development Economics, Université Paris 1 Panthéon-Sorbonne

*Supervised by Nicolas Maisonneuve-Bonteil, Deloitte*
