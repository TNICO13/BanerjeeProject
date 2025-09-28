# time_series_ols_lagged_macros_year_dummies.py
# OLS with HAC robust SEs, lagged macro vars + contemporaneous year dummies

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

# -----------------------------
# 1) Load data
# -----------------------------
file_path = "allcvs.csv"
df = pd.read_csv(file_path)

# Rename for convenience
df = df.rename(columns={
    'GDP per capita growth (annual %) [NY.GDP.PCAP.KD.ZG]': 'gdp_pc_growth',
    'Foreign direct investment, net inflows (% of GDP) [BX.KLT.DINV.WD.GD.ZS]': 'fdi_in',
    'Foreign direct investment, net outflows (% of GDP) [BM.KLT.DINV.WD.GD.ZS]': 'fdi_out',
    'Services, value added (annual % growth) [NV.SRV.TOTL.KD.ZG]': 'srv_g',
    'Manufacturing, value added (annual % growth) [NV.IND.MANF.KD.ZG]': 'mfg_g',
    'Exports of goods and services (annual % growth) [NE.EXP.GNFS.KD.ZG]': 'exp_g',
    'Poland Base': 'poland'
})

# -----------------------------
# 2) Lag macro variables by 1 period
# -----------------------------
macros = ['fdi_in','fdi_out','srv_g','mfg_g','exp_g']
for m in macros:
    df[f"{m}_l1"] = df[m].shift(1)

# -----------------------------
# 3) Identify year dummy columns
# -----------------------------
year_cols = [col for col in df.columns if col.isdigit()]
# Drop one baseline year to avoid dummy trap
year_dummies = [c for c in year_cols if c != "1996"]

# -----------------------------
# 4) Build regression dataset
# -----------------------------
y = df['gdp_pc_growth']
X = df[[f"{m}_l1" for m in macros] + ['poland'] + year_dummies]

# Drop rows with NA from lagging
X = X.dropna()
y = y.loc[X.index]

# Add constant
X = sm.add_constant(X)

# -----------------------------
# 5) Fit OLS with HAC robust SEs
# -----------------------------
ols_model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags":2})

print("\n=== OLS WITH LAGGED MACROS + CONTEMPORANEOUS YEAR DUMMIES (HAC robust) ===")
print(ols_model.summary())

# -----------------------------
# 6) Breusch–Godfrey autocorrelation check
# -----------------------------
bg_stat, bg_pval, _, _ = acorr_breusch_godfrey(ols_model, nlags=2)
print("\nBreusch–Godfrey LM test (lags=2):")
print(f"LM stat={bg_stat:.3f}, p-value={bg_pval:.4f}")
