# time_series_ols_post2008_poland_clean.py

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

# -----------------------------
# 1) Load data
# -----------------------------
path = "allcvs.csv"
df = pd.read_csv(path)

# Rename known columns
rename_map = {
    'Time': 'year',
    'GDP per capita growth (annual %) [NY.GDP.PCAP.KD.ZG]': 'gdp_pc_growth',
    'Foreign direct investment, net inflows (% of GDP) [BX.KLT.DINV.WD.GD.ZS]': 'fdi_in',
    'Foreign direct investment, net outflows (% of GDP) [BM.KLT.DINV.WD.GD.ZS]': 'fdi_out',
    'Services, value added (annual % growth) [NV.SRV.TOTL.KD.ZG]': 'srv_g',
    'Manufacturing, value added (annual % growth) [NV.IND.MANF.KD.ZG]': 'mfg_g',
    'Exports of goods and services (annual % growth) [NE.EXP.GNFS.KD.ZG]': 'exp_g',
    'Poland Base': 'poland',
    'Pre Post 2005': 'post2005'
}
df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})

# -----------------------------
# 2) Dummy for post-2008, drop post2005
# -----------------------------
df["post2008"] = (df["year"] >= 2008).astype(int)
if "post2005" in df.columns:
    df = df.drop(columns=["post2005"])

if "exp_g" in df.columns:
    df = df.drop(columns=["exp_g"])

# -----------------------------
# 4) Year dummies (contemporaneous)
# -----------------------------
year_dummy_cols = [c for c in df.columns if str(c).isdigit()]

# -----------------------------
# 5) Poland interactions
# -----------------------------
df["fdi_in_poland"]  = df["fdi_in"]  * df["poland"]
df["fdi_out_poland"] = df["fdi_out"] * df["poland"]
df["srv_g_poland"]   = df["srv_g"]   * df["poland"]
df["mfg_g_poland"]   = df["mfg_g"]   * df["poland"]


# -----------------------------
# 6) Design matrix
# -----------------------------
y = df["gdp_pc_growth"]

X_cols = (
    ["fdi_in","fdi_out","srv_g","mfg_g",] +  # contemporaneous only
    ["poland","post2008",
     "fdi_in_poland","fdi_out_poland",
     "srv_g_poland","mfg_g_poland"] +
    year_dummy_cols
)

X = df[X_cols].copy()

# Drop NA from lag
mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask]
y = y.loc[mask]

X = sm.add_constant(X)

# -----------------------------
# 7) Fit OLS (HAC robust SEs)
# -----------------------------
model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
print("\n=== OLS with post-2008 dummy, Poland interactions, contemporaneous macros, HAC robust SEs ===")
print(model.summary())

# -----------------------------
# 8) Residual autocorrelation check
# -----------------------------
bg_stat, bg_pval, _, _ = acorr_breusch_godfrey(model, nlags=2)
print("\nBreusch–Godfrey LM test (lags=2):")
print(f"LM stat={bg_stat:.3f}, p-value={bg_pval:.4f}")
