# time_series_ols_years_poland_interactions_post2008.py
# OLS with HAC robust SEs, year dummies + Poland interactions + post2008 dummy

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
    'Poland Base': 'poland'
})

# -----------------------------
# 2) Identify year dummies
# -----------------------------
year_cols = [col for col in df.columns if col.isdigit()]
# Drop one baseline year (1996) to avoid dummy trap
year_dummies = [c for c in year_cols if c != "1996"]

# -----------------------------
# 3) Structural break dummy for 2008
# -----------------------------
# Assumes "Time" column exists in the CSV with years
df['post2008'] = (df['Time'] >= 2008).astype(int)

# -----------------------------
# 4) Poland interactions
# -----------------------------
df['fdi_in_poland']  = df['fdi_in']  * df['poland']
df['fdi_out_poland'] = df['fdi_out'] * df['poland']
df['srv_g_poland']   = df['srv_g']   * df['poland']
df['mfg_g_poland']   = df['mfg_g']   * df['poland']

# -----------------------------
# 5) Build regression dataset
# -----------------------------
y = df['gdp_pc_growth']
X = df[['fdi_in','fdi_out','srv_g','mfg_g','post2008',
        'fdi_in_poland','fdi_out_poland','srv_g_poland','mfg_g_poland'] + year_dummies]

# Drop NA rows (if any)
X = X.dropna()
y = y.loc[X.index]

# Add constant
X = sm.add_constant(X)

# -----------------------------
# 6) Fit OLS with HAC robust SEs
# -----------------------------
ols_interact = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags":2})

print("\n=== OLS WITH YEAR DUMMIES + POLAND INTERACTIONS + POST2008 (HAC robust) ===")
print(ols_interact.summary())

# -----------------------------
# 7) Breusch–Godfrey autocorrelation check
# -----------------------------
bg_stat, bg_pval, _, _ = acorr_breusch_godfrey(ols_interact, nlags=2)
print("\nBreusch–Godfrey LM test (lags=2):")
print(f"LM stat={bg_stat:.3f}, p-value={bg_pval:.4f}")
