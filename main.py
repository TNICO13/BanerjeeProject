# time_series_ols_years_poland_post2008_infl_inrex.py
# OLS with HAC robust SEs, year dummies + Poland interactions + post2008 + inflation + exchange rate

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
import reportlab as reportlab

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
    'Inflation, consumer prices (annual %) [FP.CPI.TOTL.ZG]': 'infl',
    'Adj YoY Ex Rate': 'exrate',
    'Poland Base': 'poland'
})

# -----------------------------
# 2) Year dummies
# -----------------------------
year_cols = [col for col in df.columns if col.isdigit()]
year_dummies = [c for c in year_cols if c != "1996"]  # drop baseline year

# -----------------------------
# 3) Structural break dummy for 2008
# -----------------------------
df['post2008'] = (df['Time'] >= 2008).astype(int)
df['post2008_poland'] = df['post2008'] * df['poland']

# -----------------------------
# 4) Poland interactions with covariates
# -----------------------------
df['fdi_in_poland']   = df['fdi_in']   * df['poland']
df['fdi_out_poland']  = df['fdi_out']  * df['poland']
df['srv_g_poland']    = df['srv_g']    * df['poland']
df['mfg_g_poland']    = df['mfg_g']    * df['poland']
df['infl_poland']     = df['infl']     * df['poland']
df['exrate_poland']   = df['exrate']   * df['poland']

# -----------------------------
# 5) Build regression dataset
# -----------------------------
y = df['gdp_pc_growth']
X = df[[

           'fdi_in','fdi_out','srv_g','mfg_g',

        'poland','post2008'
        , 'post2008_poland',

        'fdi_in_poland','fdi_out_poland',
        'srv_g_poland',
        'mfg_g_poland',
        ]

       + year_dummies

       ]

# Drop NA rows (if any)
X = X.dropna()
y = y.loc[X.index]

# Add constant
X = sm.add_constant(X)

# -----------------------------
# 6) Fit OLS with HAC robust SEs
# -----------------------------
ols_model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags":2})

print("\n=== OLS")
print(ols_model.summary())

# -----------------------------
# 7) Breusch–Godfrey autocorrelation check
# -----------------------------
bg_stat, bg_pval, _, _ = acorr_breusch_godfrey(ols_model, nlags=2)
print("\nBreusch–Godfrey LM test (lags=2):")
print(f"LM stat={bg_stat:.3f}, p-value={bg_pval:.4f}")



from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# Save regression summary as text
summary_text = ols_model.summary().as_text()
bg_text = f"\nBreusch–Godfrey LM test (lags=2):\nLM stat={bg_stat:.3f}, p-value={bg_pval:.4f}"

doc = SimpleDocTemplate("ols_results.pdf", pagesize=letter)
styles = getSampleStyleSheet()
flowables = []

flowables.append(Paragraph("OLS Results ", styles["Heading1"]))
flowables.append(Preformatted(summary_text, styles["Code"]))
flowables.append(Spacer(1, 12))
flowables.append(Preformatted(bg_text, styles["Code"]))

doc.build(flowables)

