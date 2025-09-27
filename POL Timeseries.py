# Time-series regression for Poland (1992–2024) with a 2005 level-shift
# Requirements: pandas, numpy, statsmodels, matplotlib (optional for plots)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# -----------------------------
# 1) Build the DataFrame
# -----------------------------
data = [
    # Year, LN, fdi_in, fdi_out, srv_g, mfg_g, exp_g
    (1992, 8.484704831, 0.718699594, 0.013780376, 7.5, 3.2, 10),
    (1993, 8.518861617, 1.785655583, 0.018741575, 7.5, 3.2, 10),
    (1994, 8.568326368, 1.692182748, 0.026172426, 7.5, 3.2, 10),
    (1995, 8.643278686, 2.561633807, 0.029403832, 7.5, 3.2, 10),
    (1996, 8.701672097, 2.798280777, 0.034201209, 9.573026007, -1.072438197, 11.33167543),
    (1997, 8.762666641, 3.070785068, 0.02939448, 9.665998633, -0.553477504, 12.20128197),
    (1998, 8.807489062, 3.641554854, 0.190549792, 7.201386223, 0.201706905, 14.24357958),
    (1999, 8.853493235, 4.339078379, 0.098415711, 6.824236715, 2.67301425, -2.514461769),
    (2000, 8.909448361, 5.397403661, 0.00462552, 8.123551709, -0.187094081, 23.60728893),
    (2001, 8.921986193, 2.95949603, -0.066206799, 2.135482854, -1.289012878, 3.054803387),
    (2002, 8.941282858, 2.048629658, 0.095145352, 3.2470285, 2.084825316, 4.826011134),
    (2003, 8.976584994, 2.457434971, 0.497343477, 2.110218137, 12.49186113, 14.07969697),
    (2004, 9.026824631, 5.411508457, 0.83467094, 3.435716591, 15.1944111, 4.715669827),
    (2005, 9.059351982, 3.596417956, 1.346580186, 2.905170804, 5.693048623, 10.05950191),
    (2006, 9.120159451, 6.207906066, 3.123178374, 5.088910463, 15.46127912, 15.26544551),
    (2007, 9.186119835, 5.825021771, 1.765355565, 5.642319907, 14.13879575, 10.17662432),
    (2008, 9.228887485, 2.720999374, 0.871152949, 4.969028705, 8.971617735, 6.992556025),
    (2009, 9.254025041, 3.18105495, 1.345229013, 1.819582481, 1.204117974, -6.035552783),
    (2010, 9.288078664, 3.944894621, 2.015847217, 0.46639101, 10.67695391, 12.58972256),
    (2011, 9.338761153, 3.571289577, 0.90575224, 3.005032203, 5.54237645, 7.744991922),
    (2012, 9.353768225, 1.528860914, 0.32500339, 2.440385907, 2.80069094, 3.920005375),
    (2013, 9.361190928, 0.258983447, -0.597861936, 1.905051536, -3.693788943, 5.060498359),
    (2014, 9.400395471, 3.850891765, 1.248399457, 2.061688154, 11.72612998, 5.497597853),
    (2015, 9.444425607, 3.29546178, 1.055505995, 4.550402403, 7.172845477, 6.608697412),
    (2016, 9.47471691, 3.816087214, 2.958418443, 3.317237644, 5.320919002, 8.959246327),
    (2017, 9.524833199, 2.378128156, 0.744572715, 7.198431844, 1.851862712, 9.091990364),
    (2018, 9.585420842, 3.348039546, 0.410012076, 6.793683342, 5.681484534, 6.757367674),
    (2019, 9.630451637, 3.154556492, 0.890848612, 5.071005296, 6.379084384, 5.33745532),
    (2020, 9.621802319, 3.314990878, 0.837247273, -1.581812539, -5.576925586, -1.075662343),
    (2021, 9.703121609, 5.443067847, 1.482942882, 9.344332037, -0.790623206, 12.30316774),
    (2022, 9.758672449, 6.006260517, 1.811797542, 5.159452512, 12.34268414, 7.405650164),
    (2023, 9.764822032, 4.223761747, 1.818817092, 1.373171169, 4.857244164, 3.716713715),
    (2024, 9.797258883, 2.017718599, 0.848806199, 3.873229695, 0.749998707, 1.970598412),
]
df = pd.DataFrame(data, columns=[
    "year","LN","fdi_in","fdi_out","srv_g","mfg_g","exp_g"
]).set_index("year")

# -----------------------------
# 2) Structural break dummy (post-2005)
# -----------------------------
df["post2005"] = (df.index >= 2005).astype(int)

# -----------------------------
# 3) Quick diagnostics (optional)
# -----------------------------
def adf(name, series):
    s = series.dropna()
    stat, p, *_ = adfuller(s, autolag="AIC")
    print(f"ADF {name:>10s} | stat={stat: .3f}  p={p: .4f}")

print("ADF tests (levels):")
for col in ["LN","fdi_in","fdi_out","srv_g","mfg_g","exp_g"]:
    adf(col, df[col])

# -----------------------------
# 4) Static time-series regression with HAC SEs
# -----------------------------
X_static = df[["fdi_in","fdi_out","srv_g","mfg_g","exp_g","post2005"]]
X_static = sm.add_constant(X_static)
y = df["LN"]

ols_static = sm.OLS(y, X_static).fit(cov_type="HAC", cov_kwds={"maxlags":2})
print("\n=== STATIC OLS (HAC robust) ===")
print(ols_static.summary())

# Autocorrelation check on residuals (Breusch–Godfrey)
bg_test = acorr_breusch_godfrey(ols_static, nlags=2)
print("\nBreusch–Godfrey LM test (lags=2):")
print(f"LM stat={bg_test[0]:.3f}, p-value={bg_test[1]:.4f}")

# -----------------------------
# 5) Dynamic regression (ARDL(1) style): add lagged LN and lags of regressors
# -----------------------------
df["LN_l1"]     = df["LN"].shift(1)
df["fdi_in_l1"] = df["fdi_in"].shift(1)
df["fdi_out_l1"]= df["fdi_out"].shift(1)
df["srv_g_l1"]  = df["srv_g"].shift(1)
df["mfg_g_l1"]  = df["mfg_g"].shift(1)
df["exp_g_l1"]  = df["exp_g"].shift(1)

dyn_cols = ["LN_l1","fdi_in","fdi_in_l1","fdi_out","fdi_out_l1",
            "srv_g","srv_g_l1","mfg_g","mfg_g_l1","exp_g","exp_g_l1","post2005"]
X_dyn = sm.add_constant(df[dyn_cols])
y_dyn = df["LN"]

# Drop initial NA from lags
mask = X_dyn.notna().all(axis=1)
ols_dyn = sm.OLS(y_dyn[mask], X_dyn[mask]).fit(cov_type="HAC", cov_kwds={"maxlags":2})
print("\n=== DYNAMIC OLS with lags (HAC robust) ===")
print(ols_dyn.summary())

# -----------------------------
# 6) Optional: Allow slope changes after 2005 (interaction model)
#    Tests whether relationships changed post-shift, not just the level.
# -----------------------------
for v in ["fdi_in","fdi_out","srv_g","mfg_g","exp_g"]:
    df[f"{v}_x_post2005"] = df[v] * df["post2005"]

inter_cols = ["fdi_in","fdi_out","srv_g","mfg_g","exp_g","post2005",
              "fdi_in_x_post2005","fdi_out_x_post2005","srv_g_x_post2005",
              "mfg_g_x_post2005","exp_g_x_post2005"]
X_inter = sm.add_constant(df[inter_cols])
ols_inter = sm.OLS(y, X_inter).fit(cov_type="HAC", cov_kwds={"maxlags":2})
print("\n=== INTERACTION OLS (HAC robust) ===")
print(ols_inter.summary())

# Compare restricted (no interactions) vs unrestricted (with interactions)
fstat, fpval, df_denom = ols_inter.compare_f_test(ols_static)
print("\nF-test: do post-2005 slopes differ (interactions jointly)?")
print(f"F={fstat:.3f}, p={fpval:.4f}, df_denom={df_denom}")

# -----------------------------
# 7) (Optional) Residual ACF/PACF for the chosen model
# -----------------------------
# Choose which residuals to inspect:
resid = ols_dyn.resid  # or ols_static.resid

plt.figure()
plot_acf(resid, lags=10)
plt.title("Residual ACF (dynamic model)")
plt.show()

plt.figure()
plot_pacf(resid, lags=10, method="ywm")
plt.title("Residual PACF (dynamic model)")
plt.show()
