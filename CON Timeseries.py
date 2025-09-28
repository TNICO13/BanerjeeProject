# time_series_regression_newdata.py
# Static & dynamic time-series regression with HAC robust errors

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# -----------------------------
# 1) Build the DataFrame
# -----------------------------
data = [
    # year, LN, fdi_in, fdi_out, srv_g, mfg_g, exp_g
    (1992,8.820506337,4.031617039,0.392956006,3.67170924,6.319930147,8.12483765),
    (1993,8.803347148,4.398055205,0.482085723,3.67779705,6.175468663,7.87637733),
    (1994,8.827847405,4.453463874,0.688478465,3.952823267,6.700955515,7.909800644),
    (1995,8.870471593,3.252387989,0.070259306,3.793975014,6.452322316,7.951487397),
    (1996,8.909719852,2.772160056,0.101884396,2.8832032,3.407764619,7.423438913),
    (1997,8.965759913,3.680167614,0.506534889,5.074730388,9.658216034,14.74886551),
    (1998,9.003674388,4.91108111,0.342697357,3.411915066,6.65095431,6.533233134),
    (1999,9.023396113,4.203583703,0.362024036,2.765846966,0.867085382,0.541277063),
    (2000,9.068120878,4.595935259,0.494033515,3.616850241,9.337964441,10.24226357),
    (2001,9.112305744,4.806003543,0.873258545,4.155443803,7.865203931,9.43329796),
    (2002,9.160352292,5.817455151,0.694167324,3.836589684,5.441091927,6.21226516),
    (2003,9.216170614,3.160020736,1.546633589,4.808386719,7.086172582,7.657402108),
    (2004,9.275042321,4.517232373,1.273870957,3.962792619,7.917783639,15.04067279),
    (2005,9.343970971,9.358752462,4.164028848,6.270960261,7.641741327,14.79562579),
    (2006,9.420982983,7.0898736,4.012316633,6.128690604,9.402731709,12.56875885),
    (2007,9.493485793,12.09192663,9.157388959,6.422669464,5.151485434,10.6516933),
    (2008,9.504062868,9.181113405,7.462304946,1.411487704,0.062393673,4.128095422),
    (2009,9.402687171,1.154160958,1.25167552,-5.83941255,-15.17689755,-12.54740541),
    (2010,9.423802115,1.278780685,-0.921984524,1.36577668,12.22587155,13.54370943),
    (2011,9.459396175,3.912263478,0.774315412,1.938784417,5.834207491,10.44212628),
    (2012,9.474778989,3.500106684,1.543218627,2.237405515,0.55247987,4.711918923),
    (2013,9.489413609,1.412685805,0.995893527,2.00290717,0.363544219,2.905483909),
    (2014,9.521314669,3.180020585,2.100077829,1.667779826,5.660505348,4.304826698),
    (2015,9.55718564,0.600005731,-0.404922648,2.633602383,5.451175844,3.678660543),
    (2016,9.585609457,9.451503901,7.920894666,2.915862726,2.197565582,4.114821339),
    (2017,9.631799101,2.07357112,0.286980048,3.839948416,4.653206261,6.639147922),
    (2018,9.673073155,-3.00383543,-4.714970927,3.83193163,4.709306283,4.277880057),
    (2019,9.704922388,11.15988596,9.225709268,2.563227415,3.773760346,3.388935606),
    (2020,9.670644267,16.46098003,14.54321854,-2.280428561,-4.567186563,-4.236352471),
    (2021,9.738233934,7.800755421,6.222295047,6.572325222,5.706993311,11.17062343),
    (2022,9.753633288,2.657044276,0.571254744,3.490591974,0.629828616,6.774211194),
    (2023,9.751434682,-0.757073582,-2.507311887,1.661591126,-2.120199675,-1.912091921),
    (2024,9.762101242,-0.836907966,-2.859852697,1.160101294,-0.739622504,0.211110092),
]
df = pd.DataFrame(data, columns=["year","LN","fdi_in","fdi_out","srv_g","mfg_g","exp_g"]).set_index("year")

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
bg_stat, bg_pval, _, _ = acorr_breusch_godfrey(ols_static, nlags=2)
print("\nBreusch–Godfrey LM test (lags=2):")
print(f"LM stat={bg_stat:.3f}, p-value={bg_pval:.4f}")

# -----------------------------
# 5) Dynamic regression (ARDL(1) style): add lagged LN and lags of regressors
# -----------------------------
df["LN_l1"]     = df["LN"].shift(1)
df["fdi_in_l1"] = df["fdi_in"].shift(1)
df["fdi_out_l1"]= df["fdi_out"].shift(1)
df["srv_g_l1"]  = df["srv_g"].shift(1)
df["mfg_g_l1"]  = df["mfg_g"].shift(1)
df["exp_g_l1"]  = df["exp_g"].shift(1)

dyn_cols = ["fdi_in","fdi_in_l1","fdi_out","fdi_out_l1",
            "srv_g","srv_g_l1","mfg_g","mfg_g_l1","exp_g","exp_g_l1","post2005"]
X_dyn = sm.add_constant(df[dyn_cols])
y_dyn = df["LN"]

mask = X_dyn.notna().all(axis=1) & y_dyn.notna()
ols_dyn = sm.OLS(y_dyn[mask], X_dyn[mask]).fit(cov_type="HAC", cov_kwds={"maxlags":2})
print("\n=== DYNAMIC OLS with lags (HAC robust) ===")
print(ols_dyn.summary())

# -----------------------------
# 6) Optional: Allow slope changes after 2005 (interaction model)
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
resid = ols_dyn.resid  # or ols_static.resid

plt.figure()
plot_acf(resid, lags=10)
plt.title("Residual ACF (dynamic model)")
plt.show()

plt.figure()
plot_pacf(resid, lags=10, method="ywm")
plt.title("Residual PACF (dynamic model)")
plt.show()
