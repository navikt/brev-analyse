# %%
"""
Brev om dagpenger
"""

import pandas as pd
from statsmodels.formula.api import logit
from scipy.stats import chi2
from statsmodels.miscmodels.ordinal_model import OrderedModel


# %%
datasett_sav_sti = "../../data/Dagpenger brevmåling per 28042025.sav"
df = pd.read_spss(datasett_sav_sti)
# %%
# Logistisk regresjon
# Årsaker til at innbyggere tar kontakt om brev
df["dep"] = df["Nav_kontakt"]
df["dep"] = df["dep"].map(
    {"Jeg tok ikke kontakt med NAV om brevet": 0, "Jeg kontaktet NAV om brevet": 1}
)
df = df.dropna(subset=["dep"])
df["dep"] = df["dep"].astype(int)
# %%
# Logistisk regresjon
# Kontaktet de Nav og morsmål
model = logit("dep ~ C(Morsmål)", data=df)
res = model.fit()
print("Modell nummer 1")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())
# %%
# Log regresjon
# Kontaktet de Nav og morsmål
model = logit("dep ~ C(Alder)", data=df)
res = model.fit()
print("Modell nummer 2")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())
# %%
# Log regresjon
# Kontaktet Nav og brevtype + alder + morsmål
model = logit("dep ~ C(Brevtype) + C(Alder) + C(Morsmål)", data=df)
res = model.fit()
print("Modell nummer 3")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())
# %%
# Likelihood test for å se om alder forklarer mer enn nullhypotese - ingen variabel
# Lag begge modellene og sammenlign fit
model_with = logit("dep ~ C(Alder)", data=df).fit()
model_without = logit("dep ~ 1", data=df).fit()

# Extract log-likelihoods
llf_with = model_with.llf
llf_without = model_without.llf

# Degrees of freedom difference
df_diff = model_with.df_model - model_without.df_model

# Compute LR test statistic
lr_stat = 2 * (llf_with - llf_without)

# Compute p-value
p_value = chi2.sf(lr_stat, df_diff)

# Print results
print(f"LR statistic: {lr_stat:.3f}")
print(f"Degrees of freedom: {df_diff}")
print(f"P-value: {p_value:.5f}")

# %%
# Log regresjon
# Kontaktet Nav og alder og morsmål
model = logit("dep ~ C(Morsmål) + C(Alder)", data=df)
res = model.fit()
print("Modell nummer 4")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())

# %%
# Regresjoner om
# De som synes det er lett eller veldig lett å forstå vedtak
df["dep"] = df["Innvilgelse_1"].copy()
df["dep"] = df["dep"].map(
    {
        "Lett å forstå": 1,
        "Veldig lett å forstå": 1,
        "Verken lett eller vanskelig": 0,
        "Vanskelig å forstå": 0,
        "Veldig vanskelig å forstå": 0,
        "Jeg fant ikke forklaringen": 0,
    }
)
df = df.dropna(subset=["dep"])
df["dep"] = df["dep"].astype(int)
# %%
# Log regresjon - forstår begrunnelse og morsmål alder og kontaktet Nav
model = logit("dep ~ C(Morsmål) + C(Alder) + C(Nav_kontakt)", data=df)
res = model.fit()
print("Modell nummer 5")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())

# %%
# Før vi bytter avhengig variabel så må vi laste inn datasett på nytt
# For å bli kvitt manglende svar

df = pd.read_spss(datasett_sav_sti)
# %%
# Kombinere variablene om å forstå vedtak
df["dep_forstå"] = (
    df[["Innvilgelse_1", "Avslag_1", "Mangelbrev_1", "Stans_1"]]
    .bfill(axis=1)
    .iloc[:, 0]
)

# %%
# sett nivåer på ordinal variabel
df["dep_forstå"] = df["dep_forstå"].cat.set_categories(
    [
        "Veldig vanskelig å forstå",
        "Vanskelig å forstå",
        "Verken lett eller vanskelig",
        "Lett å forstå",
        "Veldig lett å forstå",
    ],
    ordered=True,
)
df = df.dropna(subset=["dep_forstå"])  # fjern tomme rader
# %%
# Ref https://github.com/statsmodels/statsmodels/issues/7418
# OrderedModel.from_formula('measure ~ C(factor1) + C(factor2) + C(factor1):C(factor2)', distr='logit', data=df)
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Alder) + C(Spraak) + C(Alder):C(Spraak)", distr="logit", data=df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 6")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Alder) + C(Morsmål) + C(Alder):C(Morsmål)", distr="logit", data=df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 7")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
model = OrderedModel.from_formula("dep_forstå ~ C(Tid_brukt)", distr="logit", data=df)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 8")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Klagerettigheter)", distr="logit", data=df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 9")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Antall_ganger_lest)", distr="logit", data=df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 10")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
# forståelse av begrunnelse i vedtak og brevtype
df["brev"] = df["Brevtype"].copy()
df["brev"] = df["brev"].str.replace("Ingen av disse", "Nan")

model = OrderedModel.from_formula("dep_forstå ~ C(brev)", distr="logit", data=df)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 11")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())


# %%
# forståelse av begrunnelse og alder
model = OrderedModel.from_formula(
    "dep_forstå ~ C(brev) + C(Alder)", distr="logit", data=df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 12")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
# forståelse av begrunnelse og alder og morsmål
model = OrderedModel.from_formula(
    "dep_forstå ~ C(brev) + C(Alder) + C(Morsmål)", distr="logit", data=df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 13")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
# forståelse av begrunnelse morsmål
model = OrderedModel.from_formula(
    "dep_forstå ~ C(brev) + C(Morsmål)", distr="logit", data=df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 14")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())

