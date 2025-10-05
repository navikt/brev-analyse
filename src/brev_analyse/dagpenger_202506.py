# %%
"""
Analyse av brev om dagpenger før sommer 2025
"""

import pandas as pd
from statsmodels.formula.api import logit
from scipy.stats import chi2
from statsmodels.miscmodels.ordinal_model import OrderedModel

# %%
# last inn fullstendig datasett
datasett_sti = "../../data/dagpenger_202506.pkl"
df = pd.read_pickle(datasett_sti)
# %%
# Lag kopi av dataframe når du endrer avhengig variabel
reg_df = df.copy()
# %%
# Logistisk regresjon
# Årsaker til at innbyggere tar kontakt om brev
reg_df["dep"] = reg_df["Kontaktet_Nav"]
reg_df["dep"] = reg_df["dep"].map(
    {"Jeg tok ikke kontakt med NAV om brevet": 0, "Jeg kontaktet NAV om brevet": 1}
)
reg_df = reg_df.dropna(subset=["dep"])
reg_df["dep"] = reg_df["dep"].astype(int)
# %%
# Logistisk regresjon
# Kontaktet de Nav og morsmål
model = logit("dep ~ C(Morsmål)", data=reg_df)
res = model.fit()
print("Modell nummer 1")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())
# %%
# Log regresjon
# Kontaktet de Nav og morsmål
model = logit("dep ~ C(Alder)", data=reg_df)
res = model.fit()
print("Modell nummer 2")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())
# %%
# Log regresjon
# Kontaktet Nav og brevtype + alder + morsmål
model = logit("dep ~ C(Brevtype) + C(Alder) + C(Morsmål)", data=reg_df)
res = model.fit()
print("Modell nummer 3")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())
# %%
# Likelihood test for å se om alder forklarer mer enn nullhypotese - ingen variabel
# Lag begge modellene og sammenlign fit
model_with = logit("dep ~ C(Alder)", data=reg_df).fit()
model_without = logit("dep ~ 1", data=reg_df).fit()

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
model = logit("dep ~ C(Morsmål) + C(Alder)", data=reg_df)
res = model.fit()
print("Modell nummer 4")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())

# %%
# Endrer avhengig variabel så kopierer df på nytt
reg_df = df.copy()
# %%
# Regresjoner om
# De som synes det er lett eller veldig lett å forstå vedtak
reg_df["dep"] = reg_df["Innvilgelse_hvorfor"].copy()
reg_df["dep"] = reg_df["dep"].map(
    {
        "Lett å forstå": 1,
        "Veldig lett å forstå": 1,
        "Verken lett eller vanskelig": 0,
        "Vanskelig å forstå": 0,
        "Veldig vanskelig å forstå": 0,
        "Jeg fant ikke forklaringen": 0,
    }
)
reg_df = reg_df.dropna(subset=["dep"])
reg_df["dep"] = reg_df["dep"].astype(int)
# %%
# Log regresjon - forstår begrunnelse og morsmål alder og kontaktet Nav
model = logit("dep ~ C(Morsmål) + C(Alder) + C(Kontaktet_Nav)", data=reg_df)
res = model.fit()
print("Modell nummer 5")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())

# %%
# Endrer avhengig variabel så kopierer df på nytt
reg_df = df.copy()
# %%
# Kombinere variablene om å forstå vedtak
# kombinerer spørsmål om begrunnelse for alle brevtyper
reg_df["dep_forstå"] = (
    reg_df[["Innvilgelse_hvorfor", "Avslag_hvorfor", "Mangel_hvorfor", "Stans_hvorfor"]]
    .bfill(axis=1)
    .iloc[:, 0]
)

# %%
# sett nivåer på ordinal variabel
reg_df["dep_forstå"] = reg_df["dep_forstå"].cat.set_categories(
    [
        "Veldig vanskelig å forstå",
        "Vanskelig å forstå",
        "Verken lett eller vanskelig",
        "Lett å forstå",
        "Veldig lett å forstå",
    ],
    ordered=True,
)
reg_df = reg_df.dropna(subset=["dep_forstå"])  # fjern tomme rader
# %%
# Ref https://github.com/statsmodels/statsmodels/issues/7418
# I en regresjon med flere ordinale variabler kombinerer vi disse med egen syntaks
# OrderedModel.from_formula('dependent ~ C(factor1) + C(factor2) + C(factor1):C(factor2)', distr='logit', data=reg_df)
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Alder) + C(Språket_brevet) + C(Alder):C(Språket_brevet)",
    distr="logit",
    data=reg_df,
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 6")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Alder) + C(Morsmål) + C(Alder):C(Morsmål)",
    distr="logit",
    data=reg_df,
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 7")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Tidsbruk)", distr="logit", data=reg_df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 8")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Klagerettigheter)", distr="logit", data=reg_df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 9")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Antall_ganger)", distr="logit", data=reg_df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 10")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
# forståelse av begrunnelse i vedtak og brevtype
reg_df["brev"] = reg_df["Brevtype"].copy()
reg_df["brev"] = reg_df["brev"].str.replace("Ingen av disse", "Nan")

model = OrderedModel.from_formula("dep_forstå ~ C(brev)", distr="logit", data=reg_df)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 11")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())


# %%
# forståelse av begrunnelse og alder
model = OrderedModel.from_formula(
    "dep_forstå ~ C(brev) + C(Alder)", distr="logit", data=reg_df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 12")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
# forståelse av begrunnelse og alder og morsmål
model = OrderedModel.from_formula(
    "dep_forstå ~ C(brev) + C(Alder) + C(Morsmål)", distr="logit", data=reg_df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 13")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
# forståelse av begrunnelse morsmål
model = OrderedModel.from_formula(
    "dep_forstå ~ C(brev) + C(Morsmål)", distr="logit", data=reg_df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 14")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())


# %%
"""
Tabeller for rapporten
"""
# %%
_ = df.copy()
_ = pd.melt(
    frame=df,
    id_vars=["id", "Brevtype"],
    value_vars=[
        "Innvilgelse_hvorfor",
        "Innvilgelse_informasjon",
        "Innvilgelse_gjøre",
        "Avslag_hvorfor",
        "Avslag_informasjon",
        "Mangel_hvorfor",
        "Mangel_informasjon",
        "Stans_hvorfor",
        "Stans_informasjon",
        "Klagerettigheter",
        "Finne_informasjon",
        "Språket_brevet",
        "Overskrift",
    ],
    var_name="spørsmål",
    value_name="svar",
)
_ = _.dropna(subset=["svar"])
# %%
likert_skala = [
    "Veldig vanskelig å forstå",
    "Vanskelig å forstå",
    "Verken lett eller vanskelig",
    "Lett å forstå",
    "Veldig lett å forstå",
    "Jeg fant ikke forklaringen",
]
tabell = (
    _.groupby(["Brevtype", "spørsmål", "svar"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=likert_skala, fill_value=0)
    .reset_index()
)
tabell
# %%
tabell.to_excel("../../data/tabell.xlsx", index=False)
# %%
