# %%
"""
Analyse av brev om uføretrygd jul 2025
"""

import pandas as pd
from statsmodels.formula.api import logit
from scipy.stats import chi2
from statsmodels.miscmodels.ordinal_model import OrderedModel

# %%
"""
Forbered datasett for analyse
"""
# last inn fullstendig datasett
datasett_sti = "../../data/uføretrygd_202512.pkl"
df = pd.read_pickle(datasett_sti)
# %%
# Dropp rader fra
# De som ikke leste brevet - ingen oppfølgingsspørsmål
# De som svarte at brevet de fikk var "ingen av disse" - ingen oppfølgingsspørsmål
df = df[df["Har_lest"] == "Ja"]
df = df[df["Brevtype"] != "Ingen av disse"]
# %%
# Dropp ubrukte kategorier
df["Har_lest"] = df["Har_lest"].cat.remove_categories(["Nei", "Unknown"])
df["Brevtype"] = df["Brevtype"].cat.remove_categories(["Ingen av disse"])
# %%
"""
Sammenligne svar fra enkelte grupper som svarte på undersøkelsen
"""
# %%
# Sammenlign svar fra de som får uføretrygd og jobber mot de som ikke jobber
# df["Brevtype"] = df["Brevtype"].cat.remove_categories(
#     ["Nav har avslått søknaden min om uføretrygd"]
# )
# kombiner svar fra mottagere og representanter
# df["Jobb_og_ufør"] = df["Jobb_og_ufør"].fillna(df["Rep_jobb_og_ufør"])
# velg gruppe for sammenligning
# df = df[df["Jobb_og_ufør"] == "Ja"]
# df = df[df["Jobb_og_ufør"] == "Nei"]
# %%
# Kjør analyse med og uten gruppen som fikk brev for over ett år siden
# Sammenlign fordelinger
# df = df[df["Når_fikkdu_brevet"] != "Mer enn et år siden"]
# %%
# Lag nye aldersgrupper for å tillate sammenligninger med regresjon
df = df[df["Alder"] != "Yngre enn 19 år"].copy()  # dropp svar
aldersgrupper = {
    "Yngre enn 19 år": "Under 40 år",
    "20 – 24 år": "Under 40 år",
    "25 – 29 år": "Under 40 år",
    "30 – 39 år": "Under 40 år",
    "40 – 49 år": "40 - 49 år",
    "50 – 59 år": "50 - 59 år",
    "60 eller eldre": "60 eller eldre",
}
df["Aldersgruppe"] = df["Alder"].map(aldersgrupper)
df["Aldersgruppe"] = df["Aldersgruppe"].astype("category")
# %%
"""
Logistiske regresjoner
"""
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
model = logit("dep ~ C(Aldersgruppe)", data=reg_df)
res = model.fit()
print("Modell nummer 2")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())
# %%
# Log regresjon
# Kontaktet Nav Aldersgruppe og morsmål
model = logit("dep ~ + C(Aldersgruppe) + C(Morsmål)", data=reg_df)
res = model.fit()
print("Modell nummer 3")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())
# %%
# Likelihood test for å se om Aldersgruppe forklarer mer enn nullhypotese - ingen variabel
# Lag begge modellene og sammenlign fit
model_with = logit("dep ~ C(Aldersgruppe)", data=reg_df).fit()
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
# Kontaktet Nav og Aldersgruppe og morsmål
model = logit("dep ~ C(Morsmål) + C(Aldersgruppe)", data=reg_df)
res = model.fit()
print("Modell nummer 4")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())

# %%
# Endrer avhengig variabel så kopierer df på nytt
reg_df = df.copy()
# %%
# Avhengig variabel
# Leste brevet flere ganger
reg_df["dep"] = reg_df["Antall_ganger"]
reg_df["dep"] = reg_df["dep"].map(
    {"Jeg leste brevet en gang": 0, "Jeg leste brevet flere ganger": 1}
)
reg_df = reg_df.dropna(subset=["dep"])
reg_df["dep"] = reg_df["dep"].astype(int)
# %%
# Log regresjon
# Leste brevet flere ganger og tidsbruk
model = logit("dep ~ C(Tidsbruk)", data=reg_df)
res = model.fit()
print("Modell nummer 5")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())
# %%
# Predikert sannsynlighet for å lese flere ganger
pred_df = pd.DataFrame({"Tidsbruk": df["Tidsbruk"].unique()})
pred_df["Predikert sannsynlighet"] = res.predict(pred_df)
print(pred_df)
# %%
# Leste brevet flere ganger og tidsbruk og brevtype
model = logit("dep ~ C(Brevtype)", data=reg_df)
res = model.fit()
print("Modell nummer 6")
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
# Log regresjon - forstår begrunnelse og morsmål Aldersgruppe og kontaktet Nav
model = logit("dep ~ C(Morsmål) + C(Aldersgruppe) + C(Kontaktet_Nav)", data=reg_df)
res = model.fit()
print("Modell nummer 7")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
# Endrer avhengig variabel så kopierer df på nytt
reg_df = df.copy()
# Dropp svar fra de som ikke fikk spørsmålet - for å bygge matrise
reg_df["Brevtype"] = reg_df["Brevtype"].cat.remove_categories(
    ["Nav har avslått søknaden min om uføretrygd"]
)
# %%
# kombiner svar fra mottagere og representanter
reg_df["Jobb_og_ufør"] = reg_df["Jobb_og_ufør"].fillna(reg_df["Rep_jobb_og_ufør"])
# %%
# Avhengig variabel
# Kombinerer jobb og uføretrygd
reg_df["dep"] = reg_df["Jobb_og_ufør"]
reg_df["dep"] = reg_df["dep"].map({"Nei": 0, "Ja": 1})
reg_df = reg_df.dropna(subset=["dep"])
reg_df["dep"] = reg_df["dep"].astype(int)

# %%
# Regresjon: Kombinere jobb og uføretrygd
model = logit("dep ~ C(Brevtype)", data=reg_df)
res = model.fit()
print("Modell nummer X")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())

# %%
"""
Ordinale regresjoner
"""
# Endrer avhengig variabel så kopierer df på nytt
reg_df = df.copy()
# %%
# Kombinere variablene om å forstå vedtak
# kombinerer spørsmål om begrunnelse for alle brevtyper
reg_df["dep_forstå"] = (
    reg_df[["Innvilgelse_hvorfor", "Avslag_hvorfor", "Endret_hvorfor"]]
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
    "dep_forstå ~ C(Aldersgruppe) + C(Språket_brevet) + C(Aldersgruppe):C(Språket_brevet)",
    distr="logit",
    data=reg_df,
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 8")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
# Forståelse vs Brevtype og alder
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Brevtype) + C(Aldersgruppe)",
    distr="logit",
    data=reg_df,
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 9")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
# Forståelse vs Brevtype og morsmål
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Brevtype) + C(Morsmål)",
    distr="logit",
    data=reg_df,
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 10")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Brevtype) + C(Aldersgruppe) + C(Morsmål) + C(Aldersgruppe):C(Morsmål)",
    distr="logit",
    data=reg_df,
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 11")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Tidsbruk)", distr="logit", data=reg_df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 12")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Klagerettigheter)", distr="logit", data=reg_df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 13")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Antall_ganger)", distr="logit", data=reg_df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 14")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
# forståelse av begrunnelse i vedtak og brevtype
reg_df["brev"] = reg_df["Brevtype"].copy()
reg_df["brev"] = reg_df["brev"].str.replace("Ingen av disse", "Nan")

model = OrderedModel.from_formula("dep_forstå ~ C(brev)", distr="logit", data=reg_df)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 15")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())


# %%
# forståelse av begrunnelse og Aldersgruppe
model = OrderedModel.from_formula(
    "dep_forstå ~ C(brev) + C(Aldersgruppe)", distr="logit", data=reg_df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 16")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
# forståelse av begrunnelse og Aldersgruppe og morsmål
model = OrderedModel.from_formula(
    "dep_forstå ~ C(brev) + C(Aldersgruppe) + C(Morsmål)", distr="logit", data=reg_df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 17")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
# forståelse av begrunnelse morsmål
model = OrderedModel.from_formula(
    "dep_forstå ~ C(brev) + C(Morsmål)", distr="logit", data=reg_df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 18")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())

# %%
# Har det noe å si hvor lenge siden de leste brevet?
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Når_fikkdu_brevet)", distr="logit", data=reg_df
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 19")
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
        "Avslag_hvorfor",
        "Avslag_informasjon",
        "Endret_hvorfor",
        "Endret_informasjon",
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
# %%
tabell.to_excel("../../data/tabell_uføretrygd_202512.xlsx", index=False)
# %%
_ = df.groupby(["Brevtype", "Antall_ganger","Tidsbruk"]).agg({"id":"count"})
_.reset_index(inplace=True)
_.to_excel("../../data/tidsbruk_uforetrygd_202512.xlsx", index=False)
# %%
