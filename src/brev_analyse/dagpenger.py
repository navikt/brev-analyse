# %%
"""
Brev om dagpenger

Del 1 forbereder datasett til analysen

Del 2 gjennomføres analysene om dagpenger for begge tidsperioder
Del 3 gjennomfører analyse for perioden før sommer 2025
"""

import pandas as pd
from statsmodels.formula.api import logit
from scipy.stats import chi2
from statsmodels.miscmodels.ordinal_model import OrderedModel

# %%
"""
Del 1: Forbered datasett til analyser om dagpenger

Lag to datasett. Ett for hele tidsperioden, og ett for perioden før sommerferie

Datasettene skal ha samme datatyper og variabler til slutt slik at analysen kan gjentas for hver tidsperiode

Steg
* lagrer fullstendige spørsmålsformuleringer som dictionary
* fjern første rad med fullstendige spørsmålsformuleringer
* erklære de kategoriske variablene i datasettene
* bruk kortere variabelnavn i stedet for stavelsen av alle spørsmål
* lag subset-datasett basert på identiske respondenter fra rådata
* gjenskap uttrekk fra spss med rådata fra task analytics
    > bruker respondent ID
    > verifiserer at unike respondent IDer er like i begge datasett
    > verifiserer at samme variabler er i datasettene for begge tidsperioder
    > verifiserer at samme datatypene er i datasettene for begge tidsperioder

"""
# %%
# last inn fullstendig datasett
datasett_sti = "../../data/dagpenger_202506.pkl"
df = pd.read_pickle(datasett_sti)
# %%
"""
Del 3: Analyse før sommer 2025
"""
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
