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

from get_answers import get_survey_questions

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
datasett_ta_sti = "../../data/uttrekk brev dagpenger 20250916.xlsx"
df = pd.read_excel(datasett_ta_sti)
# %%
# last inn forrige datasett fra spss
datasett_sav_sti = "../../data/Dagpenger brevmåling per 28042025.sav"
df_spss = pd.read_spss(datasett_sav_sti)
# %%
q = get_survey_questions(dataframe=df_spss)
questions = get_survey_questions(dataframe=df)
# %%
# behold første rad, dropp nummer 2
df = df.iloc[1:]
# %%
"""
Erklære hvilke variabler er kategoriske
"""
cat_col_pattern = [col for col in df.columns if col.startswith("answers.")]
df[cat_col_pattern] = df[cat_col_pattern].astype("category")
# %%
# ny mapping for spørsmål ID til tekst
questions_short = {
    "answers.t": "Brevtype",
    "answers.segment21": "Når_fikkdu_brevet",
    "answers.c": "Har_lest",
    "answers.segment": "Utbetaling_stopp_årsak",
    "answers.segment4": "Overskrift",
    "answers.segment7": "Innvilgelse_hvorfor",
    "answers.segment8": "Innvilgelse_informasjon",
    "answers.segment9": "Innvilgelse_gjøre",
    "answers.segment10": "Avslag_hvorfor",
    "answers.segment11": "Avslag_informasjon",
    "answers.segment12": "Mangel_hvorfor",
    "answers.segment22": "Mangel_informasjon",
    "answers.segment14": "Stans_hvorfor",
    "answers.segment15": "Stans_informasjon",
    "answers.segment16": "Klagerettigheter",
    "answers.segment3": "Finne_informasjon",
    "answers.segment18": "Språket_brevet",
    "answers.segment5": "Antall_ganger",
    "answers.segment6": "Tidsbruk",
    "answers.segment17": "Kontaktet_Nav",
    "answers.segment19": "Morsmål",
    "answers.segment20": "Alder",
}
# %%
# bytt ut alle kolonner som starter med answers. til de kortere formuleringene
df.rename(columns=questions_short, inplace=True)
# %%
"""
Slik kan vi gjenskape det opprinnelige uttrekket fra spss med rådata fra task analytics
"""
# 1 gjenskap uttrekk fra spss ved å bruke samme respondenter
subset = df[df["id"].isin(df_spss["V1"])]
# %%
# 2 bekreft at vi har riktig antall unike svar i begge sett, slik at diff er 0
if df_spss["V1"].nunique() - subset["id"].nunique() == 0:
    print("Datasettene er like. Vi kan gjenskape resultatene.")
else:
    print("Datasettene er ulike. Vi kan ikke gjenskape resultatene med nytt uttrekk.")
# %%
# 3 sjekk at kolonner for analysen ble like i begge datasett - skal gi True i svar
if df.columns.equals(subset.columns) == True:
    print("Datasett inneholder samme variabler.")
else:
    print("Datasettene inneholder ikke samme variabler.")
# %%
# 4 sjekk at datatyper er de samme - skal svare med True
if df.dtypes.equals(subset.dtypes) == True:
    print("Datatypene er de samme i begge datasettene.")
else:
    print("Datatypene er ulike i datasettene.")

# %%
"""
Del 2: Analyse for hele tidsperioden
"""
# %%
"""
Del 3: Analyse før sommer 2025
"""
# Logistisk regresjon
# Årsaker til at innbyggere tar kontakt om brev
subset["dep"] = subset["Nav_kontakt"]
subset["dep"] = subset["dep"].map(
    {"Jeg tok ikke kontakt med NAV om brevet": 0, "Jeg kontaktet NAV om brevet": 1}
)
subset = subset.dropna(subset=["dep"])
subset["dep"] = subset["dep"].astype(int)
# %%
# Logistisk regresjon
# Kontaktet de Nav og morsmål
model = logit("dep ~ C(Morsmål)", data=subset)
res = model.fit()
print("Modell nummer 1")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())
# %%
# Log regresjon
# Kontaktet de Nav og morsmål
model = logit("dep ~ C(Alder)", data=subset)
res = model.fit()
print("Modell nummer 2")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())
# %%
# Log regresjon
# Kontaktet Nav og brevtype + alder + morsmål
model = logit("dep ~ C(Brevtype) + C(Alder) + C(Morsmål)", data=subset)
res = model.fit()
print("Modell nummer 3")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())
# %%
# Likelihood test for å se om alder forklarer mer enn nullhypotese - ingen variabel
# Lag begge modellene og sammenlign fit
model_with = logit("dep ~ C(Alder)", data=subset).fit()
model_without = logit("dep ~ 1", data=subset).fit()

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
model = logit("dep ~ C(Morsmål) + C(Alder)", data=subset)
res = model.fit()
print("Modell nummer 4")
print(f"Formel: {res.model.formula} \n \n")
print(res.summary())

# %%
# Regresjoner om
# De som synes det er lett eller veldig lett å forstå vedtak
subset["dep"] = subset["Innvilgelse_1"].copy()
subset["dep"] = subset["dep"].map(
    {
        "Lett å forstå": 1,
        "Veldig lett å forstå": 1,
        "Verken lett eller vanskelig": 0,
        "Vanskelig å forstå": 0,
        "Veldig vanskelig å forstå": 0,
        "Jeg fant ikke forklaringen": 0,
    }
)
subset = subset.dropna(subset=["dep"])
subset["dep"] = subset["dep"].astype(int)
# %%
# Log regresjon - forstår begrunnelse og morsmål alder og kontaktet Nav
model = logit("dep ~ C(Morsmål) + C(Alder) + C(Nav_kontakt)", data=subset)
res = model.fit()
print("Modell nummer 5")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())

# %%
# Før vi bytter avhengig variabel så må vi laste inn datasett på nytt
# For å bli kvitt manglende svar

subset = pd.read_spss(datasett_sav_sti)
# %%
# Kombinere variablene om å forstå vedtak
# kombinerer spørsmål om begrunnelse for alle brevtyper
subset["dep_forstå"] = (
    subset[["Innvilgelse_1", "Avslag_1", "Mangelbrev_1", "Stans_1"]]
    .bfill(axis=1)
    .iloc[:, 0]
)

# %%
# sett nivåer på ordinal variabel
subset["dep_forstå"] = subset["dep_forstå"].cat.set_categories(
    [
        "Veldig vanskelig å forstå",
        "Vanskelig å forstå",
        "Verken lett eller vanskelig",
        "Lett å forstå",
        "Veldig lett å forstå",
    ],
    ordered=True,
)
subset = subset.dropna(subset=["dep_forstå"])  # fjern tomme rader
# %%
# Ref https://github.com/statsmodels/statsmodels/issues/7418
# I en regresjon med flere ordinale variabler kombinerer vi disse med egen syntaks
# OrderedModel.from_formula('dependent ~ C(factor1) + C(factor2) + C(factor1):C(factor2)', distr='logit', data=subset)
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Alder) + C(Spraak) + C(Alder):C(Spraak)", distr="logit", data=subset
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 6")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Alder) + C(Morsmål) + C(Alder):C(Morsmål)", distr="logit", data=subset
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 7")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
model = OrderedModel.from_formula("dep_forstå ~ C(Tid_brukt)", distr="logit", data=subset)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 8")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Klagerettigheter)", distr="logit", data=subset
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 9")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
model = OrderedModel.from_formula(
    "dep_forstå ~ C(Antall_ganger_lest)", distr="logit", data=subset
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 10")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
# forståelse av begrunnelse i vedtak og brevtype
subset["brev"] = subset["Brevtype"].copy()
subset["brev"] = subset["brev"].str.replace("Ingen av disse", "Nan")

model = OrderedModel.from_formula("dep_forstå ~ C(brev)", distr="logit", data=subset)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 11")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())


# %%
# forståelse av begrunnelse og alder
model = OrderedModel.from_formula(
    "dep_forstå ~ C(brev) + C(Alder)", distr="logit", data=subset
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 12")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
# forståelse av begrunnelse og alder og morsmål
model = OrderedModel.from_formula(
    "dep_forstå ~ C(brev) + C(Alder) + C(Morsmål)", distr="logit", data=subset
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 13")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())
# %%
# forståelse av begrunnelse morsmål
model = OrderedModel.from_formula(
    "dep_forstå ~ C(brev) + C(Morsmål)", distr="logit", data=subset
)
res = model.fit(method="bfgs", disp=False)
print("Modell nummer 14")
print(f"Formelen: {res.model.formula} \n \n")
print(res.summary())

