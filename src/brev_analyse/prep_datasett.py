# %%
import pandas as pd
from statsmodels.formula.api import logit

from get_answers import get_survey_questions

# %%
datasett_sav_sti = "../../data/Dagpenger brevmåling per 28042025.sav"
df_spss = pd.read_spss(datasett_sav_sti)
# %%
datasett_ta_sti = "../../data/uttrekk brev dagpenger 20250916.xlsx"
df = pd.read_excel(datasett_ta_sti)
# %%
"""
Forbered datasett til analyser om dagpenger

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

NOTE
Det er noen ulikheter i det opprinnelige datasettet fra spss angående morsmål
Resten av datasettet bør kontrolleres for å se om dette påvirker analysen
"""
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
Nå er datasettene for begge tidsperiodene klare til analyse.

De lagres i mappen data/
"""
# %%
df.to_pickle("../../data/dagpenger_202510.pkl")
# %%
subset.to_pickle("../../data/dagpenger_202506.pkl")
# %%
