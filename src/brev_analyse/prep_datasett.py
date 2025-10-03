# %%
import pandas as pd

from get_answers import get_survey_questions

# %%
datasett_sav_sti = "../../data/Dagpenger brevmåling per 28042025.sav"
df_spss = pd.read_spss(datasett_sav_sti)
# %%
datasett_ta_sti = "../../data/uttrekk brev dagpenger 20250916.xlsx"
df = pd.read_excel(datasett_ta_sti)
# %%
q = get_survey_questions(dataframe=df_spss)
questions = get_survey_questions(dataframe=df)
# %%
"""
Forbered datasett til analyser om dagpenger

Lag to datasett. Ett for hele tidsperioden, og ett for perioden før sommerferie

Datasettene skal ha samme datatyper og variabler til slutt slik at analysen kan gjentas for hver tidsperiode

Steg
* fjern rader uten svar - KLAR
* fjern rader fra de som ikke leste brevet - TODO
* erklære de kategoriske variablene - KLAR
* bruk kortere variabelnavn i stedet for stavelsen av alle spørsmål - KLAR
* lag subset-datasett basert på identiske respondenter fra rådata - KLAR

NOTE
Det er noen ulikheter i det opprinnelige datasettet fra spss angående morsmål
Resten av datasettet bør kontrolleres for å se om dette påvirker analysen
"""

# %%
"""
Erklære hvilke variabler er kategoriske
"""
cat_col_pattern = [col for col in df.columns if col.startswith("answers.t")]
df[cat_col_pattern] = df[cat_col_pattern].astype("category")
# %%
# fjern rader uten svar
df = df[df["answers.c"] != "Unknown"]
# %%
# behold første rad, dropp nummer 2
df = df.iloc[1:]
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
Lag construct variabler til analysen som går på tvers av brevtype
"""
# Kombinere variablene om å forstå vedtak
# kombinerer spørsmål om begrunnelse for alle brevtyper
df["dep_forstå"] = (
    df[["Innvilgelse_hvorfor", "Avslag_hvorfor", "Mangel_hvorfor", "Stans_hvorfor"]]
    .bfill(axis=1)
    .iloc[:, 0]
).astype("category")

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
"""
Slik kan vi gjenskape det opprinnelige uttrekket fra spss med rådata fra task analytics
"""
# gjenskap uttrekk fra spss ved å bruke samme respondenter
subset = df[df["id"].isin(df_spss["V1"])]
# %%
# sjekk om dataframes er identiske
pd.testing.assert_frame_equal(subset, df_spss)
# %%
# alternativ sjekk
subset.equals(df_spss)
# %%
# sjekk at kolonner for analysen ble like i begge datasett - skal gi True i svar
df.columns.equals(subset.columns)
# %%
# sjekk at datatyper er de samme - skal svare med True
df.dtypes.equals(subset.dtypes)
# %%
"""
Nå er både fullstendig og opprinnelig datasett klare til analyser

Henvis til de heretter som
* df
* subset
"""
