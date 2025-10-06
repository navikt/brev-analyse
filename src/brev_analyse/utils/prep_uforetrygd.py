# %%
import pandas as pd

from get_answers import get_survey_questions

# %%
datasett_ta_sti = "../../../data/uttrekk brev uføretrygd 20250916.xlsx"
df = pd.read_excel(datasett_ta_sti)
# %%
"""
Forbered datasett til analyser om uføretrygd

Steg
* lagrer fullstendige spørsmålsformuleringer som dictionary
* fjern første rad med fullstendige spørsmålsformuleringer
* erklære de kategoriske variablene i datasettene
* bruk kortere variabelnavn i stedet for stavelsen av alle spørsmål

"""
# %%
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
    "answers.segment4": "Overskrift",
    "answers.segment7": "Innvilgelse_hvorfor",
    "answers.segment8": "Innvilgelse_informasjon",
    "answers.segment10": "Avslag_hvorfor",
    "answers.segment11": "Avslag_informasjon",
    "answers.segment12": "Endret_hvorfor",
    "answers.segment22": "Endret_informasjon",
    "answers.segment16": "Klagerettigheter",
    "answers.segment3": "Finne_informasjon",
    "answers.segment18": "Språket_brevet",
    "answers.segment5": "Antall_ganger",
    "answers.segment6": "Tidsbruk",
    "answers.segment20": "Hvem_representerer",
    "answers.segment17": "Kontaktet_Nav",
    "answers.instead1": "Kontaktet_tema",
    "answers.instead": "Hjelp_lesebrev",
    "answers.instead2": "Jobb_og_ufør",
    "answers.instead3": "Barnetillegg",
    "answers.instead4": "Rep_jobb_og_ufør",
    "answers.instead5": "Refp_barnetillegg",
    "answers.segment19": "Morsmål",
    "answers.instead6": "Alder",
}
# %%
# bytt ut alle kolonner som starter med answers. til de kortere formuleringene
df.rename(columns=questions_short, inplace=True)
# %%
# bytt om forenklet skala til fullstendig skala
likert_map = {
    "Veldig vanskelig": "Veldig vanskelig å forstå",
    "Vanskelig": "Vanskelig å forstå",
    "Verken lett eller vanskelig": "Verken lett eller vanskelig",
    "Lett": "Lett å forstå",
    "Veldig lett": "Veldig lett å forstå",
    "Jeg fant ikke forklaringen": "Jeg fant ikke forklaringen",
}
df["Finne_informasjon"] = df["Finne_informasjon"].replace(likert_map)
# %%
"""
Nå er datasettene for begge tidsperiodene klare til analyse.

De lagres i mappen data/
"""
# %%
df.to_pickle("../../../data/uføretrygd_202510.pkl")
# %%
