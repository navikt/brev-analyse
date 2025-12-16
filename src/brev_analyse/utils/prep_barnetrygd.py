# %%
import pandas as pd

from get_answers import get_survey_questions
# %%
datasett_sti = "../../../data/uttrekk brevmålinger 20251204/brev_2025_barnetrygd.csv"
df = pd.read_csv(datasett_sti)
# %%
"""
Forbered datasett til analyser om barnetrygd

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
    "answers.segment": "Betale_tilbake",
    "answers.instead": "Forelder_EØS",
    "answers.instead1": "Utvidet_barnetrygd",
    "answers.instead2": "Delt_bosted",
    "answers.segment4": "Overskrift",
    "answers.segment7": "Innvilgelse_hvorfor",
    "answers.segment8": "Innvilgelse_informasjon",
    "answers.segment16": "Klagerettigheter",
    "answers.segment3": "Finne_informasjon",
    "answers.segment18": "Språket_brevet",
    "answers.segment5": "Antall_ganger",
    "answers.segment6": "Tidsbruk",
    "answers.segment17": "Kontaktet_Nav",
    "answers.segment19": "Morsmål",
    "answers.segment20": "Alder"
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
df.to_pickle("../../../data/barnetrygd_202512.pkl")
# %%
