# %%
"""
Hent data for ulike spørreundersøkelser via task analytics
"""

import os

import taskanalytics_data_wrapper.taskanalytics_api as task
from dotenv import load_dotenv

# %%
load_dotenv()

epost = os.getenv("epost")
passord = os.getenv("passord")
dagpenger = os.getenv("dagpenger")
# %%
get_survey = task.download_survey(
    username=epost,
    password=passord,
    survey_id=dagpenger,
    filename_path="../../data/dagpenger.csv",
)
get_survey.status_code

# %%
