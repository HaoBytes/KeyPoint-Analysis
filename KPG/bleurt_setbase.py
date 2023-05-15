"""
This script is used for setbase evaluation of bleurt.
"""

import logging
import transformers
import re
import math
from collections import defaultdict

import bleurt
from bleurt import score as bleurt_score

checkpoint = "BLEURT-20"

def calculatingScore (references, candidates) -> float:
    #Calculating the candidates-references pair score
    scorer = bleurt_score.BleurtScorer(checkpoint)
    scores = scorer.score(references=references, candidates=candidates)
    return scores

def preprocess_text_bleurt(data, metrics) -> list:
    #for bleurt
    df_compare = pd.DataFrame(columns = ["candidate","reference"])
    df_bestkp_pair = pd.DataFrame()
    cand_list = []
    refs_list = []

    for i in range(len(data)):
        for j in range(len(data)):
            if (data["stance"][i] == data["stance"][j])&(data["topic"][i] == data["topic"][j]):

                if metrics == "softPrecision":
                    cand = data["key_point"][i]
                    cand_list.append(cand)
                    refs = data["key_point_given"][j]
                    refs_list.append(refs)

                elif metrics == "softRecall":
                    cand = data["key_point_given"][i]
                    cand_list.append(cand)
                    refs = data["key_point"][j]
                    refs_list.append(refs)

    df_compare["candidate"] = cand_list
    df_compare["reference"] = refs_list

    references = df_compare["reference"].tolist()
    candidates = df_compare["candidate"].tolist()

    return references, candidates, df_compare

