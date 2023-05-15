import math
from collections import defaultdict
from numpy import *

from bleurt_setbase import calculatingScore, preprocess_text_bleurt

import bert_score
from bert_score import BERTScorer

from BARTScore.bart_score import BARTScorer

import subprocess

subprocess.run(['wget', 'https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip'])
subprocess.run(['unzip', 'BLEURT-20.zip'])

bart_scorer = BARTScorer(checkpoint='facebook/bart-large-cnn')

bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

def not_empty(text):
    return text and text.strip()

# for futher compare, we coombine the multi candidates togethers
def multi_ref_keypGenerated(data):
    data['multi_cands'] = ''
    for i in range(len(data)):
        for j in range(len(data)):
            if (data["stance"][i] == data["stance"][j])&(data["topic"][i] == data["topic"][j]):
                data["multi_cands"][i] += data["key_point"][j] +"="
    
    return data

def multi_ref_Goldkeyp(data):
    data['multi_cands'] = ''
    for i in range(len(data)):
        for j in range(len(data)):
            if (data["stance"][i] == data["stance"][j])&(data["topic"][i] == data["topic"][j]):
                data["multi_cands"][i] += data["key_point_given"][j] +"="
    
    return data

#Since BART Score need the number of the references same for each candidate, thus will deal with separete.
def list_duplicates(seq):
    dict_df = defaultdict(list)
    for i, item in enumerate(seq):
        dict_df.append(i)
    return ((key, locs) for key, locs in dict_df.items() if len(locs)>1)

def balance_ref_num(refs) -> list:
    max_len = len(max(refs, key= len))
    for element in refs:
        if len(element) == max_len:
            pass
        else:
            while len(element) != max_len:
                element += ["null"]
    return refs

def preprocess_text(data, metrics):
    if metrics == "softRecall":
        cands = data['key_point_given'].tolist()
        ref = data['multi_cands'].tolist()

        multi_refs = [element.strip().split('=') for element in ref if element.strip()!='']
        refs = [list(filter(not_empty, element)) for element in multi_refs]
  
    elif metrics == "softPrecision":
        cands = data['key_point'].tolist()
        ref = data['multi_cands'].tolist()

        multi_refs = [element.strip().split('=') for element in ref if element.strip()!='']
        refs = [list(filter(not_empty, element)) for element in multi_refs]

    return cands, refs

def softPrecision(data, metrics) -> float:
    if metrics == "BERTScore":
        softp_data = multi_ref_Goldkeyp(data)
        cands, refs= preprocess_text(softp_data, metrics = "softPrecision")

        P, R, F = bert_scorer.score(cands, refs)
        P_average = P.mean()

    elif metrics == "BARTScore":

        softp_data = multi_ref_Goldkeyp(data)
        cands, refs= preprocess_text(softp_data, metrics = "softPrecision")

        #BART score cannot be processed for different numbers of reference sentences, so
        #check for the maximum number of reference sentences and match the size.
        #We fill in the None for the missing sentences because we only pick one of the 
        #maximum values and not the average, so there is no impact on performance

        refs = balance_ref_num(refs)        

        # generation scores from the first list of texts to the second list of texts.
        P = bart_scorer.multi_ref_score(cands, refs, agg="max", batch_size=4) # agg means aggregation, can be mean or max

        #mapping the score to (0,1]
        P_average = math.tanh(math.exp((mean(P))/2+1.3))
        #P_average = math.tanh(mean(P)) + 1
        #P_average = math.exp(mean(P))

    elif metrics == "BLEURT":
        #build the compaire list
        references, candidates, df_compare_precision = preprocess_text_bleurt(data, metrics = "softPrecision")

        #For each generated keypoint, calculating the score between the generated keypoint-gold keypoint pair according to BLEURT.
        result = calculatingScore(references, candidates)
        df_compare_precision["BLEURT Score"] = result
        
        #After calculating the semantic quality of all candidates and reference pairs, the one with the highest score is selected as the correct pair. 
        df_bestkp_pair_precision = df_compare_precision.loc[df_compare_precision.groupby(["candidate"])["BLEURT Score"].idxmax()]
        #take average of all best scores as the soft precision score.
        P_average = df_bestkp_pair_precision["BLEURT Score"].mean()
    
    return P_average

def softRecall(data, metrics) -> float:
    if metrics == "BERTScore":
        softr_data = multi_ref_keypGenerated(data)
        cands, refs= preprocess_text(softr_data, metrics = "softRecall")

        P, R, F = bert_scorer.score(cands, refs)
        R_average = R.mean()

    elif metrics == "BARTScore":

        softr_data = multi_ref_keypGenerated(data)
        cands, refs= preprocess_text(softr_data, metrics = "softRecall")

        refs = balance_ref_num(refs)

        R = bart_scorer.multi_ref_score(cands, refs, agg="max", batch_size=4)

        #mapping the score to (0,1]
        R_average = math.tanh(math.exp((mean(R)/2)+1.3)) 
        #R_average = (0.25 * mean(R)) + 1
        #R_average = math.exp(mean(R))
        
    elif metrics == "BLEURT":
        #build the compaire list
        references, candidates, df_compare_recall = preprocess_text_bleurt(data, metrics = "softRecall")

        #For each golden keypoint, calculating the score between the generated keypoint-gold keypoint pair according to BLEURT.
        result = calculatingScore(references, candidates)
        df_compare_recall["BLEURT Score"] = result
        
        #After calculating the semantic quality of all candidates and reference pairs, the one with the highest score is selected as the correct pair. 
        df_bestkp_pair_recall = df_compare_recall.loc[df_compare_recall.groupby(["candidate"])["BLEURT Score"].idxmax()]
        
        #take average of all best scores as the soft precision score.
        R_average = df_bestkp_pair_recall["BLEURT Score"].mean()

    return R_average

def softF1(precision, recall) -> float:
    f1_soft = (2 * float(precision * recall)) / float(precision + recall)
    return f1_soft

def softevaluation(data, softmetrics, metrics) -> float:
    M_average = []
    if softmetrics == "precision":
        result = softPrecision(data, metrics)


    elif softmetrics == "recall":
        result = softRecall(data, metrics)
    
    else:
        precision = softPrecision(data, metrics)
        recall = softRecall(data, metrics)
        result = softF1(precision, recall)

    return result