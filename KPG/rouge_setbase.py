from rouge_score import rouge_scorer
import pandas as pd
pd.set_option("display.max_colwidth", None)
from itertools import product
from statistics import mean

def load_dataset(gt_file) -> pd.DataFrame:
    
    gt_gold_kp = pd.read_json(gt_file)

    return gt_gold_kp

def preprocess_dataset(gt_gold_kp):

    topics = ['Routine child vaccinations should be mandatory', 'Social media platforms should be regulated by the government', 'The USA is a good country to live in']
    stances = [-1, 1]

    predictions, references = [], []
    for topic in topics:
        for stance in stances:
            kps = gt_gold_kp.loc[(gt_gold_kp['topic']==topic) & (gt_gold_kp['stance']==stance), 'key_point'].tolist()
            gold_kps = gt_gold_kp.loc[(gt_gold_kp['topic']==topic) & (gt_gold_kp['stance']==stance), 'key_point_given'].tolist()
            predictions.append(kps)
            references.append(gold_kps)

    return predictions, references

def compute_rouge(predictions, references):

    rouge1_scores, rouge2_scores, rougel_scores = [], [], []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for preds, refs in zip(predictions, references):
        # compute per topic avg scores on all unique combinations of generated keypoints and ground-truth
        r_1 = []
        r_2 = []
        r_l = []
        for a,b in product(preds, refs):
            scores  = scorer.score(a,b)
            r_1.append(round(scores['rouge1'].fmeasure,3))
            r_2.append(round(scores['rouge2'].fmeasure,3))
            r_l.append(round(scores['rougeL'].fmeasure,3))
        
        # save per topic scores to compute average over all topics as the final score
        rouge1_scores.append(round(mean(r_1),3))
        rouge2_scores.append(round(mean(r_2),3))
        rougel_scores.append(round(mean(r_l),3))
    print("Rouge 1: {}".format(round(mean(rouge1_scores),3)))
    print("Rouge 2: {}".format(round(mean(rouge2_scores),3)))
    print("Rouge L: {}".format(round(mean(rougel_scores),3)))
