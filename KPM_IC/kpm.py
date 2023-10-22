import re, math, os, argparse
import pandas as pd
import numpy as np
import nltj
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from numpy import *

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_dataset(input_file, stance, topic) -> pd.DataFrame:
    '''
    Each time we only deal with single topic and stance. Since we proposed a unsupervised cluster aigo
    '''

    input_corpus = pd.read_csv(input_file)
    input_corpus = input_corpus(input_corpus["topic"] == topic)

    if stance == 1:
        input_corpus = input_corpus(input_corpus["stance"] == 1)
    else:
        input_corpus = input_corpus(input_corpus["stance"] == -1)

    input_corpus.reset_index(drop=True, inplace=True)

    return input_corpus

    
def bertopic(docs, n_neighbors=3, n_components=2, min_dist=0.0, min_cluster_size=6):
    '''
    unsup methods for cluster.
    here, we first cluster the argument if they share similarity of meaning
    we don't force all the argument going to a cluster as we have IC mechanism
    '''
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, metric='cosine')
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    sentence_model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = sentence_model.encode(docs)

    topic_model = BERTopic(embedding_model=sentence_model, umap_model=umap_model, hdbscan_model=hdbscan_model,
                           vectorizer_model=vectorizer_model, calculate_probabilities=True, low_memory=False,
                           n_gram_range=(1, 1), nr_topics="auto")
    
    topics, probs = topic_model.fit_transform(docs, embeddings)

    return topics, probs


def output(docs, cluster_class, probability) -> pd.DataFrame:
    clu = cluster_class
    distribute = probability.tolist()
    output_distribute = pd.DataFrame()
    output_distribute["argument"] = docs["argument"]
    stance_class = []
    stance_prob = []
    for i in range(0, len(docs)):
        stance_class.append(clu[i])
        stance_prob.append(distribute[i])
    output_distribute["topic_class"] = stance_class
    output_distribute["topic_distribute"] = stance_prob

    return output_distribute

def output_kmeans(docs, cluster_class) -> pd.DataFrame:
    clu = cluster_class
    output_distribute = pd.DataFrame()
    output_distribute["argument"] = docs["argument"]
    stance_class = []
    for i in range(0, len(docs)):
        stance_class.append(clu[i])
    output_distribute["topic_class"] = stance_class

    return output_distribute

#------------------------------------
#-------Multi to Multi Mapping-------
#------------------------------------

def threshold_cluster(output_distribute):
    '''
    Compute the second higher score for all of the argument
    '''
    threshold_list = []
    threshold_li = output_distribute["topic_distribute"].tolist()
    for score in threshold_li:
        score_li = list(score)
        score_li.pop(0)
        score_li.pop(-1)
        score_li = "".join(score_li)
        score = score_li.split(",")
        score = [x.strip() for x in score if x.strip() !='']
        score_list = sorted(score, reverse=True)
        threshold_list.append(score_list[1])
        threshold_list = list(map(float, threshold_list))
    threshold = mean(threshold_list)
    return threshold

def threshold_final(threshold_t):
    '''
    If threshold is too small then it will loss the meaning to building a new cluster
    '''
    if threshold_t > 0.10:
        threshold = threshold_t
    else :
        threshold = 0.10

    return threshold

def combine_argument_lists(best, single_arg, threshold, stance, topic) -> pd.DataFrame:
    arg_lists = [[] for _ in range(10)]

    for row in range(0, len(best)):
        num = best[row]
        test = num.split()
        for j in test:
            if float(j) > threshold:
                for idx, item in enumerate(test):
                    if item == j:
                        arg_lists[idx].append(single_arg[row])
                        break
                else:
                    arg_lists[9].append(single_arg[row])

    # Remove duplicates
    for i in range(10):
        arg_lists[i] = list(set(arg_lists[i]))

    # Join the argument lists and split them by '/'

    arg_lists = ['\n'.join(arg_list) for arg_list in arg_lists]

    # Create a single DataFrame with the combined argument lists
    arg_list = '\n'.join(arg_lists)

    arg = pd.DataFrame()
    arg["arg_list"] = arg_list
    arg["topic"] = topic
    arg["stance"] = stance

    return arg

#------------------------------------
#--------Iterative Clustering--------
#------------------------------------

def cosine_sim(x,y):
  a = sum([i*j for i,j in zip(x,y)])
  b = math.sqrt(sum([i*j for i,j in zip(x,x)])) * math.sqrt(sum([i*j for i,j in zip(y,y)]))
  return a/b

def hierarchical_cluster(distribute, threshold):

  # Initialising clusters
  clusters = distribute[(distribute['topic_class']!= -1)]
  for index,row in distribute.iterrows():
    current_cluster = clusters['topic_class'].unique()

    # Unclassified data
    if row["topic_class"]==-1:
      x = row["topic_distribute"]
      max_sim=-1
      best_cluster = -1

      for topic_class,group in clusters.groupby("topic_class"):
        # Take the centre vector of each cluster as the anchor point 
        # and calculate the similarity
        sim = 0

        ave_vec = np.zeros(len(clusters["topic_distribute"].values[0]))

        for vec in group["topic_distribute"].values:
          ave_vec+=np.array(vec)
        ave_vec/=len(group)
        sim = cosine_sim(x,ave_vec)
        
        # Or take the centre vector of each cluster as the anchor point and calculate the similarity (both are the same)
        #for vec in group["topic_distribute"].values:
        #  y = vec
        #  sim += cosine_sim(x,y)
        #sim /= len(group)

        if sim>max_sim and sim>threshold:
          max_sim = sim
          best_cluster = topic_class

      if best_cluster==-1:
        best_cluster = len(current_cluster)+1
      row["topic_class"] = best_cluster
      clusters.loc[index] = row

  return clusters



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_file", type=str, required=True, help="Path to a csv file into model")
    parser.add_argument("--stance", type=int, required=True, help="Stance of the debate topic")
    parser.add_argument("--topic", type=str, required=True, help="the topic of debate")
    parser.add_argument("--n_neighbors", type=str, required=True, help="Used to balances local versus global structure in the data")
    parser.add_argument("--n_components", type=int, required=True, help="Determine the dimensionality of the reduced dimension space we will be embedding the data into")
    parser.add_argument("--min_dist", type=str, required=True, help="Provides the minimum distance apart that points are allowed to be in the low dimensional representation")
    parser.add_argument("--min_cluster_size", type=str, required=True, help="minmium size of cluster")
    parser.add_argument("--similarity_threshold ", type=float, default=0.95, help="threshold for IC")

    parser.add_argument("--setting", type=str, default="bertopic", help="whether using kmeans")

    parser.add_argument("--output_file", type=str, default="kpm_output.csv", help="Generate sample")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to an output directory were the finetuned model and results are saved",
    )

    args = parser.parse_args()
    similarity_threshold = args.similarity_threshold 

    docs = load_dataset(
        input_file = args.input_file,
        stance = args.stance,
        topic = args.topic
    )

    topics, probs = bertopic(
        docs = docs,
        n_neighbors = args.n_neighbors,
        n_components = args.n_components,
        min_dist = args.min_dist,
        min_cluster_size = args.min_cluster_size
    )

    if args.setting == "bertopic":
        output_df = output(docs, topics, probs) 
    
    elif args.setting == "kmeans":
        output_df = output_kmeans(docs, topics, probs) 
    
    else:
        print("Please provide one cluster setting")

    #multi-to-multi mapping 
    threshold_t = threshold_cluster(output_df)
    threshold = threshold_final(threshold_t)
    single_arg = output_df["argument"].tolist()
    distribution_arg = output_df["topic_distribute"].tolist()

    result = []
    for row in distribution_arg:
        flag = []
        for word in row:
            if word != "[" and word != "]" and word != ",":
                flag.append(word)
        result.append(flag)
    best = ["".join(row) for row in result]

    final_df = combine_argument_lists(best, single_arg, threshold)

    #Iterative Clusterin
    if args.ic:    
        test_clusters = hierarchical_cluster(output_df, similarity_threshold)
        final_df = combine_argument_lists(test_clusters, single_arg, threshold)

    
    final_df.to_csv(os.path.join(args.output_dir, f"test_{args.setting}_{args.ic}.csv"), index=False)