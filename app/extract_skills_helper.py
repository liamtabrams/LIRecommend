import pandas as pd
from crewai import Agent, Task, Crew
import os
import joblib
import pdb

os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1'
os.environ["OPENAI_MODEL_NAME"] = 'llama3-70b-8192'
os.environ["OPENAI_API_KEY"] = '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'

extractor = Agent(
    role = "qualifications extractor",
    goal = "extract required qualifications from the job posting and return as a list with a predictable format",
    backstory = "You are an AI assistant whose job is to extract all required qualifications from job postings and return them as a list, with each item separated by a new line character. Each item should be a phrase of no more than 5 words. Please summarize the required qualifications as fully as you can with minimal redundancy.",
    verbose = True,
    allow_delegation = False
)

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from statistics import mode, StatisticsError
import numpy as np

#tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
#model = AutoModel.from_pretrained("thenlper/gte-large")

gte_model = joblib.load('../user_data/models/gte_model.joblib')
gte_tokenizer = joblib.load('../user_data/models/gte_tokenizer.joblib')

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def top_quals_analysis():
    if "req_quals.csv" not in os.listdir("../user_data/dataset/"):
        df = pd.read_csv("../user_data/dataset/myDataset.csv")
        df['req_quals'] = None
    else:
        df = pd.read_csv("../user_data/dataset/req_quals.csv")
    
    for row in df.itertuples(index=True, name='Pandas'):
        if pd.isna(row.req_quals):
            print("row was n/a")
            posting_text = row.posting_text
            
            extract_quals = Task(
                description = f"Extract required skills from the following job posting:\n\n'{posting_text}'",
                agent = extractor,
                expected_output = "Here is an example output: '3+ years experience embedded C\nExperience writing unit tests\n1+ year experience technical lead'",
            )

            crew = Crew(
                agents = [extractor],
                tasks = [extract_quals],
                verbose = 2
            )

            while True:
                try:
                    result = crew.kickoff()
                    break
                except Exception as e:
                    print(e)
                    print(e.args)
            
            req_quals = result.split("\n")
            print(f"Debug: req_quals for row {row.Index}: {req_quals}")
            
            df.at[row.Index, 'req_quals'] = req_quals
            df.to_csv('../user_data/dataset/req_quals.csv', index=False)

    import ast
    df['req_quals'] = df['req_quals'].apply(ast.literal_eval)
    row_labels = []
    for row in df.itertuples(index=True, name='Pandas'):
        req_quals = row.req_quals
        for phrase in req_quals:
            row_labels.append(row.Index)
    
    print("generated row labels")

    # Flatten the lists of phrases into a single list
    all_phrases = [phrase for sublist in df['req_quals'] for phrase in sublist]
    '''
    print("about to go through gte tokenizer")
    batch_dict = gte_tokenizer(all_phrases, max_length=512, padding=True, truncation=True, return_tensors='pt')
    print("done with gte tokenizer")
    outputs = gte_model(**batch_dict)
    print("generated outputs")
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    print("got through average pool")'''

    batch_size = 32 # Adjust this to a value that works within your memory constraints
    embeddings_list = []
    len_all = len(all_phrases)
    print(f"all_phrases size is {len_all}")
    for i in range(0, len_all, batch_size):
        print(f"iter is {i}")
        batch_phrases = all_phrases[i:i + batch_size]
        print(batch_phrases)
        batch_dict = gte_tokenizer(batch_phrases, max_length=512, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():  # Disable gradient tracking
            outputs = gte_model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings_list.append(embeddings)

    embeddings = torch.cat(embeddings_list, dim=0)

    # (Optionally) normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    print("got through normalization")

    # we need to think about how many clusters we want in our phrase embedding space
    # 1 < N < 1000
    # N=500?
    # we can always change N

    kmeans = KMeans(n_clusters=500, random_state=57)
    print("fitting embeddings to K Means")
    kmeans.fit(embeddings)
    print("done fitting embeddings to K Means")
    #centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    # Step 1: Group embeddings by their cluster labels
    cluster_members_dict = defaultdict(list)
    embeddings_list = embeddings.tolist()
    for label, embedding in zip(cluster_labels, embeddings_list):
        cluster_members_dict[label].append(embedding)  # Keep embeddings as lists or arrays
        
    data = {'phrase': all_phrases, 'row_label': row_labels, 'cluster_label': cluster_labels}
    #pdb.set_trace()
    clusters_df = pd.DataFrame(data)

    # Create the qual_clusters column if it doesn't exist
    if 'qual_clusters' not in df.columns:
        df['qual_clusters'] = [[] for _ in range(len(df))]
    
    print("created clusters dataframe")
    cluster_score_dict = {}
    for row in df.itertuples(index=True, name='Pandas'):
        clusters_in_row = clusters_df.loc[clusters_df['row_label'] == row.Index, 'cluster_label'].tolist()
        # Generate value counts
        df.at[row.Index, 'qual_clusters'] = clusters_in_row
        for cluster in set(clusters_in_row): # we just want to look at unique cluster numbers per row
            if cluster not in cluster_score_dict.keys():
                cluster_score_dict[cluster] = row.rating 
            else:
                cluster_score_dict[cluster] += row.rating
    clusters_ranked = sorted(cluster_score_dict.items(), key=lambda item: item[1], reverse=True)
    top_10_cluster_scores = clusters_ranked[:10]
    # Extract the keys from the top n items
    top_10_clusters = [item[0] for item in top_10_cluster_scores]

    # Step 2: Calculate the mode embedding for each cluster
    top_10_cluster_modes = {}
    top_10_cluster_phrases = {}
    # Subset the cluster_member_dict
    top_10_dict = {key: cluster_members_dict[key] for key in top_10_clusters if key in cluster_members_dict}
    for label, embeddings in top_10_dict.items():
        try:
            # Use scipy.stats.mode to find the mode along axis 0
            mode_embedding = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=embeddings)
            top_10_cluster_modes[label] = mode_embedding
            # Find indices of embeddings equal to embedding_to_find
            indices = [index for index, emb in enumerate(embeddings_list) if emb == mode_embedding]
            top_10_cluster_phrases[label] = all_phrases[indices[0]]
        except Exception as e:
            print(e)
            # Handle case where there's no unique mode (e.g., all unique embeddings)
            cluster_emb_rep = embeddings[0] #representative embedding
            top_10_cluster_modes[label] = cluster_emb_rep
            indices = [index for index, emb in enumerate(embeddings_list) if emb == cluster_emb_rep]
            top_10_cluster_phrases[label] = all_phrases[indices[0]]

    top_10_phrase_scores = {}

    for item in top_10_cluster_scores:
        # Get the corresponding phrase from top_10_cluster_phrases using the cluster_label
        phrase = top_10_cluster_phrases[item[0]]
    
        # Get the score corresponding to the cluster_label from top_10_cluster_scores
        score = item[1]
    
        # Map the phrase to the score in the phrase_to_score dictionary
        top_10_phrase_scores[phrase] = score

    print("done getting top 10 phrase scores")  

    return top_10_phrase_scores


print("get insights endpoint triggered")
top_10_phrase_scores = top_quals_analysis()
print(top_10_phrase_scores)
#return {"message": "success"}
