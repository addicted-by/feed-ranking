import os
TRAIN_DATA_PATH = 'data/mind_large_train/news.tsv'
TEST_DATA_PATH = 'data/mind_large_dev/news.tsv'
assert os.path.exists(TRAIN_DATA_PATH)
assert os.path.exists(TEST_DATA_PATH)



import pandas as pd

train_df = pd.read_csv(TRAIN_DATA_PATH, sep='\t', names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
test_df =  pd.read_csv(TEST_DATA_PATH, sep='\t', names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
train_df_clf = train_df[['news_id', 'category', 'title', 'abstract']]

train_df_clf['abstract'].fillna('', inplace=True)
train_df_clf['text'] = train_df_clf["title"] + train_df_clf['abstract']
assert train_df_clf.text.isna().sum() == 0

test_df_clf = test_df[['news_id', 'category', 'title', 'abstract']]
test_df_clf['abstract'].fillna('', inplace=True)
test_df_clf['title'].fillna('', inplace=True)
test_df_clf['text'] = test_df_clf["title"] + test_df_clf['abstract']


test_df_clf = test_df_clf.drop(columns=['title', 'abstract']).set_index('news_id')
train_df_clf = train_df_clf.drop(columns=['title', 'abstract']).set_index('news_id')

train_df_clf.drop_duplicates(inplace=True)
test_df_clf.drop_duplicates(inplace=True)


df_clf = pd.concat([train_df_clf,test_df_clf]).drop_duplicates().reset_index(drop=True)


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

model_emb = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("it5/it5-base-news-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("it5/it5-base-news-summarization")

import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


from transformers import pipeline

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)


from tqdm import tqdm
texts = df_clf['text'].values
embs = {}
for idx, text in enumerate(tqdm(texts)):
    try:
        summarized = summarizer(text, max_length=int(len(text.split()) * 0.75), min_length=int(len(text.split()) * 0.35), num_beams=10, repetition_penalty=600., length_penalty=0.6, early_stopping=True)[0]['summary_text']
        embs[idx] = model_emb.encode(summarized).tolist()
    except:
        print(f"Cannot process {idx} text. \nText: {text}")

import json
with open('embs.json', 'w', encoding='utf-8') as f:
    json.dump(embs, f, ensure_ascii=False, indent=4)