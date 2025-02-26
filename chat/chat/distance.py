import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
from dateutil.parser import parser
from scipy.spatial.distance import cosine as cosine_distance


load_dotenv()


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_URL = os.getenv('OPENAI_URL')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')

df = pd.read_csv('embeddings.csv')

print(df)
print(df['embeddings'])
df.embeddings = df.embeddings.apply(eval).apply(np.array)
print("____________________________")

def get_embedding(text):
    openai = OpenAI(
        base_url =OPENAI_URL,
        api_key = OPENAI_API_KEY,
    )
    text = text.replace("\n", " ")
    return openai.embeddings.create(input = [text], model=EMBEDDING_MODEL_NAME).data[0].embedding


question = "When did Russia invade Ukraine"

question_embeddings = get_embedding(question)
print("____________________________")
print(question_embeddings)


df['distances'] = df['embeddings'].apply(lambda row: cosine_distance(row, question_embeddings))

print(df['distances'])


df.to_csv('distance.csv', index=False)