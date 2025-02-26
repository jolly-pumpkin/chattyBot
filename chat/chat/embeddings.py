import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
import pandas as pd
from dateutil.parser import parser


load_dotenv()


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_URL = os.getenv('OPENAI_URL')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')


print("++++++Params++++")
print(OPENAI_API_KEY)
print(OPENAI_URL)
print(EMBEDDING_MODEL_NAME)

openai = OpenAI(
    base_url =OPENAI_URL,
    api_key = OPENAI_API_KEY,
)

params = {
    "action": "query", 
    "prop": "extracts",
    "exlimit": 1,
    "titles": "2023_Turkey–Syria_earthquakes",
    "explaintext": 1,
    "formatversion": 2,
    "format": "json"
}
resp = requests.get("https://en.wikipedia.org/w/api.php", params=params)
response_dict = resp.json()
print("____________________________")

print(response_dict)



text_data = response_dict["query"]["pages"][0]["extract"].split("\n")

print("____________________________")

print(text_data)

# Load page text into a dataframe
df = pd.DataFrame()
df["text"] = text_data

# Clean up dataframe to remove empty lines and headings
df = df[(
    (df["text"].str.len() > 0) & (~df["text"].str.startswith("=="))
)].reset_index(drop=True)
df.head()

print("____________________________")

print(df["text"][0])

print("____________________________")


batch_size = 100
embeddings = []
 
#Create embeddings
for i in range(0, len(df), batch_size):
    # Send data to OpenAI to get embeddings
    response = openai.embeddings.create(
            input=df.iloc[i:i+batch_size]["text"].tolist(),
            model=EMBEDDING_MODEL_NAME
    )
    print(response)
    embeddings.extend([data.embedding for data in response.data])

df["embeddings"] = embeddings


df.to_csv('embeddings.csv', index=False)