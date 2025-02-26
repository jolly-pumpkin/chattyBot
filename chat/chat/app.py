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

def chat(prompt: str):
    openai = OpenAI(
        base_url =OPENAI_URL,
        api_key = OPENAI_API_KEY,
    )

    answer = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt
    )
    return answer



ukraine_prompt = """
Question: When did Russia invade Ukraine?
Answer: 
"""


#print(ukraine_answer)
#print(ukraine_answer["choices"][0]["text"])



params = {
    "action": "query", 
    "prop": "extracts",
    "exlimit": 1,
    "titles": "2022",
    "explaintext": 1,
    "formatversion": 2,
    "format": "json"
}

resp = requests.get("https://en.wikipedia.org/w/api.php", params=params)
response_dict = resp.json()
response_dict["query"]["pages"][0]["extract"].split("\n")

print(response_dict)


df = pd.DataFrame()
df["text"] = response_dict["query"]["pages"][0]["extract"].split("\n")


# only keep records of df[text] if the length of the string is greater then 0
df = df[df["text"].str.len() > 0]


# remove headings from data frame
df = df[~df["text"].str.startswith("==")]

# Moding dates with multi events

prefix = ""
for (i, row) in df.iterrows():
    # If the row already has " - ", it already has the needed date prefix
    if " – " not in row["text"]:
        try:
            # If the row's text is a date, set it as the new prefix
            parse(row["text"])
            prefix = row["text"]
        except:
            # If the row's text isn't a date, add the prefix
            row["text"] = prefix + " – " + row["text"]

df = df[df["text"].str.contains(" – ")].reset_index(drop=True)

print(df)




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


print(embeddings[0])