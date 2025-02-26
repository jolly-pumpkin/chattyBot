import os
from openai import OpenAI
from dotenv import load_dotenv

import pandas as pd
import numpy as np


df = pd.read_csv("distance.csv")

current_shortest = df.iloc[0]["distances"]
current_index = 0
for index, distance in enumerate(df["distances"].values):
    if distance < current_shortest:
        current_shortest = distance
        current_index = index


print('loc', current_shortest, index)


df.sort_values(by="distances")

print(df)



print(df.iloc[0]["text"])