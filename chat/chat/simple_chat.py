import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
import pandas as pd
from dateutil.parser import parser

import tiktoken


load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_URL = os.getenv('OPENAI_URL')


print("------------Params------------")
print(OPENAI_API_KEY)
print(OPENAI_URL)
print("------------Params------------")

# load wiki data




params = {
    "action": "query", 
    "prop": "extracts",
    "exlimit": 1,
    "titles": "2024–25_NBA_season",
    "explaintext": 1,
    "formatversion": 2,
    "format": "json"
}
resp = requests.get("https://en.wikipedia.org/w/api.php", params=params)
response_dict = resp.json()
print("____________________________")


text_data = response_dict["query"]["pages"][0]["extract"].split("\n")


better_spacing = "\n".join(text_data)

print(better_spacing)

df = pd.DataFrame()
df["text"] = text_data


df = df[(
    (df["text"].str.len() > 0) & (~df["text"].str.startswith("=="))
)].reset_index(drop=True)
df.head()


print("____________________________")

print(df)
print("____________________________")

print(df[:-30])

def create_prompt(question, df, max_token_count, answer_token_count):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    prompt_template = """
        Answer the question based on the context below, and if the question
        can't be answered based on the context, say "I don't know"

        Context: 
        {}
        ---
        Question: {}
        Answer:
    """
    current_token_count = len(tokenizer.encode(prompt_template)) + \
                        len(tokenizer.encode(question)) + \
                        answer_token_count
                            
    context = []
    for text in df["text"]:
        # Increase the counter based on the number of tokens in this row
        text_token_count = len(tokenizer.encode(text))
        current_token_count += text_token_count

        # Add the row of text to the list if we haven't exceeded the max
        if current_token_count <= max_token_count:
            context.append(text)
        else:
            break

    return prompt_template.format("\n###\n".join(context), question)

#

openai = OpenAI(
    base_url =OPENAI_URL,
    api_key = OPENAI_API_KEY,
)

def chat(prompt: str, debug: bool):

    if debug:
        print("....")

    openai = OpenAI(
        base_url =OPENAI_URL,
        api_key = OPENAI_API_KEY,
    )

    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt
    )
    if debug:
        print(response)
        print("....")


    return response.choices[0].text




print("____________________________")
print("***** Q1 *******")

question_klay_thompson = """
Question: What team does Klay Thompson play for ?
"""

#Example 1 of how model cannot answer
question_klay_thompson_answer = chat(prompt=question_klay_thompson, debug=True)

prompt_question_klay_thompson = create_prompt(question_klay_thompson, df, max_token_count=3500, answer_token_count=500)
print(prompt_question_klay_thompson)

prompt_prompt_question_klay_thompson_answer = chat(prompt_question_klay_thompson, True)

print(f"Without context answer to: {question_klay_thompson} \n  {question_klay_thompson_answer}")
print(f"With context answer to: {question_klay_thompson} \n  {prompt_prompt_question_klay_thompson_answer}")

print("____________________________")



print("____________________________")
print("***** Q2 *******")
#Example 2 of how model cannot answer
question_kemba= """
Question: What team does Kemba Walker play for?
"""
question_question_kemba_answer = chat(question_kemba, True)

prompt_question_kemba= create_prompt(question_kemba, df, max_token_count=3500, answer_token_count=500)

#print(prompt_question_jimmy_butler)

prompt_question_kemba_answer = chat(prompt_question_kemba, True)

print(f"Without context answer to: {question_kemba} \n  {question_question_kemba_answer}")
print(f"With context answer to: {question_kemba} \n  {prompt_question_kemba_answer}")

print("____________________________")


print("____________________________")
print("***** Q3 *******")

#Example 3 of how model cannot answer
question_russell="""
Question: What team does Russell Westbrook  play for?
"""
question_russell_answer = chat(prompt=question_russell, debug=True)
#print(question_russell_answer)

prompt_question_russell= create_prompt(question_russell, df, max_token_count=3500, answer_token_count=500)

#print(prompt_question_jimmy_butler)

prompt_question_russell_answer = chat(prompt_question_russell, True)

print(f"Without context answer to: {question_russell} \n  {question_russell_answer}")
print(f"With context answer to: {question_russell} \n  {prompt_question_russell_answer}")

print("____________________________")