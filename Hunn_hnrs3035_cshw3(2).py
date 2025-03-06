import os
import json
import time

import openai
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv('C:\\CSC4700\\Homework3\\.env')

# Establish client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-06-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Load dataset
with open('dev-v2.0 (1).json', 'r') as f:
    data = json.load(f)

# Create dict of the questions and answers
q_and_as = {}
for topic in data["data"]:
    for paragraph in topic["paragraphs"]:
        for qas in paragraph["qas"]:
            if not qas["is_impossible"]:
                question = qas["question"]
                q_id = qas["id"]
                ans = []
                for answer in qas["answers"]:
                    ans.append(answer["text"])
                    q_and_as[question, q_id] = ", ".join(ans)
            if len(q_and_as) >= 500:
                break
        if len(q_and_as) >= 500:
            break
    if len(q_and_as) >= 500:
        break
print(f'Data loaded from json dataset.')

answers = []
for q in q_and_as:
    user_prompt = f'Question: {q[0]}\n'
    print(user_prompt)

    # Use Llama to create an answer with the given prompts
    response = client.chat.completions.create(
            model="Llama-3.2-11B-Vision-Instruct",
            messages=[
                {"role": "system", "content": "You are a genius AI assistant that can answer any question."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=100
    )

    print(response)
    answers.append(response.choices[0].message.content)

# Save the answers into a file
output_file_name = "../output_llama.jsonl"
with open(output_file_name, 'w') as file:
    for i in answers:
        file.write(json.dumps(i) + '\n')
print(f'Responses saved to output file.')

results = []
with open(output_file_name, 'r') as file:
    for line in file:
        json_object = json.loads(line.strip())
        results.append(json_object)

print(f'Model answers: {answers}')