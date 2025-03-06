from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time


# load environmental variables
load_dotenv('.env')

# System and user prompt to be filled in
system_prompt = "You are a genius AI assistant who can answer any question concisely."
user_prompt = 'The question is: "{question}". Return your response in json, with one key, being "response"'

with open('dev-v2.0 (1).json', 'r') as f:
    data = json.load(f)
questions = []
count = 0
for item in data['data']:
    for paragraph in item['paragraphs']:
        for qas in paragraph['qas']:
            if not qas['is_impossible']:  # Skip impossible questions
                questions.append({
                    "id": qas['id'],
                    "question": qas['question']
                })
                count+=1
            if count >= 500:  # Stop after 500 questions
                break
        if count >= 500:
            break
    if count >= 500:
        break

tasks = []
for question in questions:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(question=question['question'])}
    ]


    custom_id = f"question={question['id']}"

    task = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini", 
            "temperature": 0.2,
            "max_tokens": 100,
            "messages": messages
        }
    }
    tasks.append(task)


# Here, we are writing a local file to store the tasks. This is a jsonl file, newline delimited)
with open("../input_batch.jsonl", 'w') as jfile:
    for task in tasks:
        jfile.write(json.dumps(task) + '\n')

# establish OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Upload the batch file to openai
batch_file = client.files.create(
    file=open("../input_batch.jsonl", 'rb'),
    purpose='batch'
)

# Run the batch using the completions endpoint
batch_job = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)

# loop until the status of our batch is completed
complete = False
while not complete:
    check = client.batches.retrieve(batch_job.id)
    print(f'Status: {check.status}')
    if check.status == 'completed':
        complete = True
    time.sleep(1)
print("Done processing batch.")

print("Writing data...")
# Write the results to a local file in jsonl format
result = client.files.content(check.output_file_id).content
output_file_name = "../output_batch.jsonl"
with open(output_file_name, 'wb') as file:
    file.write(result)

# load the output file, extract each sample output, and append to a list
results = []
with open(output_file_name, 'r') as file:
    for line in file:
        # this converts the string into a Json object
        json_object = json.loads(line.strip())
        results.append(json_object)

# Show the responses
for item in results:
    print("Model's Response:")
    print('\t', item['response']['body']['choices'][0]['message']['content'])
