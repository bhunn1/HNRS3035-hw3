import re
import time

from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv('.env')

with open('dev-v2.0 (1).json', 'r') as f:
    data = json.load(f)

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
                    q_and_as[(question, q_id)] = ans
            if len(q_and_as) >= 500:
                break
        if len(q_and_as) >= 500:
            break
    if len(q_and_as) >= 500:
        break

context = {}

def score_llama():
    """
    method that scores the accuracy of the Llama 3.2 11B Vision Instruct answers

    :return:
    """

    llama_answers = []
    with open('../output_llama.jsonl', 'r') as f:
        for line in f:
            llama_answers.append(json.loads(line.strip()))

    scoring_prompt = (
        "You are a teacher tasked with determining whether a student’s answer to a question was correct, based "
        "on a set of possible correct answers. You must only use the provided correct answers to determine if "
        "the student's response was correct.")
    user_prompt = ("Question: {question}\n\n"
                   "Student’s Response: {student_response}\n\n"
                   "Possible Correct Answers:{correct_answers}\n\n"
                   "Your response should only be a valid Json as shown below:\n"
                   "{{\n"
                   "    \"explanation\" (str): \"A short explanation of why the student’s answer was correct or incorrect.\",\n"
                   "    \"score\" (bool): \"true if the student’s answer was correct, false if it was incorrect.\"\n"
                   "}}\n\n"
                   "Your response:")

    # Loop through the Q&A's and form the API calls to gpt-4o with the proper prompts and parameters
    tasks = []
    count = 0
    for (q, q_id), a in q_and_as.items():
        messages = [
            {"role": "system", "content": scoring_prompt},
            {"role": "user", "content": user_prompt.format(question=q, student_response=llama_answers[count], correct_answers=a)}
        ]

        custom_id = q_id
        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "temperature": 0.2,
                "max_tokens": 100,
                "response_format": {"type": "json_schema",
                                    "json_schema": {
                                        "name": "score_response",
                                        "strict": True,
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "explanation": {
                                                    "type": "string",
                                                },
                                                "score": {
                                                    "type": "boolean",
                                                }
                                            },
                                            "required": ["explanation", "score"],
                                            "additionalProperties": False
                                        },
                                    }
                                    },
                "messages": messages
            }
            }
        tasks.append(task)
        count += 1

    print(tasks[0])

    # Dump the llama model's answers into a jsonl file for the batch
    with open("../llama_score.jsonl", 'w') as jfile:
        for task in tasks:
            jfile.write(json.dumps(task) + '\n')
    print(f'Batch input file created.')

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Upload the batch file to openai
    batch_file = client.files.create(
        file=open("../llama_score.jsonl", 'rb'),
        purpose='batch'
    )
    batch_id = batch_file.id
    print(f'Batch file uploaded successfully. File ID: {batch_id}')

    # Submit the batch job to the completions endpoint
    batch_job = client.batches.create(
        input_file_id=batch_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    batch_job_id = batch_job.id
    print(f'Batch job submitted. Batch job ID: {batch_job_id}')

    # Check status of the batch job
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job_id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        if check.status == 'failed':
            print(f'Batch failed! {check}')
            exit()
        time.sleep(30)
    print("Done processing batch.")

    # Get the results back from GPT-4o
    results = []
    result_bytes = client.files.content(check.output_file_id).content
    result_str = result_bytes.decode('utf-8')
    print(result_str)  # Debugging: Check the raw response

    for line in result_str.splitlines():
        data = json.loads(line)
        content_str = data["response"]["body"]["choices"][0]["message"]["content"].strip()

        # Parse the content as JSON
        content_json = json.loads(content_str)

        # Extract the score
        score = content_json.get("score", False)  # Default to False if missing
        results.append(score)

    print(results)
    # Count correct answers
    count_correct = sum(1 for score in results if score)

    print(f"LLama's accuracy was {(count_correct/len(results))*100}")
    return ((count_correct/len(results))*100)

def score_mini():
    """
        method that scores the accuracy of the GPT-4o-mini answers

        :return:
        """

    mini_answers = []
    with open('C:\\CSC4700\\output_batch.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            mini_results = data["response"]["body"]["choices"][0]["message"]["content"].strip()
            print(f'mini results1: {mini_results}')
            if mini_results.startswith("```json"):
                mini_results = re.sub(r"```json\n(.*?)\n```", r"\1", mini_results, flags=re.DOTALL)
            print(f'mini results: {mini_results}')
            content = json.loads(mini_results)
            txt = content.get("response", "No resposne")
            mini_answers.append(txt)

    scoring_prompt = (
        "You are a teacher tasked with determining whether a student’s answer to a question was correct, based "
        "on a set of possible correct answers. You must only use the provided correct answers to determine if "
        "the student's response was correct.")
    user_prompt = ("Question: {question}\n\n"
                   "Student’s Response: {student_response}\n\n"
                   "Possible Correct Answers:{correct_answers}\n\n"
                   "Your response should only be a valid Json as shown below:\n"
                   "{{\n"
                   "    \"explanation\" (str): \"A short explanation of why the student’s answer was correct or incorrect.\",\n"
                   "    \"score\" (bool): \"true if the student’s answer was correct, false if it was incorrect.\"\n"
                   "}}\n\n"
                   "Your response:")

    # Loop through the Q&A's and form the API calls to gpt-4o with the proper prompts and parameters
    tasks = []
    count = 0
    for (q, q_id), a in q_and_as.items():

        messages = [
            {"role": "system", "content": scoring_prompt},
            {"role": "user",
             "content": user_prompt.format(question=q, student_response=mini_answers[count], correct_answers=a)}
        ]

        custom_id = q_id
        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "temperature": 0.2,
                "max_tokens": 100,
                "response_format": {"type": "json_schema",
                                    "json_schema": {
                                        "name": "score_response",
                                        "strict": True,
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "explanation": {
                                                    "type": "string",
                                                },
                                                "score": {
                                                    "type": "boolean",
                                                }
                                            },
                                            "required": ["explanation", "score"],
                                            "additionalProperties": False
                                        },
                                    }
                                    },
                "messages": messages
            }
        }
        tasks.append(task)
        count += 1


    # Dump the llama model's answers into a jsonl file for the batch
    with open("../mini_score.jsonl", 'w') as jfile:
        for task in tasks:
            jfile.write(json.dumps(task) + '\n')
    print(f'Batch input file created.')

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Upload the batch file to openai
    batch_file = client.files.create(
        file=open("../mini_score.jsonl", 'rb'),
        purpose='batch'
    )
    batch_id = batch_file.id
    print(f'Batch file uploaded successfully. File ID: {batch_id}')

    # Submit the batch job to the completions endpoint
    batch_job = client.batches.create(
        input_file_id=batch_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    batch_job_id = batch_job.id
    print(f'Batch job submitted. Batch job ID: {batch_job_id}')

    # Check status of the batch job
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job_id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        if check.status == 'failed':
            print(f'Batch failed! {check}')
            exit()
        time.sleep(30)
    print("Done processing batch.")

    # Get the results back from GPT-4o
    results = []
    result_bytes = client.files.content(check.output_file_id).content
    result_str = result_bytes.decode('utf-8')
    print(result_str)  # Debugging: Check the raw response

    for line in result_str.splitlines():
        data = json.loads(line)  # Parse each line as JSON
        content_str = data["response"]["body"]["choices"][0]["message"]["content"].strip()

        # Parse the content as JSON
        content_json = json.loads(content_str)

        # Extract the score
        score = content_json.get("score", False)  # Default to False if missing
        results.append(score)

    print(results)
    # Count correct answers
    count_correct = sum(1 for score in results if score)

    print(f"GPT-Mini's accuracy was {(count_correct / len(results)) * 100}")
    return ((count_correct / len(results)) * 100)

if __name__ == '__main__':
    score_llama()
    score_mini()
