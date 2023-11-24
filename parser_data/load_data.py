import json
import tqdm
import pandas as pd

from datasets import Dataset
def load_json(data_path: str):
    with open(data_path) as f:
        data_list = json.load(f)

    contexts = []
    questions = []
    answers = []

    for data in tqdm(data_list):
        context = data.get("input", "")
        print(context)
        instruction = data.get("instruction", "")
        if "output" in data:
            output_text = data["output"]
            qa_pairs = [pair.split(': ') for pair in output_text.split(' [SEP] ')]
            for qa_pair in qa_pairs:
                if len(qa_pair) == 2:
                    question, answer = qa_pair
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    dict_obj = {'contexts': contexts, 'questions': questions, 'answers': answers}
    datasets = Dataset.from_dict(dict_obj)
    return datasets