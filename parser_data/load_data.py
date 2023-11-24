import json
import tqdm
import pandas as pd

from datasets import Dataset
import json
from tqdm import tqdm


def load_json(data_path: str):
    with open(data_path) as f:
        data_list = json.load(f)

    contexts = []
    questions = []
    answers = []

    for data in  tqdm.tqdm(data_list, desc="Processing data"):  # Use tqdm as a function, not a module
        context = data.get("input", "")
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
