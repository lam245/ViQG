import json
import tqdm
import pandas as pd

from datasets import Dataset
def load_json(data_path: str, dataset: str):
    questions = []
    contexts, answers = [], []

    if dataset in ['ViNewsQA', 'ViQuAD', 'ViMMRC1.0', 'ViMMRC2.0']:
        df = pd.read_parquet(data_path)

        for i in range(len(df)):
            if dataset in ['ViNewsQA', 'ViQuAD']:
                question = df.loc[i, 'question']
                context = df.loc[i, 'context']
                answer = df.loc[i, 'answers']
            elif dataset in ['ViMMRC1.0', 'ViMMRC2.0']:
                question = df.loc[i, 'question']
                context = df.loc[i, 'article']
                answer = df.loc[i, 'answers']
            else:
                question = df.loc[i, 'input_text']
                context = df.loc[i, 'story']
                answer = df.loc[i, 'input_text']  # Assuming the answer is in the same column

            questions.append(question)
            contexts.append(context)
            answers.append(answer)

    dict_obj = {'contexts': contexts, 'answers': answers, 'questions': questions}
    datasets = Dataset.from_dict(dict_obj)  # Assuming you have a Dataset class or function
    return datasets