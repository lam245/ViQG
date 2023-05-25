import json
import tqdm
from datasets import Dataset
def load_json(data_path:str):
    questions = []
    contexts, answers = [],[]
    with open(data_path) as f:
        data = json.load(f)
        del data["version"]
    for i in tqdm.tqdm(range(len(data['data']))):
        for paragraph in data['data'][i]['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                context = paragraph['context']
                answer = qa['answers'][0]['text']
                answers.append(answer)
                contexts.append(context)
                questions.append(question)
    dict_obj = {'contexts':contexts, 'answers':answers, 'questions':questions}
    datasets = Dataset.from_dict(dict_obj)
    return datasets