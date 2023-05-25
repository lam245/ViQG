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

def prepare_ViMMRC(data_path:str):
    question_list = []
    context_list = []
    answer_list = []
    with open(data_path) as f:
        data_list = json.load(f)
    for data in tqdm(data_list):
        article = data['article']
        questions = data['questions']
        options = data['options']
        answers = data['answers']

        for i, question in enumerate(questions):
            answer = options[i][ord(answers[i]) - 65]
            question_list.append(question)
            context_list.append(article)
            answer_list.append(answer)
    dict_obj = {'contexts':context_list, 'answers':answer_list, 'questions':question_list}
    datasets = Dataset.from_dict(dict_obj)
    return datasets