import torch
from pre_trained.preprocess import preprocess_function
from datasets import Dataset
from flask import Flask, request, jsonify
from flask_cors import CORS
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

app = Flask(__name__)
CORS(app)

tokenizer = AutoTokenizer.from_pretrained('ViT5')
model = AutoModelForSeq2SeqLM.from_pretrained('ViT5')

@app.route('/q', methods=['POST'])
def generate_question():

    data = request.get_json()
    context = data.get('context', '')
    answer = data.get('answer', '')

    dict_obj = {'contexts': [context], 'answers': [answer]}
    data = Dataset.from_dict(dict_obj)
    tokenized_test = data.map(preprocess_function, batched=True, remove_columns=['contexts', 'answers'], num_proc=32)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

    max_target_length = 256
    dataloader = torch.utils.data.DataLoader(tokenized_test, collate_fn=data_collator, batch_size=32)

    for i, batch in enumerate(tqdm(dataloader)):
        outputs = model.generate(
            input_ids=batch['input_ids'],  # .to('cuda'),
            max_length=max_target_length,
            attention_mask=batch['attention_mask'],  # .to('cuda'),
        )
        with tokenizer.as_target_tokenizer():
            outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=True, skip_special_tokens=True) for out in
                       outputs]

    question = outputs[0]

    return jsonify({'prediction': question})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5005)