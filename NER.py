import pandas as pd
import numpy as np
from datasets import load_metric
from datasets import DatasetDict, Dataset
from transformers import Trainer, TrainingArguments, AutoModelForTokenClassification,AutoTokenizer

'''df = pd.read_json('/kaggle/input/covid19/train_word.json',lines = True)
df1 = pd.read_json('/kaggle/input/covid19/dev_word.json',lines = True)
df2 = pd.read_json('/kaggle/input/covid19/test_word.json',lines = True)'''
df = pd.read_json('/content/drive/MyDrive/Datasets/Covid19/train_word.json',lines = True)
df1 = pd.read_json('/content/drive/MyDrive/Datasets/Covid19/dev_word.json',lines = True)
df2 = pd.read_json('/content/drive/MyDrive/Datasets/Covid19/test_word.json',lines = True)

train_src, train_tg = df['words'].to_list(), df['tags'].to_list()
valid_src, valid_tg = df1['words'].to_list(), df1['tags'].to_list()
test_src, test_tg = df2['words'].to_list(), df2['tags'].to_list()

list_label = ['O', 'B-AGE', 'I-AGE', 'B-DATE', 'I-DATE', 'B-GENDER', 'I-GENDER', 'B-JOB', 'I-JOB'
              'B-LOCATION', 'I-LOCATION', 'B-NAME', 'I-NAME', 'B-ORGANIZATION', 'I-ORGANIZATION',
              'B-PATIENT_ID', 'I-PATIENT_ID', 'B-SYMPTOM_AND_DISEASE', 'I-SYMPTOM_AND_DISEASE', 'B-TRANSPORTATION', 'I-TRANSPORTATION']

dict_label = {'O': 0,
'B-AGE': 1,
'I-AGE': 2,
'B-DATE': 3,
'I-DATE': 4,
'B-GENDER': 5,
'I-GENDER': 6,
'B-JOB': 7,
'I-JOB': 8,
'B-LOCATION': 9,
'I-LOCATION': 10,
'B-NAME': 11,
'I-NAME': 12,
'B-ORGANIZATION': 13,
'I-ORGANIZATION': 14,
'B-PATIENT_ID': 15,
'I-PATIENT_ID': 16,
'B-SYMPTOM_AND_DISEASE': 17,
'I-SYMPTOM_AND_DISEASE': 18,
'B-TRANSPORTATION': 19,
'I-TRANSPORTATION': 20}

def convert_ids(list, label_dict):
  tg = []
  for i in list:
    sent = []
    for j in i:
      sent.append(label_dict[j])
    tg.append(sent)
  return tg

train_tg_ids = convert_ids(train_tg, dict_label)
valid_tg_ids = convert_ids(valid_tg, dict_label)
test_tg_ids = convert_ids(test_tg, dict_label)

data_train = {
    'tokens': train_src,
    'ner_tags': train_tg_ids
}
data_valid = {
    'tokens': valid_src,
    'ner_tags': valid_tg_ids
}
data_test = {
    'tokens': test_src,
    'ner_tags': test_tg_ids
}
train_dataset = Dataset.from_dict(data_train)
valid_dataset = Dataset.from_dict(data_valid)
test_dataset = Dataset.from_dict(data_test)

dataset = DatasetDict({
    'Train': train_dataset,
    'Valid': valid_dataset,
    'Test': test_dataset
})


model = AutoModelForTokenClassification.from_pretrained("manhtt-079/vipubmed-deberta-xsmall")
tokenizer = AutoTokenizer.from_pretrained("manhtt-079/vipubmed-deberta-xsmall")

label_all_tokens = True
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], max_length=256,
        padding="max_length", truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels,remove_columns=['tokens'], batched=True, num_proc=8)

model_name = 'vipubmed-deberta-base'
args = TrainingArguments(
    f"{model_name}-finetuned-ner",
    evaluation_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    save_strategy="no",
    gradient_accumulation_steps = 50
)

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer)

metric = load_metric("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [list_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [list_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["Train"],
    eval_dataset=tokenized_datasets["Valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

predictions, labels, _ = trainer.predict(tokenized_datasets["Test"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [list_label[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [list_label[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
print(results)