#%%
# !gdown --id '1AVgZvy3VFeg0fX-6WQJMHPVrx3A-M1kb' --output hw7_data.zip

#%%
# !mkdir -p ./data
# !unzip -o hw7_data.zip -d ./data

#%%
# !pip install transformers==4.18.0
# !pip install datasets==2.1.0

#%%
"""## Import Packages"""
import random
import json
import numpy as np
import pandas as pd
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    DefaultDataCollator,
    TrainingArguments, 
)

output_name = "roberta_wwm_stride128"
# bert-base-chinese
# hfl/chinese-roberta-wwm-ext
tokenizer_checkpoint = "hfl/chinese-roberta-wwm-ext"
model_checkpoint = "hfl/chinese-roberta-wwm-ext"
batch_size = 2
num_epoch = 1
set_seed = 0
max_length = 384
stride = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

#%%
"""## Set Seeds"""
# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(set_seed)

#%%
"""## Read Data"""
def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data

train_data = read_data("./data/hw7_train.json")
valid_data = read_data("./data/hw7_dev.json")
test_data = read_data("./data/hw7_test.json")

train_questions = train_data['questions']
valid_questions = valid_data['questions']
test_questions = test_data['questions']

train_paragraphs = train_data['paragraphs']
valid_paragraphs = valid_data['paragraphs']
test_paragraphs = test_data['paragraphs']

#%%
train_questions[0]

#%%
train_paragraphs[3884][141:145]

#%%
"""## Paragraph ID to Context"""
def unfold_questions(mode, questions, paragraphs):
    for question in questions:
        question['context'] = paragraphs[question['paragraph_id']]
        question['question'] = question['question_text']
        
        if mode != 'test':
            question['answers'] = {
                'answer_start': [question['answer_start']],
                'text': [question['context'][question['answer_start'] : question['answer_end'] + 1]],
            }

    if mode != 'test':
        return pd.DataFrame(questions)[['id', 'context', 'question', 'answers']]

    return pd.DataFrame(questions)[['id', 'context', 'question']]

unfold_train_data = unfold_questions('train', train_questions, train_paragraphs)
unfold_valid_data = unfold_questions('valid', valid_questions, valid_paragraphs)
unfold_test_data = unfold_questions('test', test_questions, test_paragraphs)

#%%
"""## Data to Dataset"""
train_dataset = Dataset.from_pandas(pd.DataFrame(data=unfold_train_data))
valid_dataset = Dataset.from_pandas(pd.DataFrame(data=unfold_valid_data))
test_dataset = Dataset.from_pandas(pd.DataFrame(data=unfold_test_data))

#%%
train_dataset[1]

#%%
"""
# Question Answering
"""

#%%
"""## Load Tokenizer"""
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

#%%
"""## Preprocess & Tokenize Data"""
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs["offset_mapping"]
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]
        
    inputs["example_id"] = example_ids
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True, remove_columns=valid_dataset.column_names)

#%%
for k, v in tokenized_train_dataset[0].items() :
    print (k)

#%%
"""## Load Data Collator"""
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator()

#%%
"""## Load QA Model"""
from transformers import AutoModelForQuestionAnswering
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint).to(device)

#%%
"""## Post-processing & Compute Exact-Match"""
from transformers import EvalPrediction
from utils_qa import postprocess_qa_predictions
from datasets import load_metric

n_best = 20
max_answer_length = 30
metric = load_metric("squad")

def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=False,
        n_best_size=n_best,
        max_answer_length=max_answer_length,
        output_dir=training_args.output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    try:
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    except:
        return formatted_predictions

def compute_metrics(p: EvalPrediction):
    return metric.compute(predictions=p.predictions, references=p.label_ids)

#%%
"""## Trainer"""
from transformers import TrainingArguments
from trainer_qa import QuestionAnsweringTrainer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

training_args = TrainingArguments(
    seed=set_seed,
    output_dir=f"./results/{output_name}",
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    fp16=True,
    per_device_eval_batch_size=batch_size,
    optim='adamw_torch',
    learning_rate=3e-5,
    # weight_decay=0.01,
    num_train_epochs=num_epoch,
    # warmup_ratio=0.05,
    logging_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="exact_match",
    label_names=["start_positions", "end_positions"],
)

trainer = QuestionAnsweringTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    eval_examples=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    post_process_function=post_processing_function,
    compute_metrics=compute_metrics,
)

#%%
"""## Training"""
trainer.train()

#%%
"""## Save QA Model"""
trainer.save_model(f'./model/{output_name}')

#%%
"""## Preprocess Test Data"""
def preprocess_test_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

tokenized_test_dataset = test_dataset.map(preprocess_test_examples, batched=True, remove_columns=test_dataset.column_names)

#%%
"""## Generate Predictions"""
predictions = trainer.predict(tokenized_test_dataset, test_dataset)
predictions_df = pd.DataFrame(predictions)
predictions_df['ID'] = predictions_df['id']
predictions_df['Answer'] = predictions_df['prediction_text']
predictions_df[['ID', 'Answer']].to_csv(f'./{output_name}.csv', index=False)

#%%
