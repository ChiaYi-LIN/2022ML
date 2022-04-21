#%%
"""## Import Packages"""
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

# bert-base-chinese
# hfl/chinese-bert-wwm
# hfl/chinese-bert-wwm-ext
# hfl/chinese-roberta-wwm-ext
# hfl/chinese-roberta-wwm-ext-large
# hfl/chinese-macbert-base
# hfl/chinese-macbert-large
model_checkpoints = ['./model/roberta_wwm_seed0', './model/roberta_wwm_seed326', './model/roberta_wwm_seed1121', './model/roberta_wwm_seed3261121', './model/roberta_wwm_seed1121326', './model/roberta_wwm_seed3', './model/roberta_wwm_seed6']
config = {
    "output_name" : "ensemble_logits_7",
    "tokenizer_checkpoint" : "hfl/chinese-roberta-wwm-ext",
    "batch_size" : 256,
    "max_length" : 384,
    "stride" : 128,
    "n_best" : 20,
    "max_answer_length" : 30,
    "device" : "cuda" if torch.cuda.is_available() else "cpu",
}

#%%
"""## Read Data"""
def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data

test_data = read_data("./data/hw7_test.json")
test_questions = test_data['questions']
test_paragraphs = test_data['paragraphs']

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

unfold_test_data = unfold_questions('test', test_questions, test_paragraphs)

#%%
"""## Data to Dataset"""
test_dataset = Dataset.from_pandas(pd.DataFrame(data=unfold_test_data))

#%%
"""
# Question Answering
"""

#%%
"""## Load Tokenizer"""
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_checkpoint"], use_fast=True)

#%%
def prepare_validation_features(examples):
    question_column_name = "question"
    context_column_name = "context"
    pad_on_right = True
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=config["max_length"],
        stride=config["stride"],
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

tokenized_test_dataset = test_dataset.map(prepare_validation_features, batched=True, remove_columns=test_dataset.column_names, desc="Running tokenizer on prediction dataset")

#%%
"""## Load Data Collator"""
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator()

#%%
"""## Post-processing & Compute Exact-Match"""
from transformers import EvalPrediction
from utils_qa import postprocess_qa_predictions
from datasets import load_metric

metric = load_metric("squad")
def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=False,
        n_best_size=config["n_best"],
        max_answer_length=config["max_answer_length"],
        output_dir=f'./results/{config["output_name"]}',
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    
    return formatted_predictions

def get_predictions(examples, features, predictions, stage="eval"):
    return predictions

def compute_metrics(p: EvalPrediction):
    return metric.compute(predictions=p.predictions, references=p.label_ids)

#%%
from transformers import AutoModelForQuestionAnswering
from transformers import TrainingArguments
from trainer_qa import QuestionAnsweringTrainer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
def get_logits(model_checkpoint):
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint).to(config["device"])
    print(f'Using Model: {model_checkpoint}')
    training_args = TrainingArguments(
        output_dir=f'./results/{config["output_name"]}',
        fp16=True,
        per_device_eval_batch_size=config["batch_size"],
        label_names=["start_positions", "end_positions"],
    )
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=get_predictions,
        compute_metrics=compute_metrics,
    )
    logits = trainer.predict(tokenized_test_dataset, test_dataset)
    return logits

#%%
"""## Ensemble Logits"""
all_logits = []
for cp in model_checkpoints:
    logits = get_logits(cp)
    all_logits.append(logits)

sum_start_logits = np.zeros(all_logits[0][0].shape)
sum_end_logits = np.zeros(all_logits[0][1].shape)

for logits in all_logits:
    sum_start_logits += logits[0]
    sum_end_logits += logits[1]

#%%
"""## Generate Predictions"""
predictions = post_processing_function(test_dataset, tokenized_test_dataset, (sum_start_logits, sum_end_logits), "pred")
predictions_df = pd.DataFrame(predictions)
predictions_df['ID'] = predictions_df['id']
predictions_df['Answer'] = predictions_df['prediction_text']
predictions_df[['ID', 'Answer']].to_csv(f'./{config["output_name"]}.csv', index=False)