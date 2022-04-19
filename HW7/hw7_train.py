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

# bert-base-chinese
# hfl/chinese-bert-wwm
# hfl/chinese-bert-wwm-ext
# hfl/chinese-roberta-wwm-ext
# hfl/chinese-roberta-wwm-ext-large
# hfl/chinese-xlnet-base
# hfl/chinese-macbert-large
config = {
    "output_name" : "roberta_wwm_seed0",
    "tokenizer_checkpoint" : "hfl/chinese-roberta-wwm-ext",
    "model_checkpoint" : "hfl/chinese-roberta-wwm-ext",
    "batch_size" : 16,
    "gradient_accumulation_steps": 2,
    "learning_rate" : 1e-4,
    "num_epoch" : 1,
    "set_seed" : 0,
    "max_length" : 384,
    "stride" : 128,
    "n_best" : 20,
    "max_answer_length" : 30,
    "device" : "cuda" if torch.cuda.is_available() else "cpu",
}

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
same_seeds(config["set_seed"])

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
tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_checkpoint"], use_fast=True)

#%%
"""## Preprocess & Tokenize Data"""
def prepare_train_features(examples):
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"
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
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

tokenized_train_dataset = train_dataset.map(prepare_train_features, batched=True, remove_columns=train_dataset.column_names, desc="Running tokenizer on train dataset")

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

tokenized_valid_dataset = valid_dataset.map(prepare_validation_features, batched=True, remove_columns=valid_dataset.column_names, desc="Running tokenizer on validation dataset")
tokenized_test_dataset = test_dataset.map(prepare_validation_features, batched=True, remove_columns=test_dataset.column_names, desc="Running tokenizer on prediction dataset")

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
model = AutoModelForQuestionAnswering.from_pretrained(config["model_checkpoint"]).to(config["device"])

#%%
"""## Post-processing & Compute Exact-Match"""
from transformers import EvalPrediction
from utils_qa import postprocess_qa_predictions
from datasets import load_metric

metric = load_metric("squad")

def post_processing_function(examples, features, predictions, stage="eval"):
    answer_column_name = "answers"
    
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=False,
        n_best_size=config["n_best"],
        max_answer_length=config["max_answer_length"],
        output_dir=training_args.output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    try:
        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
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
    seed=config["set_seed"],
    output_dir=f'./results/{config["output_name"]}',
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=config["batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    fp16=True,
    per_device_eval_batch_size=config["batch_size"],
    optim='adamw_torch',
    learning_rate=config["learning_rate"],
    # weight_decay=0.01,
    num_train_epochs=config["num_epoch"],
    # warmup_ratio=0.05,
    logging_steps=100,
    save_strategy="steps",
    save_steps=100,
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
trainer.save_model(f'./model/{config["output_name"]}')

#%%
# """## Generate Predictions"""
predictions = trainer.predict(tokenized_test_dataset, test_dataset)
predictions_df = pd.DataFrame(predictions)
predictions_df['ID'] = predictions_df['id']
predictions_df['Answer'] = predictions_df['prediction_text']
predictions_df[['ID', 'Answer']].to_csv(f'./{config["output_name"]}.csv', index=False)

#%%
# df = pd.DataFrame(valid_questions)
# df["answer_len"] = df["answer_end"] - df["answer_start"] + 1
# df["longer"] = df["answer_len"] > 15
# df["longer"].sum() / len(df)

#%%
