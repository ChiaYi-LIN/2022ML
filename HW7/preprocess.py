#%%
import json
import pandas as pd

#%%
def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as reader:
        return json.load(reader)

def write_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as writer:
        json.dump(data, writer, ensure_ascii=False, indent=4)

#%%
train_data = read_json('data/hw7_train.json')
valid_data = read_json('data/hw7_dev.json')
test_data = read_json('data/hw7_test.json')

train_questions = train_data['questions']
valid_questions = valid_data['questions']
test_questions = test_data['questions']

train_paragraphs = train_data['paragraphs']
valid_paragraphs = valid_data['paragraphs']
test_paragraphs = test_data['paragraphs']

#%%
def unfold_questions(mode, questions, paragraphs):
    for question in questions:
        question['context'] = paragraphs[question['paragraph_id']]
        question['question'] = question['question_text']
        
        if mode != 'test':
            question['answers'] = {
                'answer_start': [question['answer_start']],
                'text': [question['context'][question['answer_start'] : question['answer_end'] + 1]],
            }
        else:
            question['answers'] = {
                'answer_start': [None],
                'text': [None],
            }

    return pd.DataFrame(questions)[['id', 'context', 'question', 'answers']]

unfold_train_data = unfold_questions('train', train_questions, train_paragraphs)
unfold_valid_data = unfold_questions('valid', valid_questions, valid_paragraphs)
unfold_test_data = unfold_questions('test', test_questions, test_paragraphs)

#%%
all = pd.concat([unfold_train_data, unfold_valid_data], ignore_index=True)
all['id'] = all.index

#%%
all_json = {"data" : all.to_dict(orient="records")}
test_json = {"data" : unfold_test_data.to_dict(orient="records")}

#%%
write_json("./data/all.json", all_json)
write_json("./data/test.json", test_json)

#%%
