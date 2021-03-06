#%%
import json
import csv

#%%
result_dir = "macbert_runqa"
output_name = "macbert_runqa_5000step"

#%%
def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as reader:
        return json.load(reader)

#%%
predictions = read_json(f'./results/{result_dir}/predict_predictions.json')

#%%
with open(f'{output_name}.csv', 'w') as f:  
    writer = csv.writer(f)
    writer.writerow(["ID", "Answer"])
    for k, v in predictions.items():
       writer.writerow([k, v])

#%%
