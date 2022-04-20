#%%
import pandas as pd
output_name = 'ensemble'

#%%
csv_names = ['roberta_wwm_seed0.csv', 'roberta_wwm_seed326.csv', 'roberta_wwm_seed1121.csv', 'roberta_wwm_seed3261121.csv', 'roberta_wwm_seed1121326.csv']
for i, name in enumerate(csv_names):
    if i == 0:
        res = pd.read_csv(name)
        res["Answer_1"] = res["Answer"]
    else:
        res = pd.merge(
            left=res,
            right=pd.read_csv(name),
            on='ID',
            how='left',
            suffixes=("", f"_{i+1}")
        )

#%%
output = res.drop(["ID", "Answer"], axis=1).mode(axis=1)[[0]]
output["ID"] = output.index

#%%
output.set_axis(['Answer', 'ID'], axis=1, inplace=True)

#%%
output[["ID", "Answer"]].to_csv(f'./{output_name}.csv', index=False)

#%%
output[["ID", "Answer"]]

#%%
