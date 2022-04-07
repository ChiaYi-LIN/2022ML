#%%
import pandas as pd
gnorm_df = pd.read_csv("./gnorm.csv", index_col=0, names=["gnorm"], header=0)
ax = gnorm_df.plot(figsize=(20, 5))
ax.set_xlim(0, len(gnorm_df))
ax.set_xlabel("Step")
ax.set_ylabel("Grad norm")
ax.axhline(y=1, xmin=0, xmax=len(gnorm_df), color='r', linestyle='--')
fig = ax.get_figure()
fig.savefig('gnorm.jpg')
#%%
