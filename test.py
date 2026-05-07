import pandas as pd
from datetime import date
df = pd.read_csv('../data/austria/processed/all_results.csv')
df['date'] = pd.to_datetime(df['date']).dt.date
start = date(2025,10,6)
end   = date(2025,10,12)
mask  = (df['date']>=start) & (df['date']<=end) & (df['played']==1)
print(len(df[mask]))
print(df[mask][['date','home_name']].head())
