from SurvSet.data import SurvLoader

loader = SurvLoader()

# Identify all time‑varying dataset names from the loader metadata
td_names = loader.df_ds.loc[loader.df_ds['is_td'], 'ds'].tolist()
print("Time‑varying datasets:", td_names)

# Loop to load each dataset and save as CSV
for ds in td_names:
    df, ref = loader.load_dataset(ds_name=ds).values()
    filename = f"{ds}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Saved {filename}")
