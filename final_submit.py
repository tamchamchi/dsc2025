import pandas as pd

df_01 = pd.read_csv("./data/submit_01.csv")
df_02 = pd.read_csv("./data/submit_02.csv")
df_12 = pd.read_csv("./data/submit_12.csv")

df_all = pd.concat([df_01, df_02, df_12], ignore_index=True)

df_private = pd.read_csv("./data/qwen2.5-7b.csv")

mapping = df_all.set_index("id")["predict_label"]
df_private.loc[df_private["id"].isin(df_all["id"]), "predict_label"] = (
    df_private.loc[df_private["id"].isin(df_all["id"]), "id"].map(mapping)
)

df_private.set_index("id").to_csv("submit.csv")
print("Final Submit saved in ./data/submit.csv")
