import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("C:\\Urfu\\ADuUU\\lab2\\brainsize.txt", sep="\t", na_values="NA")
features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height", "MRI_Count"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(
    df[df["Gender"] == "Male"][features].corr(),
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    ax=axes[0],
)
axes[0].set_title("Корреляции: Мужчины")

sns.heatmap(
    df[df["Gender"] == "Female"][features].corr(),
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    ax=axes[1],
)
axes[1].set_title("Корреляции: Женщины")
plt.tight_layout()
plt.show()

sns.lmplot(data=df, x="Height", y="MRI_Count", hue="Gender")
plt.title("Зависимость MRI_Count от Height")
plt.show()
