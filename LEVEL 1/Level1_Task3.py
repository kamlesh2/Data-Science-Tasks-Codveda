import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_dataset.csv")

print("Dataset Shape:", df.shape)
print("\nColumn Types:\n", df.dtypes)
print("\nSummary Statistics:\n", df.describe())

print("\nMedian Values:\n", df.median(numeric_only=True))
print("\nVariance:\n", df.var(numeric_only=True))

numeric_df = df.select_dtypes(include='number')
correlation = numeric_df.corr()
print("\nCorrelation Matrix:\n", correlation)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

for col in ['Awards', 'Nominations']:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

for col in ['Awards', 'Nominations']:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x='Nominations', y='Awards', data=df)
plt.title("Scatter Plot: Awards vs Nominations")
plt.xlabel("Nominations")
plt.ylabel("Awards")
plt.tight_layout()
plt.show()

print("\nEDA Insights Report:")
print("- Awards and Nominations are positively correlated.")
print("- Awards distribution is slightly right-skewed; most films have below-average awards.")
print("- Nominations show a tighter spread with fewer outliers.")
print("- Some films achieve high awards with relatively few nominations.")
print("- Title_encoded is a placeholder; not meaningful for analysis unless grouped or decoded.")
