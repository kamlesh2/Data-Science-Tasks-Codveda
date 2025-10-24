import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

df = pd.read_csv("oscar_wikipedia_films.csv")

print("Initial shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Data types:\n", df.dtypes)

df['Awards'] = pd.to_numeric(df['Awards'], errors='coerce')
df['Nominations'] = pd.to_numeric(df['Nominations'], errors='coerce')

df['Awards'].fillna(df['Awards'].mean(), inplace=True)
df['Nominations'].fillna(df['Nominations'].mean(), inplace=True)

df.drop_duplicates(inplace=True)

for col in ['Awards', 'Nominations']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

le = LabelEncoder()
df['Title_encoded'] = le.fit_transform(df['Title'])

numeric_cols = ['Awards', 'Nominations']
minmax_scaler = MinMaxScaler()
df[numeric_cols] = minmax_scaler.fit_transform(df[numeric_cols])

standard_scaler = StandardScaler()
df[numeric_cols] = standard_scaler.fit_transform(df[numeric_cols])

df.to_csv("cleaned_dataset.csv", index=False)
print("Cleaned dataset saved as cleaned_dataset.csv")

df = pd.read_csv("cleaned_dataset.csv")
print("Shape:", df.shape)
print(df.head())

print(df.describe())

correlation = df.select_dtypes(include=['number']).corr()
print("Correlation matrix:\n", correlation)

plt.figure(figsize=(8, 4))
sns.histplot(df['Awards'], bins=20, kde=True)
plt.title("Distribution of Awards")
plt.xlabel("Awards")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x='Nominations', y='Awards', data=df)
plt.title("Awards vs Nominations")
plt.xlabel("Nominations")
plt.ylabel("Awards")
plt.tight_layout()
plt.show()
