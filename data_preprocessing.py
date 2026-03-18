# 📊 Disease Symptom Dataset - Data Preprocessing and Visualization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Step 1: Load dataset
df = pd.read_csv("disease_symptoms.csv")

# ✅ Step 2: Handle missing values
df = df.fillna("")

# ✅ Step 3: Combine all symptom columns into one text column
symptom_cols = [col for col in df.columns if "Symptom" in col]
df["all_symptoms"] = df[symptom_cols].apply(lambda x: " ".join(x.astype(str)), axis=1)

# ✅ Step 4: Lowercase all text
df["all_symptoms"] = df["all_symptoms"].str.lower()
df["Disease"] = df["Disease"].str.lower()

print("✅ Data Preprocessing Completed")
print(df.head())

# -------------------------------------------------------------------
# 🧩 Visualization Section
# -------------------------------------------------------------------

# 🎯 1️⃣ Top 10 Most Common Diseases
plt.figure(figsize=(10, 5))
df["Disease"].value_counts().head(10).plot(kind="bar", color="royalblue")
plt.title("Top 10 Most Common Diseases")
plt.xlabel("Disease")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 🎯 2️⃣ Top 10 Most Frequent Symptoms
from collections import Counter

# Split all symptoms and count frequency
all_symptoms = " ".join(df["all_symptoms"]).split()
symptom_freq = Counter(all_symptoms)
symptom_df = pd.DataFrame(symptom_freq.items(), columns=["Symptom", "Count"])
symptom_df = symptom_df.sort_values(by="Count", ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x="Symptom", y="Count", data=symptom_df, palette="mako")
plt.title("Top 10 Most Frequent Symptoms")
plt.xlabel("Symptom")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 🎯 3️⃣ Disease–Symptom Heatmap (sample)
# Create binary mapping for presence of symptoms
sample_df = df.head(15).copy()  # take a small sample for heatmap
unique_symptoms = list(symptom_df["Symptom"])  # top 10 frequent symptoms

for symptom in unique_symptoms:
    sample_df[symptom] = sample_df["all_symptoms"].apply(lambda x: 1 if symptom in x else 0)

plt.figure(figsize=(12, 6))
sns.heatmap(sample_df[unique_symptoms], cmap="coolwarm", cbar=True)
plt.title("Disease–Symptom Heatmap (Sample of 15 Diseases)")
plt.xlabel("Symptom")
plt.ylabel("Disease Index")
plt.tight_layout()
plt.show()

print("✅ Visualization Completed Successfully!")
