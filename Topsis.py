import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def topsis(data, weights, impacts):
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(data)
    
    weighted_data = norm_data * weights
    
    ideal_best = np.max(weighted_data, axis=0) * impacts + np.min(weighted_data, axis=0) * (1 - impacts)
    ideal_worst = np.min(weighted_data, axis=0) * impacts + np.max(weighted_data, axis=0) * (1 - impacts)
    
    dist_best = np.linalg.norm(weighted_data - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted_data - ideal_worst, axis=1)
    
    topsis_score = dist_worst / (dist_best + dist_worst)
    
    return topsis_score

models = ["DialoGPT", "BlenderBot 3", "GPT-3.5 Turbo", "Mistral-7B", "T5 Dialogue"]
data = np.array([
    [7.5, 8.2, 6.9, 450, 1.2],
    [8.0, 8.5, 7.8, 520, 2.7],
    [9.5, 9.0, 9.0, 250, 5.0],
    [8.2, 8.3, 8.1, 300, 7.2],
    [7.8, 8.1, 7.0, 410, 2.5]
])

weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
impacts = np.array([1, 1, 1, -1, -1])

scores = topsis(data, weights, impacts)

results = pd.DataFrame({"Model": models, "TOPSIS Score": scores})
results = results.sort_values(by="TOPSIS Score", ascending=False).reset_index(drop=True)
print(results)

plt.figure(figsize=(10, 6))
sns.barplot(x="TOPSIS Score", y="Model", data=results, palette="viridis")
plt.xlabel("TOPSIS Score")
plt.ylabel("Model")
plt.title("TOPSIS Ranking of Conversational AI Models")
plt.show()

results.to_csv("topsis_results.csv", index=False)
