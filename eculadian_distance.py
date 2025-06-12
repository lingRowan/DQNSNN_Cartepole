import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

data = pd.read_csv('Eculadian_distance.csv')
avg_distance = data['Eucledian distance'].mean()
pearson_corr, p_pearson = pearsonr(data['Eucledian distance'], data['Performance'])
print(f"Pearson correlation: {pearson_corr:.3f}, p-value: {p_pearson:.4f}")




sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.regplot(x='Eucledian distance', y='Performance', data=data, scatter=True, ci=None, marker='o', color='blue')
plt.axvline(x=avg_distance, color='red', linestyle='--', label=f'Avg Distance: {avg_distance:.2f}')
plt.xlabel('Eucledian Distance')
plt.ylabel('Performance')
plt.title('Performance vs Eucledian Distance')
plt.show()

