import pandas as pd
import matplotlib.pyplot as plt



data = pd.read_csv('conv.csv')
print(data.head(20))


numeric_data = data.iloc[:, 1:]
print(numeric_data)


#subset = numeric_data.iloc[1:4]
#print(subset)

data_mean = numeric_data.mean(axis=0).to_numpy()
data_std = numeric_data.std(axis=0).to_numpy()
blocks = [2,4,6,10]

plt.figure(figsize=(8, 5))
plt.plot(blocks, data_mean, '-o', label='Mean (configs)')
plt.fill_between(blocks, data_std - data_std, data_mean + data_std, alpha=0.2, label='±1 SD')

plt.title('Mean and Standard Deviation for Configurations (Blocks 2, 4, 6, 10)')
plt.xlabel('Number of Blocks')
plt.ylabel('Metric Value')
plt.xticks(blocks)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()