import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df_2_blocks = pd.read_csv('Analysis_2_blocks.csv', header = None)
df_2_blocks.columns = ['Number of blocks', 'Blocks position', 'Configuration', 'Episode', 'Total number of rewards', 'Total steps', 'Average reward per action', 'Difference between steps']

#print(df_2_blocks.head(20))
df_4_blocks = pd.read_csv('Analysis_4_blocks.csv', header = None)
df_4_blocks.columns = ['Number of blocks', 'Blocks position', 'Configuration', 'Episode', 'Total number of rewards', 'Total steps', 'Average reward per action', 'Difference between steps']

df_6_blocks = pd.read_csv('Analysis_6_blocks.csv', header = None)
df_6_blocks.columns = ['Number of blocks', 'Blocks position', 'Configuration', 'Episode', 'Total number of rewards', 'Total steps', 'Average reward per action', 'Difference between steps']


df_10_blocks = pd.read_csv('Analysis_10_blocks.csv', header = None)
df_10_blocks.columns = ['Number of blocks', 'Blocks position', 'Configuration', 'Episode', 'Total number of rewards', 'Total steps', 'Average reward per action', 'Difference between steps']


df_combined = pd.concat([df_2_blocks, df_4_blocks, df_6_blocks, df_10_blocks], axis=0, ignore_index=True)
#print(df_combined.head(20))

summary = df_combined.groupby('Number of blocks').agg({
    'Average reward per action': 'mean',
    'Total steps': 'mean',
    'Difference between steps': 'mean',
    'Total number of rewards': 'mean'
}).reset_index()

#print(summary)


group_stats = df_combined.groupby('Number of blocks')['Difference between steps'].agg(['mean', 'std']).reset_index()

# Plot
'''plt.figure(figsize=(8, 5))
plt.errorbar(group_stats['Number of blocks'], group_stats['mean'], yerr=group_stats['std'],
             fmt='-o', capsize=5, label='Mean ± SD')

plt.title('Difference Between Steps vs Number of Blocks')
plt.xlabel('Number of Blocks')
plt.ylabel('Difference Between Steps')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()'''


plt.figure(figsize=(8, 5))
plt.plot(group_stats['Number of blocks'], group_stats['mean'], '-o', label='Mean')
plt.fill_between(group_stats['Number of blocks'], group_stats['mean'] - group_stats['std'], group_stats['mean'] + group_stats['std'], alpha=0.2, label='±1 SD')

plt.title('Mean and Standard Deviation (Shaded) for Block Configurations')
plt.xlabel('Number of Blocks')
plt.ylabel('Metric Value')
plt.xticks(group_stats['Number of blocks'])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#sns.lineplot(data=df_combined, x='Number of blocks', y='Average reward per action')
#plt.title('Average Reward vs Number of Blocks')
#plt.show()


'''groups = [group['Average reward per action'].values for name, group in df_combined.groupby('Number of blocks')]
f_stat, p_val = f_oneway(*groups)

print("ANOVA F-statistic:", f_stat)
print("P-value:", p_val)

# Tukey HSD post-hoc test
tukey = pairwise_tukeyhsd(endog=df_combined['Average reward per action'],
                          groups=df_combined['Number of blocks'],
                          alpha=0.05)

print("\nTukey HSD Test Results:")
print(tukey)

# Optional: visualize Tukey results
tukey.plot_simultaneous()
plt.title('Tukey HSD: Average Reward per Action by Number of Blocks')
plt.xlabel('Mean Difference')
plt.show()'''