import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df_2_blocks = pd.read_csv('Analysis_2_blocks.csv', header = None)
df_2_blocks.columns = ['Number of blocks', 'Blocks position', 'Configuration', 'Episode', 'Total number of rewards', 'Total steps', 'Average reward per action', 'Difference between steps']

df_10_blocks = pd.read_csv('Analysis_10_blocks.csv', header = None)
df_10_blocks.columns = ['Number of blocks', 'Blocks position', 'Configuration', 'Episode', 'Total number of rewards', 'Total steps', 'Average reward per action', 'Difference between steps']




group_stats = df_2_blocks.groupby('Blocks position')['Difference between steps'].agg(['mean', 'std']).reset_index()




'''plt.figure(figsize=(8, 5))
plt.plot(group_stats['Blocks position'], group_stats['mean'], '-o', label='Mean')
plt.fill_between(group_stats['Blocks position'], group_stats['mean'] - group_stats['std'], group_stats['mean'] + group_stats['std'], alpha=0.2, label='±1 SD')

plt.title('Mean and Standard Deviation (Shaded) for Block Configurations')
plt.xlabel('Blocks position')
plt.ylabel('Metric Value')
plt.xticks(group_stats['Blocks position'])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#sns.lineplot(data=df_combined, x='Number of blocks', y='Average reward per action')
#plt.title('Average Reward vs Number of Blocks')
#plt.show()'''


groups = [group['Difference between steps'].values for name, group in df_2_blocks.groupby('Blocks position')]
f_stat, p_val = f_oneway(*groups)

print("ANOVA F-statistic:", f_stat)
print("P-value:", p_val)

# Tukey HSD post-hoc test
tukey = pairwise_tukeyhsd(endog=df_2_blocks['Difference between steps'],
                          groups=df_2_blocks['Blocks position'],
                          alpha=0.05)

print("\nTukey HSD Test Results:")
print(tukey)

# Optional: visualize Tukey results
tukey.plot_simultaneous()

plt.title('Tukey HSD: Average Reward per Action by Number of Blocks')
plt.xlabel('Mean Difference')
plt.show()