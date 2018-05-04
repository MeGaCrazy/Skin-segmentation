import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../output.csv', index_col=['id'])
sns.distplot(df['class'])
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
