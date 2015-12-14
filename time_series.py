import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sag

df = pd.read_csv('LoanStats3a.csv', header=0, low_memory=False)
print(df.head())

# converts string to datetime object in pandas:
df['issue_d_format'] = pd.to_datetime(df['issue_d']) 
dfts = df.set_index('issue_d_format') 
year_month_summary = dfts.groupby(lambda x : x.year * 100 + x.month).count()
loan_count_summary = year_month_summary['issue_d']

print(loan_count_summary)
#we're left with a data table of year+month x # of loans issued
plt.xlabel('2015 Issue Date (Month)')
plt.ylabel('Loans Issued')
loan_count_summary.plot()
plt.show()

#ACF
sag.plot_acf(loan_count_summary)
plt.show()

#PACF
sag.plot_pacf(loan_count_summary)
plt.show()

print ("There are autocorrelated structures in the data, specfically there seems to be Seasonality and a need to add an Auto Regressive term.")

#output
#201501    2616
#201502    2588
#201503    3002
#201504    3067
#201505    3167
#201506    3494
#201507    3694
#201508    3729
#201509    3873
#201510    4181
#201511    4439
#201512    4685
#Name: issue_d, dtype: int64