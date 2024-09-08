
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from numpy import log
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
df1 = pd.read_csv(r"C:\Users\hp\Desktop\projet econométrie\unemployment.csv").reset_index(drop=True)
df2 = pd.read_csv(r"C:\Users\hp\Desktop\projet econométrie\inflationnnnnnn.csv").reset_index(drop=True)
df=pd.merge(df1,df2,on='DATE')

df=df[df['DATE']<'2020-01-01']
df=df[df['DATE']>'2015-01-01']


x = log(df[['inflation' ]])
y = log(df['TauxChomage'])

corr_matrix = df[['inflation', 'TauxChomage']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


print("Correlation Matrix")
print(corr_matrix)

# Add constant term to x for intercept in statsmodels
x = sm.add_constant(x)

# Fit the linear regression model using statsmodels
model = sm.OLS(y, x).fit()

influence = model.get_influence()

# Calculate Cook's distance

# Print the model summary
print(model.summary())

# Extract the coefficients and intercept
coefficients = model.params[1]
intercept = model.params[0]
#test de heteroscedasticite
Bp=het_breuschpagan(model.resid,model.model.exog)
print ("test Breuschpagan p value",Bp[1])
#test de white
white_test=het_white(model.resid,model.model.exog)
print ("test white p value",white_test[1])
bg_test = acorr_breusch_godfrey(model, nlags=1)
print("Farrar-Glauber test p-value:", bg_test[3])

# Make predictions
predictions = model.predict(x)
mae = mean_absolute_error(y, predictions)
print(mae)


fig, ax = plt.subplots()
ax.scatter(df['inflation'], y)
ax.plot(df['inflation'], predictions, color='red')
ax.set_xlabel('inflation')
ax.set_ylabel('TauxChomage')
ax.set_title('Linear Regression')

plt.show()