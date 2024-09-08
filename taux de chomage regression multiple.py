
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
# Read CSV file into a DataFrame
df1 = pd.read_csv(r"C:\Users\hp\Desktop\projet econométrie\unemployment.csv").reset_index(drop=True)
df2 = pd.read_csv(r"C:\Users\hp\Desktop\projet econométrie\inflationnnnnnn.csv").reset_index(drop=True)
df3 = pd.read_csv(r"C:\Users\hp\Desktop\projet econométrie\taux d'intere.csv").reset_index(drop=True)
df4 = pd.read_csv(r"C:\Users\hp\Desktop\projet econométrie\GDP(2).csv").reset_index(drop=True)
df5 = pd.read_csv(r"C:\Users\hp\Desktop\projet econométrie\Population Growth for the United States(1).csv").reset_index(drop=True)
df=pd.merge(df1,df4,on='DATE')

df=pd.merge(df,df3,on='DATE')
df=pd.merge(df,df2,on='DATE')
df=pd.merge(df,df5,on='DATE')
#df=df[df['DATE']<'2020-01-01']
df=df[df['DATE']>'1990-01-01']
df=df.dropna()
x = df[['inflation', 'TauxInteret','GDP_PC1','SPPOPGROWUSA']]
y = (df['TauxChomage'])




corr_matrix = df[['inflation', 'TauxInteret', 'TauxChomage','GDP_PC1','SPPOPGROWUSA']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Print correlation matrix
print("Correlation Matrix")
print(corr_matrix)

# Add constant term to x for intercept in statsmodels
x = sm.add_constant(x)

# Fit the linear regression model using statsmodels
model = sm.OLS(y, x).fit()

influence = model.get_influence()

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
vif = pd.DataFrame()
vif["features"] = x.columns[1:]
vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(1, x.shape[1])]
# Print the VIF DataFrame
print(vif)

# Plot the data and the regression line
fig, ax = plt.subplots()
ax.scatter(df['inflation'], y)
ax.plot(df['inflation'], predictions, color='red')
ax.set_xlabel('inflation')
ax.set_ylabel('unemployment')
ax.set_title('Linear Regression')

fig, ax2 = plt.subplots()
ax2.scatter(df['TauxInteret'], y)
ax2.plot(df['TauxInteret'], predictions, color='red')
ax2.set_xlabel('taux intérêt')
ax2.set_ylabel('unemployment')
ax2.set_title('Linear Regression')

fig, ax2 = plt.subplots()
ax2.scatter(df['GDP_PC1'], y)
ax2.plot(df['GDP_PC1'], predictions, color='red')
ax2.set_xlabel('taux croissance')
ax2.set_ylabel('unemployment')
ax2.set_title('Linear Regression')

fig, ax2 = plt.subplots()
ax2.scatter(df['SPPOPGROWUSA'], y)
ax2.plot(df['SPPOPGROWUSA'], predictions, color='red')
ax2.set_xlabel('croissance population')
ax2.set_ylabel('unemployment')
ax2.set_title('Linear Regression')
plt.show()