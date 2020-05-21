import requests
import pandas
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel as W, DotProduct as DP
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os

## Liste des départements centre val de loire
##lstdeps = ['18', '28', '36', '37', '41','45']
lstdeps = ['18']
starting_date = pandas.Timestamp('2020-03-18').date()
## Données hospitalières
## <URL: https://www.data.gouv.fr/fr/datasets/donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/ >
fn =  'donnees-hospitalieres-nouveaux-covid19-2020-04-30-19h00.csv'
fd = open(fn, 'r')

df = pandas.read_csv(fd, sep=";", parse_dates=['jour'])
tmp_deps = [np.array(df[df['dep'] == dep]['incid_hosp']) for dep in lstdeps]
tmp_new_cases = sum(tmp_deps)
tmp_total_cases = tmp_new_cases.cumsum()
tmp_days = np.array((df[df['dep'] == '18']['jour'].dt.date - starting_date).dt.days)
data_hosp = pandas.DataFrame({'days': tmp_days, 'total_cases': tmp_total_cases, 'new_cases_variation': tmp_new_cases})
data_hosp = data_hosp.set_index('days')
data_hosp = data_hosp[data_hosp['total_cases']>=1]

# kernel = C() * RBF() + C() * DP() + W()
kernel = C() * RBF() + W()
gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=200, normalize_y = True)
x = np.array(data_hosp.index).reshape(-1,1)
y = np.log(np.array(data_hosp['total_cases']))

gp.fit(x, y)
pred = gp.predict(x)
data_hosp['smoothed_total_cases'] = np.exp(pred)
data_hosp['smoothed_new_cases_variation'] = np.diff(np.insert(np.exp(pred), 0, 0))
data_hosp = data_hosp.iloc[1:]


plt.scatter(data_hosp.index, data_hosp['new_cases_variation']/data_hosp['total_cases'], color='red')
plt.plot(data_hosp.index, data_hosp['smoothed_new_cases_variation']/data_hosp['smoothed_total_cases'], color='blue')

plt.suptitle('COVID-19/ Région Centre Val de Loire', fontsize=14, fontweight='bold')
plt.xlabel('days')
plt.ylabel('coef logarithmique')


plt.show()
