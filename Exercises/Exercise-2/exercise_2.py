# load necessary packages 
import pyblp 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

# set up paths 
home_dir = Path.home()
work_dir = (home_dir / 'Demand-Estimation')
exercises = (work_dir / 'Exercises')
data = (exercises / 'Data')
exercise_1 = Path.cwd() 

# load the data
products = pd.read_csv(f'{data}/products.csv')

##########################################################################
# compute market shares 
# we multiply city_population by 90 
products['market_size'] = products['city_population'] * 90
# now compute market share variable
products['market_share'] = products['servings_sold'] / products['market_size']
# now create outside good (1 - sum of market shares)
products['outside_good'] = 1 - products.groupby('market')['market_share'].transform('sum')

##########################################################################
# estimate pure logit model w/OLS
# create column for delta
products['logit_delta'] = np.log(products['market_share'] / products['outside_good'])
# now use statsmodels to run a simple OLS regression
X = products[['mushy', 'price_per_serving']]
X = sm.add_constant(X)
y = products['logit_delta']
ols_model = sm.OLS(y, X).fit(cov_type='HC0')
print(ols_model.summary())

##########################################################################
# run same regression using PyBLP
# rename columns 
products.rename(columns={
    'price_per_serving': 'prices',
    'product': 'product_ids',
    'market_share': 'shares',
    'market': 'market_ids',
    }, inplace=True)
# create 'demand_instruments0' column equal to prices column 
products['demand_instruments0'] = products['prices']

ols_problem = pyblp.Problem(pyblp.Formulation('1 + mushy + prices'),
                            products)
# estimate model w/ 1-step GMM instead of default 2-step GM 
ols_result = ols_problem.solve(method='1s')
print(ols_result)

##########################################################################
# now add market and product fixed effects 
# first run using dummy variables (automatically generated wtihin PyBLP) 
ols_problem = pyblp.Problem(pyblp.Formulation('1 + prices + C(market_ids) + C(product_ids)'),
                            products)
ols_result = ols_problem.solve(method='1s')
print(ols_result)

# now run using 'absorbed' fixed effects 
ols_problem = pyblp.Problem(pyblp.Formulation('1 + prices', 
                            absorb='C(market_ids) + C(product_ids)'),
                            products)
ols_result = ols_problem.solve(method='1s')
print(ols_result)

##########################################################################
# add an instrument for price
# first run first-stage regression using statsmodels 
# now run regression using statsmodels
first_stage_formula = 'prices ~ price_instrument + C(market_ids) + C(product_ids)'
first_stage = smf.ols(first_stage_formula, data=products).fit(cov_type='HC0')
print(first_stage.summary())

# set demand_instruments0 column equal to price instrument (whose relevance we just checked)
products['demand_instruments0'] = products['price_instrument']
blp_problem = pyblp.Problem(pyblp.Formulation('1 + prices',
                            absorb='C(market_ids) + C(product_ids)'),
                            products)
blp_result = blp_problem.solve(method='2s')
print(blp_result)

##########################################################################
# run a counterfactual (cutting price in half)

# create counterfactual dataframe w/just data from city-quarter 'C01Q2'
counterfactual = products[products['market_ids'] == 'C01Q2'].copy()
# now create column 'new_prices' identical to 'prices', except that the price of 
# F1B04 is halved 
counterfactual['new_prices'] = counterfactual['prices']
counterfactual.loc[counterfactual['product_ids'] == 'F1B04', 'new_prices'] = counterfactual['new_prices'] / 2
# use .compute_shares on results from last question, passing market_id = 'C01Q2'
counterfactual['new_shares'] = blp_result.compute_shares(prices=counterfactual['new_prices'], 
                                                         market_id='C01Q2')
# now compute change in market shares 
counterfactual['change_in_shares'] = (counterfactual['new_shares'] - counterfactual['shares']) / counterfactual['shares']
# plot change in market shares
counterfactual.plot(x='product_ids', y='change_in_shares', kind='bar')
plt.axhline(0, color='black', linewidth=0.5)
plt.show()

##########################################################################
# compute demand elasticities
# use .compute_elasticities on results from last question, passing market_id = 'C01Q2'
elasticities = blp_result.compute_elasticities(name='prices',
                                                  market_id='C01Q2')
# plot elasticity matrix as a heatmap
plt.imshow(elasticities, cmap='coolwarm', vmin=-1, vmax=1)
cbar = plt.colorbar()
cbar.set_label('Elasticities', fontsize=12)
cbar.set_ticks([-1, -0.5, 0, 0.5, 1]) 
cbar.ax.tick_params(labelsize=10)
plt.show()

##########################################################################
# supplemental questions -- try different standard errors 
products['clustering_ids'] = products['market_ids']
blp_problem = pyblp.Problem(pyblp.Formulation('1 + prices',
                            absorb='C(market_ids) + C(product_ids)'),
                            products)
blp_result = blp_problem.solve(method='2s', se_type='clustered')
print(blp_result)

##########################################################################
# impute marginal costs from pricing optimality
products['firm_ids'] = products['product_ids'].str.extract(r'F(\d+)B')
blp_problem = pyblp.Problem(pyblp.Formulation('1 + prices',
                            absorb='C(market_ids) + C(product_ids)'),
                            products)
blp_result = blp_problem.solve(method='2s')
# use .compute_costs on results from last question
products['cost'] = blp_result.compute_costs()
# plot scatterplot of price and marginal cost, adding y = x line 
plt.scatter(products['prices'], products['cost'])
plt.plot([0, 0.25], [0, 0.25], color='black', linewidth=0.5)
plt.xlabel('Prices')
plt.ylabel('Marginal Costs')
plt.show()