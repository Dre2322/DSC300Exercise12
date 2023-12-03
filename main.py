import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#
# 1. Root mean square error
def rmse(sales_price, sales_pred):
  return np.sqrt(np.mean((np.array(sales_price) - np.array(sales_pred))** 2))

# Open housing data
data = pd.read_csv('housing_data.csv')
# Pull sales price from data
sale_price = pd.DataFrame(data, columns=['sale_price'])
# Pull predicted sales price from data
sale_pred = pd.DataFrame(data, columns=['sale_price_pred'])
# Perform RMSE calculation
rmse_error = rmse(sale_price,sale_pred)

# Print results
print(f'RMSE: {rmse_error:.4f}') 

#
# 2. Mean absolute error
def mae (sales_price, sales_pred):
  return np.mean(np.abs(np.array(sales_price) - np.array(sales_pred)))
  
# Open housing data
data = pd.read_csv('housing_data.csv')
# Pull sales price from data
sale_price = pd.DataFrame(data, columns=['sale_price'])
# Pull predicted sales price from data
sale_pred = pd.DataFrame(data, columns=['sale_price_pred'])
# Perform MAE calculation
mean_absolute_error = mae(sale_price, sale_pred)

# Print results
print(f'MAE: {mean_absolute_error:.4f}')

#
# 3. Check accuracy of predictions
def accur(act, pred):
  accuracy =len([act.iloc[i] for i in range (0,len(act)) if act.iloc[i].values == pred.iloc[i].values])/len(act)
  print(f'Accuracy: {accuracy}')

# Mushroom data
shroom = pd.read_csv('mushroom_data.csv')
a = pd.DataFrame(shroom, columns=['actual'])
p = pd.DataFrame(shroom, columns=['predicted'])

accur(a,p)

#
# 4. Use matplotlib to plot this function
def f(x): 
  return 0.005*x**6 - 0.27*x**5 + 5.998*x**4 - 69.919*x**3 + 449.17*x**2-1499.7*x+2028

# Create x-values
x = np.linspace(-40, 50, 100)
# Create y-values
y = np.array([f(x) for x in x])
# Find the indez of the value that minimizes the error
min_index = np.argmin(y)

# Print the value that minimizes the error
print(f'The value that minimizes the error is: {x[min_index]}')

# The minimum x and y values
min_x = np.array(x[min_index])
min_y = np.min(y)

# Print the min
print(f'The min x and y values are: ({min_x},{min_y})')

# Plot the function
plt.plot(x, y, label='y')
# Plot the minimum x and y values
plt.plot(min_x, min_y, 'ro', label='min')
# Set title and labels for the axes
plt.title('Plot of $f(x)$')
plt.xlabel('x')
plt.ylabel('y')

#Display the plot
plt.show()