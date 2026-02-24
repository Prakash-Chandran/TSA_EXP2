# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('NSE-TATAGLOBAL11.csv',parse_dates=['Date'],index_col='Date')
data.head()

resampled_data.index = resampled_data.index.year

resampled_data.reset_index(inplace=True)
resampled_data.rename(columns={'Date':'Year'}, inplace=True)

resampled_data.head()

Years = resampled_data['Year'].tolist()
Turnover = resampled_data['Turnover (Lacs)'].tolist()

# linear trend estimation
X = [i - Years[len(Years) // 2] for i in Years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, Turnover)]

n = len(Years)
b = (n * sum(xy) - sum(Turnover) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(Turnover) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]

# Polynomial Trend Estimation (Degree 2)

x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, Turnover)]

coeff = [[len(X), sum(X), sum(x2)],
[sum(X), sum(x2), sum(x3)],
[sum(x2), sum(x3), sum(x4)]]
Y = [sum(Turnover), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)

solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]

# Visualising results
print(f"Linear Trend: y={a:.2f} + {b:.2f}x")
print(f"\nPolynomial Trend: y={a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

resampled_data['Linear Trend'] = linear_trend
resampled_data['Polynomial Trend'] = poly_trend

resampled_data.set_index('Year',inplace=True)

resampled_data['Turnover (Lacs)'].plot(kind='line',color='blue',marker='o') #alpha=0.3 makes 
resampled_data['Linear Trend'].plot(kind='line',color='black',linestyle='--')

resampled_data['Turnover (Lacs)'].plot(kind='line',color='blue',marker='o')
resampled_data['Polynomial Trend'].plot(kind='line',color='black',marker='o')

### OUTPUT
A - LINEAR TREND ESTIMATION
<img width="614" height="485" alt="image" src="https://github.com/user-attachments/assets/82897221-35d3-44a2-b5ae-c33ab8a9872b" />


B- POLYNOMIAL TREND ESTIMATION
<img width="599" height="479" alt="image" src="https://github.com/user-attachments/assets/1344f0e6-c578-4a48-b9bc-393aff460ffe" />


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
