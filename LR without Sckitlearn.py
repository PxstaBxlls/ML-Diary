import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Salary_Data.csv')

data = data.dropna(subset=['Years of Experience', 'Salary'])
x = data.iloc[:, -2].values
y = data.iloc[:, -1].values


def loss_ftn(m,b,x,y):
    total_cost = 0
    n = len(x)
    for i in range(n):
        total_cost += (y[i] - (m*x[i] + b))**2
    return total_cost/float(n)


def gradient_descent(x,y,m,b,learning_rate):
    n = len(x)
    m_gradient = 0
    b_gradient = 0
    for i in range(n):
        m_gradient += (-2/n) * x[i] * (y[i] - (m*x[i] + b))
        b_gradient += (-2/n) * (y[i] - (m*x[i] + b))
    m -= learning_rate * m_gradient
    b -= learning_rate * b_gradient
    return m,b

m = 0
b = 0
learning_rate = 0.0001
epochs = 5000
n = len(x)
x = (x - x.mean()) / x.std()
y = (y - y.mean()) / y.std()

cost = []
for i in range(epochs):
    m,b = gradient_descent(x,y,m,b,learning_rate)
    cost.append(loss_ftn(m,b,x,y))
    
print(m,b)


plt.scatter(x, (m*x) + b, color='red')
plt.scatter(x, y, color='blue')
plt.plot(x, (m*x) + b, color='green')
plt.title('Linear Regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
