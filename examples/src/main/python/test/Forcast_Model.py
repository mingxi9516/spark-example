# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:57:20 2017

@author: wyl
"""
import numpy as np
import pandas as pd
from scipy import log
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# coding=utf-8 导入样本历史数据
data = [129.08, 127.24, 136.95, 125.34, 126.86, 129.34, 131.91, 136.22, 131.56, 134.62, 144.62, 154.62,
        151.48, 126.74, 148.57, 136.6, 138.83, 136.6, 146.21, 146.09, 140.04, 142.02, 150.84, 165.27,
        155.3, 138.5, 133.27, 151.41, 155.63, 155.7, 162.98, 163.41, 157.57, 160.15, 168.13, 180.71,
        179.94, 147.29, 172.45, 169.98, 173.21, 177.43, 184.29, 183.53, 175.41, 179.64, 188.89, 197.62]
# 预测的周期个数
num = 1
# 数据周期的长度d
d = 12
validate_data = data[len(data) - num * d:len(data) + 1]
model_data = data[0:len(data) - num * d]

# 样本数据的长度，创建时间自变量序列t
t = np.arange(1, len(model_data) + 1, 1)
# 样本数据的周期个数m
m = int(len(model_data) / d)
# 将原始数据转化为m*d的矩阵形式data_matrix
data_matrix = np.zeros((m, d))

model_data = np.array(model_data)

for i in range(m):
    data_matrix[i, :] = model_data[d * i:d * i + d]
data_matrix = pd.DataFrame(data_matrix)
# 求解各个周期之间的相关系数矩阵
data_matrix.T.corr()
'''
进行季节调整
'''
# 求解各个周期数据均值mean_data_matrix
mean_data_matrix = np.mean(data_matrix, 1)

Index_sa = data_matrix.T / mean_data_matrix
# 求解季节指数Index
Index = np.mean(Index_sa, 1)
# 求解去除季节因素的数据data_matrix_sa
data_matrix_sa = np.array(data_matrix / Index)
# 将去除季节因素的数据转化为时间序列
data_sa = data_matrix_sa.reshape((m * d,))

'''
建立回归模型
'''
# 初始化拟合优度矩阵R
R = np.zeros((1, 7))
# 初始化回归系数矩阵coeff
coeff = {}
# 初始化回归模型矩阵
model = {}
'''
多项式拟合
'''


def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    results['polynomial'] = coeffs

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)  # or [p(z) for z in x]
    ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot  # 准确率
    return results


# 一次多项式拟合
z1 = polyfit(t, data_sa, 1)
R1 = z1['determination']
R[0][0] = R1

p1 = np.poly1d(z1['polynomial'])
model[1] = p1
print(p1)  # 在屏幕上打印拟合多项式
# 二次多项式拟合
z2 = polyfit(t, data_sa, 2)
R2 = z2['determination']
R[0][1] = R2

p2 = np.poly1d(z2['polynomial'])
model[2] = p2
print(p2)  # 在屏幕上打印拟合多项式


# 三次多项式拟合

def func3(x, a, b):
    y = a * log(x) + b
    return y


def polyfit3(x, y):
    results = {}
    # coeffs = numpy.polyfit(x, y, degree)
    popt, pcov = curve_fit(func3, x, y)
    results['polynomial'] = popt

    # r-squared
    yhat = func3(x, popt[0], popt[1])  # or [p(z) for z in x]
    ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results


# 对数函数拟合
z3 = polyfit3(t, data_sa)
R3 = z3['determination']
R[0][2] = R3
coeff[3] = z3['polynomial']
a = z3['polynomial'][0]
b = z3['polynomial'][1]
yvals3 = func3(t, a, b)

model[3] = func3


# 逆函数拟合
def func4(x, a, b):
    y = a / x + b
    return y


def polyfit4(x, y):
    results = {}
    # coeffs = numpy.polyfit(x, y, degree)
    popt, pcov = curve_fit(func4, x, y)
    results['polynomial'] = popt

    # r-squared
    yhat = func4(x, popt[0], popt[1])  # or [p(z) for z in x]
    ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results


z4 = polyfit4(t, data_sa)
R4 = z4['determination']
R[0][3] = R4
coeff[4] = z4['polynomial']
a = z4['polynomial'][0]
b = z4['polynomial'][1]
yvals4 = func4(t, a, b)
model[4] = func4


# 复合函数拟合
def func5(x, a, b):
    y = a * b ** x
    return y


def polyfit5(x, y):
    results = {}
    # coeffs = numpy.polyfit(x, y, degree)
    popt, pcov = curve_fit(func5, x, y)
    results['polynomial'] = popt

    # r-squared
    yhat = func5(x, popt[0], popt[1])  # or [p(z) for z in x]
    ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results


z5 = polyfit5(t, data_sa)
R5 = z5['determination']
R[0][4] = R5
coeff[5] = z5['polynomial']
a = z5['polynomial'][0]
b = z5['polynomial'][1]
yvals5 = func5(t, a, b)
model[5] = func5


# 幂函数拟合
def func6(x, a, b):
    y = a * x ** b
    return y


def polyfit6(x, y):
    results = {}
    # coeffs = numpy.polyfit(x, y, degree)
    popt, pcov = curve_fit(func6, x, y)
    results['polynomial'] = popt

    # r-squared
    yhat = func6(x, popt[0], popt[1])  # or [p(z) for z in x]
    ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results


z6 = polyfit6(t, data_sa)  # 用3次多项式拟合
R6 = z6['determination']
R[0][5] = R6
coeff[6] = z6['polynomial']
a = z6['polynomial'][0]
b = z6['polynomial'][1]
yvals6 = func6(t, a, b)
model[6] = func6
np.max(R)

loc = np.argmax(R)
if loc <= 1:
    Model = model[loc + 1]
elif loc == 2:
    def Model(x):
        a = z3['polynomial'][0]
        b = z3['polynomial'][1]
        return a * log(x) + b

elif loc == 3:
    def Model(x):
        a = z4['polynomial'][0]
        b = z4['polynomial'][1]
        return a / x + b

elif loc == 4:
    def Model(x):
        a = z5['polynomial'][0]
        b = z5['polynomial'][1]
        return a * b ** x

else:
    def Model(x):
        a = z6['polynomial'][0]
        b = z6['polynomial'][1]
        return a * x ** b
# 回归预测值
model_value = np.zeros((1, num * d))
for i in range(1, num * d + 1):
    model_value[0, i - 1] = Model(len(data) + i)

# 回归预测值添加季节指数
Index = np.array(Index)
# data_sa=data_matrix_sa.reshape((m*d,))
model_value_matrix = model_value.reshape((num, d))
forecast_value_matrix = model_value_matrix * Index.T
forecast_value = forecast_value_matrix.reshape((1, num * d))
plt.figure
plot = plt.plot(t, data_sa, '*', label='original values', )
plot = plt.plot(t, p1(t), 'r-*', label='一次多项式拟合')
plot = plt.plot(t, p2(t), 'g--', label='二次多项式拟合')
plot = plt.plot(t, yvals3, 'y', label='对数函数拟合')
plot = plt.plot(t, yvals4, 'p--', label='逆函数拟合')
plot = plt.plot(t, yvals5, 'b-*', label='复合函数拟合')
plot = plt.plot(t, yvals6, 'y-o', label='幂函数拟合')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=2)  # 指定legend的位置,读者可以自己help它的用法
plt.title('回归拟合与原始数据对比图')
plt.show()

plt.figure
plot2 = plt.plot(np.arange(len(data) - num * d + 1, len(data) + 1), validate_data, '*', label='original values', )
plot2 = plt.plot(np.arange(len(data) - num * d + 1, len(data) + 1), forecast_value.T, 'r-*', label='forecast values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=2)  # 指定legend的位置,读者可以自己help它的用法
plt.title('预测值与去除季节因素数据对比图')
plt.show()


def mape(x, y):
    abs_error = np.ones((1, len(x)))
    for i in range(len(x)):
        abs_error[:, i] = abs(x[i] - y[i]) / y[i]
    return np.mean(abs_error)


mape(forecast_value.T, validate_data)

print("本预测模型的平均绝对百分比误差(mape)为" '%.2f%%' % (mape(forecast_value.T, validate_data) * 100))
