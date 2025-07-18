import numpy as np
import math

x = np.linspace(-math.pi,math.pi,2000)
y = np.sin(x)

a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6 # 梯度法里的步长
for t in range(1000):
    y_pred = a + b*x + c*x**2 + d*x**3  #  把所有“预测-真实”误差的平方累加起来，得到当前整体误差。

    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t,loss) # 每100次训练就打印一次当前损失值

    # 手工推导并计算每个参数的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # 用“梯度下降法”根据每个参数的梯度，更新参数值
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')


