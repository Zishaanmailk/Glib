import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("data2")
x=np.array(data['x'])
y=np.array(data['y'])
y=y**(2)

def grd(x,y):
    m = c = 0
    p = 1
    iter = 100
    n = len(x)
    lern = 0.0000001
    yp = pd.DataFrame()

    for i in range(iter):
       yperr = (m * (x ** p)) + c
       yp[i] = yperr
       # partial differentation of MSE
       mse = np.mean((y - yperr) ** 2)
       md = -(2 / n) * sum(x * (y - yperr))
       cd = -(2 / n) * sum(y - yperr)
       pdd = -(2 / n) * sum(m * p * x * (y - yperr))
       m -= lern * md
       c -= lern * cd
       p -= lern * pdd

       print(f"c= {c} m= {m} p= {p} itter= {i} cost= {mse}")

    def plot():
        fig, (right, left) = plt.subplots(1, 2, figsize=(13, 6.5))
        right.scatter(x, y, c="blue", label="actual points")
        right.set_title("Gradient Descent trace")
        right.set_xlabel("Feature")
        right.set_ylabel("Dependent value")
        plt.tight_layout()

        for i in range(iter):
            left.cla()
            right.plot(x, yp[i], c="red")
            left.scatter(x, y, c="blue", label="actual points")
            left.plot(x, yp[i], c="red", label="predicted line")
            left.set_title("Gradient Descent live")
            left.set_xlabel("Feature")
            plt.pause(0.01)
            left.legend()

        right.legend()
        plt.show()

    plot()


grd(x,y)