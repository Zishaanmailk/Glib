import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([x for x in range(0,101)])
y = np.array([1 if i > 49 else 0 for i in x])
#y = y * y  # squaring step

def grd(x,y):
   m=0
   c=-1
   iter=100
   n=len(x)
   lern=0.001
   lamda = 0.9
   yp = pd.DataFrame()

   for i in range(iter):
       yperr=1/(1+np.exp((-m*x)-c))
       yp[i] = yperr
       #partial differentation of MSE
       md=-(2/n)*sum(((1/yperr)**-2)*(y-yperr)*(x*np.exp((-m*x)-c)))
       cd=-(2/n)*sum(((1/yperr)**-2)*(y-yperr)*(np.exp((-m*x)-c)))
       mse =np.mean((y - yperr)**2)
       m-=lern*md
       c-=lamda*cd
       print(f"c= {c} m= {m} itter= {i} cost= {mse}")

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
