import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data2")
x = np.array(data['x'])
y = np.array(data['y'])
#y = y * y  # squaring step

def gd(x,y):
   mc=c=0
   iter=100
   n=len(x)
   lern=0.00001
   lamda = 0.02
   yp=pd.DataFrame()

   for i in range(iter):
       yperr=(mc*x)+c
       yp[i]=yperr
       #partial differentation of MSE
       md=-(2/n)*sum(x*(y-yperr))+ (lamda * abs(mc))
       cd=-(2/n)*sum(y-yperr)
       mse =np.mean((y - yperr)**2)
       mc-=lern*md
       c-=lern*cd
       print(f"c= {c} m= {mc} itter= {i} cost= {mse}")

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

gd(x,y)