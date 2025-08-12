import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([x for x in range(0,101)])
y = np.array([1 if i > 49 else 0 for i in x])
#y = y * y  # squaring step

def LogesticReg(x,y,iter=150,lern=0.001,lamda = 0.95,c=-1, m=0,plot='False'):

   import matplotlib.pyplot as plt
   import numpy as np
   n = len(x)

   if plot != 'False':
       fig, (right, left) = plt.subplots(1, 2, figsize=(13, 6.5))
       right.scatter(x, y, c="blue", label="actual points")
       right.set_title("Gradient Descent trace")
       right.set_xlabel("Feature")
       right.set_ylabel("Dependent value")
       plt.tight_layout()

   for i in range(iter):
       yperr=1/(1+np.exp((-m*x)-c))
       #partial differentation of MSE
       md=-(2/n)*sum(((1/yperr)**-2)*(y-yperr)*(x*np.exp((-m*x)-c)))
       cd=-(2/n)*sum(((1/yperr)**-2)*(y-yperr)*(np.exp((-m*x)-c)))
       mse =np.mean((y - yperr)**2)
       m-=lern*md
       c-=lamda*cd
       print(f"c= {c} m= {m} itter= {i} cost= {mse}")

       if plot != 'False':
           left.cla()
           right.plot(x, yperr, c="red")
           left.scatter(x, y, c="blue", label="actual points")
           left.plot(x, yperr, c="red", label="predicted line")
           left.set_title("Gradient Descent live")
           left.set_xlabel("Feature")
           plt.pause(0.01)
           left.legend()

   if plot != 'False':
       right.legend()
       plt.show()

LogesticReg(x, y, plot='True')