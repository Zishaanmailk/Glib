import pandas as pd
import numpy as np

data = pd.read_csv("data2")
x = np.array(data['x'])
y = np.array(data['y'])
#y = y * y  # squaring step

def Ridge(x,y,iter=100,lern=0.00001,lamda = 0.02,c=0, m=0,plot='False'):

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
       yperr=(m*x)+c
       #partial differentation of MSE
       md=-(2/n)*sum(x*(y-yperr))+ (lamda * m * m)
       cd=-(2/n)*sum(y-yperr)
       mse =np.mean((y - yperr)**2)
       m-=lern*md
       c-=lern*cd
       print(f"c= {c} m= {m} itter= {i} cost= {mse}")

       if plot!='False':
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


Ridge(x,y,plot='True')



