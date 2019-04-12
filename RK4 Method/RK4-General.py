#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 21:59:24 2019

@author: aliheydari
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:21:44 2019

@author: aliheydari
"""

import numpy as np
from numpy import linalg as LA
from scipy import sparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import time

import warnings
warnings.filterwarnings("ignore")

"""
EXAMPLE OF MULTI-D RK METHOD 

"""

# One-Dimensional RK 

def RK4_1D (f,t,y0):
    
    N = len(t);
    y = np.zeros((N, len(y0)))
    y[0] = y0
    
    for n in range (0,N-1):
        
        dt = t[n+1] - t[n];
        Y1 = y[n];
        Y2 = y[n] + (dt/2) * f(t[n] + (dt/2),Y1);
        Y3 = y[n] + (dt/2) * f(t[n] + (dt/2),Y2);
        Y4 = y[n] + (dt) * f(t[n] ,Y3);
        
        y[n+1] = y[n] + dt * (((1/6) * f(t[n],Y1))\
         + ((1/3) * f(t[n] + (dt/2),Y2)) + \
            ((1/3) * f(t[n] + (dt/2),Y3)) + \
            ((1/6) * f(t[n] + dt,Y4)));
         
        
    return y
   
    

# Multi-Dimensional RK 

def RK4_nD (f,t,y0):
    
    # we will use this later fir getting dt    
    t_min = np.min(t);
    t_max = np.max(t);
    

    N = len(t);
    y = np.zeros((len(y0), N), dtype = "float");
    
    # initialize the matrix of solutions for the use in Y1
    y[:,0] = y0;   
    
    for n in range (0,N-1):
        
        # getting dt
        dt = (t_max - t_min) / N;
        
        # setting up the Yi's for RK
        Y1 = y[:,n];

        Y2 = np.array(f(t[n] + (dt/2),Y1))
        Y2 = y[:,n] + (dt/2) * Y2;
             
        Y3 = np.array(f(t[n]+ (dt/2),Y2));
        Y3 = y[:,n] + (dt/2) * Y3;
        
        Y4 = np.array(f(t[n],Y3)); 
        Y4 = y[:,n] + (dt) * Y4;

        K1 = np.array(f(t[n],Y1));
        K2 = np.array(f(t[n] + dt/2,Y2))
        K3 = np.array(f(t[n] + dt/2, Y3))
        K4 = np.array(f(t[n],Y4));
              
        y[:,n+1] = y[:,n] + dt * (K1/6 + K2/3 + K3/3 + K4/6)

                
    return y;


## the "f" part of RK as a vector function
    
def f(t,y,*argv):

    f = [];
    # this part needs to be modified depending on the problem
    
    f1 = lambda t,y,*argv: y[0] + (2 * y[1]) + 1
    f2 = lambda t,y,*argv: t - y[0] + y[1] ;
    
    f.append(f1(t,y)); #f1(t,y);
    f.append(f2(t,y)); #f2(t,y);
    
    

    
    return f


#### main ####

"""
This is an example that I got from MATLAB's website in order to compare the 
accuracy of my Multi-D RK4 method 

Source : https://www.mathworks.com/help/symbolic/solve-a-system-of-differential-equations.html 


EXAMPLE vvv

"""

t = np.linspace(0, 10, 10000);

rl_x_sol = (2* t/3 + (17*np.exp(t)*np.cos(2**(1/2)*t))/9 - (7*2**(1/2)*np.exp(t)\
         *np.sin(2**(1/2)*t))/9 + 1/9 );
         
rl_y_sol = (- t/3 - (7*np.exp(t)*np.cos(2**(1/2)*t))/9 - (17*2**(1/2)*np.exp(t)\
                 *np.sin(2**(1/2)*t))/18 - 2/9 );

y0 = np.array([rl_x_sol[0],rl_y_sol[0]]);



y_sol = RK4_nD(f,t,y0)



plt.plot(t,y_sol[0,:],'-b',label = "x_approx");
plt.plot(t,y_sol[1,:],'-r',label = "y_approx");

plt.plot(t,rl_x_sol,'-m',label = "x_exact");
plt.plot(t,rl_y_sol,'-y',label = "y_exact");


plt.legend(loc = "upper left");

plt.show()

N = len(t);

#error_x = (y_sol[0,N-1] - rl_x_sol[N-1]);
#error_y = (y_sol[1,N-1] - rl_y_sol[N-1]);








#### Example 2 (with some stability issues)####
"""
##### CHANGE THIS IN f ######

    f1 = lambda t,y: -1 * y[0] + np.sqrt(y[1]) - y[2] * np.exp(2 * t);
    f2 = lambda t,y: -2 * y[0]**2;
    f3 = lambda t,y,*argv: -3 * y[0] * y[1];
    
    f.append(f1(t,y));
    f.append(f2(t,y));
    f.append(f3(t,y))    

"""


""" 

for initial condition: 
    y0 = np.array([1,1,1]);


everything else stays exactly the same 


to plot the solution : 
plt.plot(t,np.exp(-1*t),'-g',label = "e^-t_exact");
plt.plot(t,np.exp(-2*t),'-y',label = "e^-2t_exact");
plt.plot(t,np.exp(-3*t),'-m',label = "e^-3t_exact");


plt.plot(t,y_sol[0,:],'-b',label = "e^-t_approx");
plt.plot(t,y_sol[1,:],'-r',label = "e^-2t_approx");
plt.plot(t,y_sol[1,:],'-k',label = "e^-3t_approx");


""" 







