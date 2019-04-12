#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 19:25:06 2019

@author: aliheydari
"""

import numpy as np
import time
from numpy import linalg as LA


import warnings
warnings.filterwarnings("ignore")

###Project1: Solving Dirchlet Boundary Condition on a Unit Squere (5 pt Lap) ### 
        ##MATRIX FREE IMPLEMENTATION FOR AU is stored iteratively ###
         #MATRICES ARE STORED ONLY FOR THE HEAT MAP TO TEST ACCURACY#
         
         
# set up the grid

h = 0.025; #also 0.1, 0.025

x_Mgrid = int(1 / h)
y_Lgrid = int(1 / h)

print(x_Mgrid)
# the interior points
inter_x = x_Mgrid - 1
inter_y = y_Lgrid - 1
#a = 1.0e-2;

# Creating the mesh so we could find the boundary values
x = np.linspace(0, 1, x_Mgrid + 1)
y = np.linspace(0, 1, y_Lgrid + 1)

##### MATRICES ARE STORED ONLY FOR THE HEAT MAP TO TEST ACCURACY #####

X, Y = np.meshgrid(x, y)

##### MATRICES ARE STORED ONLY FOR THE HEAT MAP TO TEST ACCURACY #####
# initialization of
    # the temporary vector that holds the finite difference values of the grid
V = np.zeros_like(X)
# the actual solution just so that we could check our work
real_sol = np.zeros_like(X)
# the RHS of the poisson equation (just initialized here)
f = np.zeros_like(X)

Q = np.zeros_like(X)

#RHS = np.zeros_like(((int(1/h)*int(1/h)),1),dtype = float);

# Dirchlet boundary conditions : Actual solution is cos(x+2y)

##### MATRICES ARE STORED ONLY FOR THE HEAT MAP TO TEST ACCURACY #####

# right
V[:,-1] = np.cos(X[:,-1] + 2 * Y[:,-1])
# left
V[:,0] = np.cos(X[:,0] + 2 * Y[:,0])
# top of the square
V[-1,:] = np.cos(X[-1,:] + 2 * Y[-1,:])
# Bottom 
V[0,:] = np.cos(X[0,:] + 2 * Y[0,:])



real_sol[:,-1] = np.cos(X[:,-1] + 2 * Y[:,-1])
# left
real_sol[:,0] = np.cos(X[:,0] + 2 * Y[:,0])
# top of the square
real_sol[-1,:] = np.cos(X[-1,:] + 2 * Y[-1,:])
# Bottom 
real_sol[0,:] = np.cos(X[0,:] + 2 * Y[0,:])

RHS = np.zeros(((inter_x ) * (inter_x),1),dtype = float)


# given tolerance
tol=1.0e-7


# here we we vary values of omega to find the optimal value

#for q in range(0,5):
#    # to increment by 0.1

omega = 1.8 #+ q/10;
counter = 0
stop_res = 1.0


for i in range(0,inter_x + 2):
    for j in range(0,inter_x + 2):
        
        f[i,j] = -5 * np.cos(X[i,j] + 2 * Y[i,j]);
        real_sol[i,j] = np.cos(X[i,j] + 2 * Y[i,j])  





# to get the time elapsed


# here is the SOR method:

##### MATRICES ARE STORED ONLY FOR THE HEAT MAP TO TEST ACCURACY #####

t_start = time.clock()

while (stop_res > tol):
    dVmax = 0.0
    bounter = 0;
    
    for i in range(1,inter_y + 1):
        
        for j in range(1, inter_x + 1):
            # real solution at each point
            real_sol[i,j] = np.cos(X[i,j] + 2 * Y[i,j]) 
                
            RHS[bounter] = h**2 * (f[i,j]);

            # the RHS
            f[i,j] = -5 * np.cos(X[i,j] + 2 * Y[i,j]);
            # residual 
            resid = (V[i,j-1] + V[i-1,j] + V[i,j+1] + V[i+1,j] - 4.0 * V[i,j]) \
            - h**2 * f[i,j]
            dV = 0.25 * omega * resid
            V[i,j]+= dV
            dVmax = np.max([np.abs(dV),dVmax])
#            print(bounter)
            bounter += 1;
            
    # calculating the total residual    
    stop_res = dVmax/np.max(np.abs(V))
    counter += 1



## Write RHS into a txt file
#outfile = open("V01.txt", "w+")
#outfile.write(np.array2string(V, precision=6, separator=',',suppress_small=False));
#outfile.close()



t_end = time.clock()

x_approx = np.reshape(V,(x_Mgrid + 1) **2);
x_true = np.reshape(real_sol,(x_Mgrid + 1) **2);


err = LA.norm(x_approx - x_true,np.inf);
print(err)


print ("SOR with Omega = {0} <----> CPU time = \t{1:0.2f} \t <----> iterations = {2}"\
       .format(omega,t_end - t_start,counter))


# error from the h = 0.1 is 0.00026760649441859297

order = np.log(err/0.00026760649441859297)/ np.log(h / 0.1);
print("order of convergence is : {}".format(order));




