import numpy as np
import cv2
import pandas as pd
import argparse
from InvertedPendulum import InvertedPendulum
from scipy.integrate import solve_ivp

U = []
#가속도 u(t)
global_h = 0
def u( t ,j):
    global global_h
    time = np.random.randn()
    if (time>0.5):
        global_h = 25*np.random.rand()-12.5
    
    a = global_h+np.random.randn()

    return a


# Y : [ x, x_dot, theta, theta_dot]
def func3( t, y):
    g = 9.8 
    L = 1.5 
    m = 1.0
    M = 2.0 
    k = 0.8 # coefficients c/m
    
    x_ddot = u(t,j)
    
    U.append(x_ddot)
    theta_ddot = -k*y[3]*np.cos(y[2]+np.pi/2)-(g/L)*np.sin(y[2]+np.pi/2)+(x_ddot/L)*np.cos(y[2]+np.pi/2)

# xdot, xddot, thetadot, theta_ddot
    return [ y[1], x_ddot, y[3], theta_ddot ]


# Both cart and the pendulum can move.
if __name__=="__main__":

    parser = argparse.ArgumentParser(
        prog="water-pendulum simulation",
        description="make training dataset",
        epilog="2023_UGRP"
    )

    parser.add_argument("--mode", type=str, help="train or test",required=True)
    parser.add_argument("--num", type=int, help="num of dataset",required=False)
    parser.add_argument("--timelength", type=int, help="time length",required=False)
    parser.add_argument("--Hz", type=int, help="data hz",required=False)
    args = parser.parse_args()

    if args.num == None:
        data = 1
    else:
        data = args.num
    if args.timelength == None:
        timelength = 600
    else:
        timelength = args.timelength
    if args.Hz == None:
        Hz = 50
    else:
        Hz = args.Hz

    for j in range(0,data):
        rand_x_dot = 0 #np.random.uniform(-0.5,0.5)
        rand_theta = -np.pi/2 #np.random.uniform(0,np.pi)
        rand_theta_dot = np.random.uniform(-0.5,0.5)
        sol = solve_ivp(func3, [0, timelength], [ 0, rand_x_dot, rand_theta, rand_theta_dot],  t_eval=np.linspace( 0, timelength, timelength*Hz)  )

        syst = InvertedPendulum()

        for i, t in enumerate(sol.t):
            rendered = syst.step( [sol.y[0,i], sol.y[1,i], sol.y[2,i], sol.y[3,i] ], t )
            cv2.imshow( 'im', rendered )
            cv2.moveWindow( 'im', 100, 100 )

            if cv2.waitKey(30) == ord('q'):
                break


        df = pd.DataFrame(U[0:timelength*Hz], columns=['u(t)'])
        U = []
        df['theta'] =sol.y[2,0:timelength*Hz]
        df['theta_dot'] = sol.y[3,0:timelength*Hz]
        if args.mode == "train":
            df.to_csv("../train/train"+str(j)+".csv", index = False)
        else:
            df.to_csv("../test/test"+str(j)+".csv", index = False)

