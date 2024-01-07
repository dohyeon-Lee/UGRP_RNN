import numpy as np
import cv2
import pandas as pd
import argparse
from InvertedPendulum import InvertedPendulum
from scipy.integrate import solve_ivp

global_h = 0
decide_wave_w = 0.0
jump_duration = 0
f = 0.1
hz = 50
stop_count = 0

stop_time = 1
def u(t):
    global global_h, jump_duration, stop_time, stop_count
    
    if jump_duration > t: # maintain amplitude during duration & reduce jump_duration
        a = global_h + 0.5 * np.random.randn()
    else:
        if t >= stop_time: # 2.5 # probability related parameter, (The larger the value, the lower the probability)
            stop_time = t + 1 + 5*np.random.rand()
            global_h = 10 * np.random.rand() - 5
            if global_h > 0 and global_h < 0.5:
                global_h = 0.5
            if global_h < 0 and global_h > -0.5:
                global_h = -0.5
            jump_duration = t + 0.001 + 0.1*np.random.rand() # 900,1000 # duration related parameter, (The larger the value, the longer the duration)
            a = global_h + 0.5 * np.random.randn()
        else: # otherwise, just 0
            global_h = 0
            a = 0
           
    
    return a *0.2

choice = 1
zero_dist = 0
state = 0
duration = 5
# Y : [ x, x_dot, theta, theta_dot]
def func3( t, y):
    global duration, choice,state,zero_dist
    g = 9.8 
    L = 1./21.  
    k = 0.58 # coefficients c/m
    LQR_u = 0
    if(t >= duration):
        duration = 4 + t
        zero_dist = y[0]
        choice = np.random.randint(0,10) # 0,1,2,3
    if(choice >= 1):
        state = 0
        LQR_k = [10.0000, 11.3488, -4.3096, -0.2765]
        LQR_state = np.array(y).reshape([4,1])
        LQR_state[0,0] = y[0] - zero_dist
        LQR_state[2,0] = y[2] + np.pi/2
        
        LQR_k = np.array(LQR_k).reshape([1,4])
        LQR_U = -LQR_k.dot(LQR_state)
        LQR_u = LQR_U.squeeze()
    else:
        state = 1
    he = 1
    if(state == 0):
        
        x_ddot = u(t) + LQR_u
        if he == 1 and (x_ddot > 10 or x_ddot < -10):
            he = 0
            x_ddot = u(t)
        else:
            x_ddot = u(t) + LQR_u
            

    elif(state == 1):
        x_ddot = u(t)

    theta_ddot = -k*y[3]*np.cos(y[2]+np.pi/2)-(g/L)*np.sin(y[2]+np.pi/2)-(x_ddot/L)*np.cos(y[2]+np.pi/2)

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
    parser.add_argument("--animation", type=str, help="visualization, False or True",required=False)
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
        Hz = hz
    else:
        Hz = args.Hz
    if args.animation == None:
        animation = "True"
    else:
        animation = args.animation


    for j in range(0,data):
        rand_x_dot = 0 #np.random.uniform(-0.5,0.5)
        rand_theta = -np.pi/2 #np.random.uniform(0,np.pi)
        rand_theta_dot = 0 #np.random.uniform(-0.5,0.5)
        sol = solve_ivp(func3, [0, timelength], [0, rand_x_dot, rand_theta, rand_theta_dot],  t_eval=np.linspace( 0, timelength, timelength*Hz)  )

        syst = InvertedPendulum()
        if animation == "True":
            for i, t in enumerate(sol.t):
                rendered = syst.step( [sol.y[0,i], sol.y[1,i], sol.y[2,i], sol.y[3,i]], t )
                cv2.imshow( 'im', rendered )
                cv2.moveWindow( 'im', 100, 100 )
                if cv2.waitKey(30) == ord('q'):
                    break
        
        before_xdot = 0
        bbefore_xdot = 0
        dt = 1/Hz
        U = np.zeros_like(sol.y[1,0:timelength*Hz])
        for idx, xdot in enumerate(sol.y[1,0:timelength*Hz]):
            if idx > 1:
                xddot = (xdot - bbefore_xdot)/(2*dt) 
                U[idx-1] = xddot
            bbefore_xdot = before_xdot
            before_xdot = xdot       
        
        df = pd.DataFrame(U[1:timelength*Hz-1], columns=['u(t)'])

        df['theta'] =sol.y[2,1:timelength*Hz-1]
        df['theta_dot'] = sol.y[3,1:timelength*Hz-1]
        if args.mode == "train":
            df.to_csv("../mk/train/train_withcontrol3_Hz"+str(Hz)+"_"+str(j)+".csv", index = False)
        else:
            df.to_csv("../test/test_withcontrol3_Hz"+str(Hz)+"_"+"0"+".csv", index = False)

