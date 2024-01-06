import numpy as np
import cv2
import pandas as pd
import argparse
from InvertedPendulum import InvertedPendulum
from scipy.integrate import solve_ivp


#가속도 u(t)
# global_h = 0
# def u(t):
#     global global_h
#     time = np.random.randn()
#     if (time > 1.5):
#         global_h = 10*np.random.rand()-5
    
#     a = global_h + 0.5*np.random.randn()

#     # if t>0 and t < 0.5:
#     #     return 15
#     # elif t >= 0.5  and t < 1:
#     #     return -35
#     # else:
#     #     return -35

#     return a*0.5

# global_h = 0
# jump_duration = 0

# def u(t):
#     global global_h, jump_duration
#     time = np.random.randn()

#     if jump_duration > 0: # maintain amplitude during duration & reduce jump_duration
#         jump_duration = jump_duration - 1
#         a = global_h + 0.5 * np.random.randn()
#     else:
#         if time > 2.5: # 1.5 # probability related parameter, (The larger the value, the lower the probability)
#             global_h = 10 * np.random.rand() - 5
#             jump_duration = np.random.randint(900, 1000) # 100,200 # duration related parameter, (The larger the value, the longer the duration)
#             a = global_h + 0.5 * np.random.randn()
#         else: # otherwise, just 0
#             global_h = 0
#             a = 0
    
#     return a * 0.5

global_h = 0
decide_waveform = 0
decide_wave_w = 0.0
jump_duration = 0
f = 0.1
def u(t):
    global global_h, jump_duration, decide_waveform
    time = np.random.randn()
    if jump_duration > 0: # maintain amplitude during duration & reduce jump_duration
        jump_duration = jump_duration - 1
        if decide_waveform == -1:
            a = global_h * np.sin(2*np.pi*f*t)
        else:
            a = global_h + 0.5 * np.random.randn()
    else:
        if time > 2.6: # 2.5 # probability related parameter, (The larger the value, the lower the probability)
            decide_waveform = np.random.randint(0,2)
            global_h = 10 * np.random.rand() - 5
            if decide_waveform == -1:
                jump_duration = np.random.randint(90, 100) # 100,200 # duration related parameter, (The larger the value, the longer the duration)
                a = global_h * np.sin(2*np.pi*f*t)
            else:
                jump_duration = np.random.randint(900, 1000) # 900,1000 # duration related parameter, (The larger the value, the longer the duration)
                a = global_h + 0.5 * np.random.randn()
        else: # otherwise, just 0
            global_h = 0
            a = 0
    
    return a * 0.5


# Y : [ x, x_dot, theta, theta_dot]
def func3( t, y):
    # g = 9.8 
    # L = 1.5 
    # m = 1.0 
    # k = 8 # coefficients c/m
    g = 9.8 
    L = 1./21.  
    m = 1.0 
    k = 0.58 # coefficients c/m
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
        Hz = 50
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
                # cv2.resizeWindow(winname='im', width=2000, height=150)
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
            df.to_csv("../mk/train/train_dataset4_Hz"+str(Hz)+"_"+str(j)+".csv", index = False)
        else:
            df.to_csv("../test/test_dataset_4_Hz"+str(Hz)+"_"+str(j)+".csv", index = False)

