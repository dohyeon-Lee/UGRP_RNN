import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = "../mk/train/train_withcontrol3_Hz50.csv" #"../test/test_withcontrol3_Hz50_0.csv" #""../test/test_withcontrol_Hz50_2.csv" #"../train/train_real0.csv"#"../test/test_real_LQR_1.csv"

data = pd.read_csv(file_path)
plt.figure(figsize=(20, 20))
i=0
for column in data.columns:
    i += 1
    plt.subplot(3,1,i)
    time = np.linspace(0, len(data[column])/50, len(data[column]))
    plt.plot(time, data[column], label=column)
    plt.title(column)
plt.show()
