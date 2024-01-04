import pandas as pd
import matplotlib.pyplot as plt


file_path = "../test/test_dataset4_100Hz_4.csv"#"../train/train_real0.csv"#"../test/test_real_LQR_1.csv"

data = pd.read_csv(file_path)

for column in data.columns:
    plt.figure(figsize=(20, 3))
    plt.plot(data[column], label=column)
    plt.title(column)
plt.show()
