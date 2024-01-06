import pandas as pd
import matplotlib.pyplot as plt


file_path = "../test/test_dataset_4_Hz80_4.csv"#"../train/train_real0.csv"#"../test/test_real_LQR_1.csv"

data = pd.read_csv(file_path)
plt.figure(figsize=(20, 20))
i=0
for column in data.columns:
    i += 1
    plt.subplot(3,1,i)
    plt.plot(data[column], label=column)
    plt.title(column)
plt.show()
