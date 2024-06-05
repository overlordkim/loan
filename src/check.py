import pandas as pd

# 设置显示选项，确保所有行都被显示
pd.set_option('display.max_rows', None)

# 读取CSV文件
csv_file_path = '../assets/train_public_work_year_filled.csv'  # 将 'your_file.csv' 替换为实际的CSV文件路径
df = pd.read_csv(csv_file_path)


column = 'known_outstanding_loan'
# 获取既不是0也不是NaN的元素
non_zero_non_nan = df[((df[column].notna()) & (df[column]>=0))][column]

# 打印非零的元素
print(non_zero_non_nan)
print(non_zero_non_nan.count())

# 计算并打印均值和方差
mean_value = non_zero_non_nan.mean()
variance_value = non_zero_non_nan.var()

print(f"Mean of non-zero and non-NaN elements: {mean_value}")
print(f"Variance of non-zero and non-NaN elements: {variance_value}")