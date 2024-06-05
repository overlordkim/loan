import pandas as pd

# 读取第一个CSV文件
csv_file_path_1 = '../assets/train_public_work_year_filled.csv'  # 请替换为实际的CSV文件路径
df1 = pd.read_csv(csv_file_path_1)

# 读取第二个CSV文件
csv_file_path_2 = '../assets/test_public_to_predict.csv'  # 请替换为实际的CSV文件路径
df2 = pd.read_csv(csv_file_path_2)

# 获取列集合
columns_a = set(df1.columns)
columns_b = set(df2.columns)

# 计算对称差
only_in_a = columns_a - columns_b
only_in_b = columns_b - columns_a

# 打印结果
print(f"Columns only in {csv_file_path_1}:")
print(only_in_a)

print(f"Columns only in {csv_file_path_2}:")
print(only_in_b)

