import pandas as pd

# 读取CSV文件
csv_file_path = '../assets/train_internet_work_year_filled.csv'  # 请替换为实际的CSV文件路径
df = pd.read_csv(csv_file_path)

# 查看原始列名称
print("Original columns:")
print(df.columns)

# 重命名列名称
df.rename(columns={'is_default': 'isDefault'}, inplace=True)

# 确认更改
print("Updated columns:")
print(df.columns)

# 保存修改后的数据
output_csv_file_path = '../assets/train_internet_work_year_filled.csv'
df.to_csv(output_csv_file_path, index=False)
print(f"Processed data saved to {output_csv_file_path}")