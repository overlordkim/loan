import pandas as pd

# 读取CSV文件
csv_file_path = '../assets/train_public.csv'  # 将 'your_file.csv' 替换为实际的CSV文件路径
df = pd.read_csv(csv_file_path)

# 获取 pub_dero_bankrup 和 isDefault 列中非零的元素
column = 'pub_dero_bankrup'
non_zero_pub_dero_bankrup = df[(df[column] != 0) & (df[column].notna())]
non_zero_isDefault = df[df['isDefault'] != 0]

# 计算非零元素的数量
non_zero_pub_dero_bankrup_count = non_zero_pub_dero_bankrup.shape[0]
non_zero_isDefault_count = non_zero_isDefault.shape[0]

# 获取 pub_dero_bankrup 和 isDefault 同时非零的交集
intersection = df[((df[column] != 0) & (df[column].notna())) & (df['isDefault'] != 0)]
intersection_count = intersection.shape[0]

# 打印结果
print(f"Number of non-zero elements in pub_dero_bankrup: {non_zero_pub_dero_bankrup_count}")
print(f"Number of non-zero elements in isDefault: {non_zero_isDefault_count}")
print(f"Number of non-zero elements in both pub_dero_bankrup and isDefault: {intersection_count}")
