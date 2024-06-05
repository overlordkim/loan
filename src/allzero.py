import pandas as pd

def find_all_zero_columns(csv_file_path):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    # 初始化一个空列表来存储所有值为0的列
    all_zero_columns = []
    
    # 遍历每一列，检查是否所有值都为0
    for column in df.columns:
        if (df[column] == 0).all():
            all_zero_columns.append(column)
    
    return all_zero_columns

# 示例使用
csv_file_path = '../assets/train_public.csv'  # 将 'your_file.csv' 替换为实际的CSV文件路径
all_zero_columns = find_all_zero_columns(csv_file_path)

# 打印所有值为0的列
print("Columns with all values as 0:")
print(all_zero_columns)
