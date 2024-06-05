import pandas as pd

# 读取CSV文件
csv_file_path = '../assets/val_processed.csv'
df = pd.read_csv(csv_file_path)

# 定义 work_year 的映射
work_year_mapping = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10
}

# 替换 work_year 为有序变量
df['work_year'] = df['work_year'].map(work_year_mapping)

# 检查结果
print(df['work_year'].unique())

# 保存处理后的数据
output_csv_file_path = '../assets/val_made_work_year.csv'
df.to_csv(output_csv_file_path, index=False)
print(f"Processed data saved to {output_csv_file_path}")