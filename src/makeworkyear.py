import pandas as pd

# 读取CSV文件
csv_file_path = '../assets/test_processed.csv'
df = pd.read_csv(csv_file_path)

# 定义 work_year 的映射
work_year_mapping = {
    "< 1 year": "less_than_1",
    "1 year": "1_year",
    "2 years": "2_years",
    "3 years": "3_years",
    "4 years": "4_years",
    "5 years": "5_years",
    "6 years": "6_years",
    "7 years": "7_years",
    "8 years": "8_years",
    "9 years": "9_years",
    "10+ years": "more_than_10"
}

# 使用众数填充缺失的 work_year
mode_work_year = df['work_year'].mode()[0]
df['work_year'].fillna(mode_work_year, inplace=True)

# 替换 work_year 为名义变量
df['work_year'] = df['work_year'].map(work_year_mapping)

# 将名义变量进行独热编码
df = pd.get_dummies(df, columns=['work_year'])

# 检查结果
print(df.head())

# 保存处理后的数据
output_csv_file_path = '../assets/test_made_work_year.csv'
df.to_csv(output_csv_file_path, index=False)
print(f"Processed data saved to {output_csv_file_path}")
