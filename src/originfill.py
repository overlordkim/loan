import pandas as pd
import datetime as dt
from sklearn.preprocessing import LabelEncoder

# 读取CSV文件
csv_file_path = '../assets/val.csv'  # 请替换为实际的CSV文件路径
df = pd.read_csv(csv_file_path)

# Step 1: 替换 pub_dero_bankrup 中的缺失值为 0
df['pub_dero_bankrup'].fillna(0, inplace=True)
df['pub_dero_bankrup'] = df['pub_dero_bankrup'].astype(int)  # 确保为整数

# Step 2: 删除 f1 变量
df.drop(columns=['f1'], inplace=True)

# Step 3: 对 f0, f2, f3, f4 使用均值进行填充，并确保转换为整数
mean_values = {
    'f0': 6,
    'f2': 8,
    'f3': 15,
    'f4': 8
}

for col, mean in mean_values.items():
    df[col].fillna(mean, inplace=True)
    df[col] = df[col].astype(int)

# 额外步骤 1: 删除指定的列
columns_to_drop = ['loan_id', 'user_id', 'post_code', 'region', 'title']
df.drop(columns=columns_to_drop, inplace=True)

# 额外步骤 2: 转换 issue_date 格式
df['issue_date'] = pd.to_datetime(df['issue_date'], format='%Y/%m/%d')
df['issue_date'] = df['issue_date'].dt.strftime('%Y%m%d')

# 额外步骤 3: 转换 earlies_credit_mon 格式
def convert_earlies_credit_mon(date_str):
    try:
        if '-' in date_str:
            parts = date_str.split('-')
            if parts[0].isdigit():
                # 格式为 '92-Feb' 或 '5-Mar'
                year, month = parts
                month = dt.datetime.strptime(month, '%b').month
                year = int(year)
                if year < 20:  # 小于20的年份为2000年后的年份
                    year += 2000
                else:  # 大于等于20的年份为1900年后的年份
                    year += 1900
            else:
                # 格式为 'Jan-83' 或 'Mar-6'
                month, year = parts
                month = dt.datetime.strptime(month, '%b').month
                year = int(year)
                if year < 20:  # 小于20的年份为2000年后的年份
                    year += 2000
                else:  # 大于等于20的年份为1900年后的年份
                    year += 1900
            return f'{year:04d}{month:02d}'
    except Exception as e:
        print(f"Error processing date: {date_str}, Error: {e}")
    return date_str

df['earlies_credit_mon'] = df['earlies_credit_mon'].apply(convert_earlies_credit_mon)

# 额外步骤 4: 标签编码（Ordinal）和独热编码（Nominal）

# 标签编码有序分类特征（Ordinal）
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

# 独热编码无序分类特征（Nominal）
df = pd.get_dummies(df, columns=['employer_type', 'industry'])

# 删除 recircle_u 和 debt_loan_ratio 中含有缺失值的样本
df.dropna(subset=['recircle_u', 'debt_loan_ratio'], inplace=True)

# 检查结果
print(df.isnull().sum())
print(df.head())

# 保存处理后的数据
output_csv_file_path = '../assets/val_processed.csv'
df.to_csv(output_csv_file_path, index=False)
print(f"Processed data saved to {output_csv_file_path}")