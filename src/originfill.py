import pandas as pd
import datetime as dt
from sklearn.preprocessing import LabelEncoder

# 读取CSV文件
csv_file_path = '../assets/test_public.csv'  # 请替换为实际的CSV文件路径
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

# 将issue_date转换为datetime对象
df['issue_date'] = pd.to_datetime(df['issue_date'], format='%Y/%m/%d')

# 提取年和月
df['issue_year'] = df['issue_date'].dt.year
df['issue_month'] = df['issue_date'].dt.month

# 删除原始的issue_date列
df.drop(columns=['issue_date'], inplace=True)

# 转换earlies_credit_mon格式的函数
def convert_earlies_credit_mon(date_str):
    try:
        if '-' in date_str:
            parts = date_str.split('-')
            if parts[0].isdigit():  # 格式为 '92-Feb' 或 '5-Mar'
                year, month = parts
                month = dt.datetime.strptime(month, '%b').month
                year = int(year)
                if year < 20:  # 小于20的年份为2000年后的年份
                    year += 2000
                else:  # 大于等于20的年份为1900年后的年份
                    year += 1900
            else:  # 格式为 'Jan-83' 或 'Mar-6'
                month, year = parts
                month = dt.datetime.strptime(month, '%b').month
                year = int(year)
                if year < 20:  # 小于20的年份为2000年后的年份
                    year += 2000
                else:  # 大于等于20的年份为1900年后的年份
                    year += 1900
            return year, month
    except Exception as e:
        print(f"Error processing date: {date_str}, Error: {e}")
        return None, None  # 在异常情况下返回None

# 应用转换函数并拆分为两个新列
df['earlies_credit_year'], df['earlies_credit_month'] = zip(*df['earlies_credit_mon'].apply(convert_earlies_credit_mon))

# 删除原始的earlies_credit_mon列
df.drop(columns=['earlies_credit_mon'], inplace=True)

# 作比值

# 创建新的布尔变量
#df['early_return_bool'] = df['early_return'] > 0
# 检查并修改early_return
#df.loc[(df['early_return_amount'] != 0) & (df['early_return'] == 0), 'early_return'] = 1
# 计算比值并创建新列
df['early_return_ratio'] = df['early_return_amount'] / df['early_return']

# 额外步骤 4: 标签编码（Ordinal）和独热编码（Nominal）

# 独热编码无序分类特征（Nominal）
df = pd.get_dummies(df, columns=['class', 'employer_type', 'industry'])

# 删除 recircle_u 和 debt_loan_ratio 中含有缺失值的样本
df.dropna(subset=['recircle_u', 'debt_loan_ratio'], inplace=True)

# 检查结果
print(df.isnull().sum())
print(df.head())

# 保存处理后的数据
output_csv_file_path = '../assets/test_processed.csv'
df.to_csv(output_csv_file_path, index=False)
print(f"Processed data saved to {output_csv_file_path}")