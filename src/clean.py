import pandas as pd

# 读取CSV文件
csv_file_path = '../assets/train_processed.csv'
df = pd.read_csv(csv_file_path)

# 异常值处理规则
# 1. debt_loan_ratio: 删除 > 50 的
df = df[df['debt_loan_ratio'] <= 50]

# 2. earlies_credit_mon: 删除 < 197000 的
df = df[df['earlies_credit_mon'].astype(int) >= 197000]

# 3. house_exist: 删除 > 2 的
df = df[df['house_exist'] <= 2]

# 4. known_outstanding_loan: 删除 > 36 的
# df = df[df['known_outstanding_loan'] <= 36]

# 5. recircle_b: 删除 > 174000 的
df = df[df['recircle_b'] <= 174000]

# 保存处理后的数据
output_csv_file_path = '../assets/train_cleaned.csv'
df.to_csv(output_csv_file_path, index=False)
print(f"Cleaned data saved to {output_csv_file_path}")