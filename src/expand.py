import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# 读取public和internet表
public_csv_path = '../assets/train_built_work_year.csv'
internet_csv_path = '../assets/train_internet_work_year_filled.csv'

public_df = pd.read_csv(public_csv_path)
internet_df = pd.read_csv(internet_csv_path)

# 假设target是违约标志列，1代表违约，0代表未违约
target_col = 'isDefault'

# 检查并补齐internet表缺失的表项
missing_columns = set(public_df.columns) - set(internet_df.columns)
for col in missing_columns:
    public_mean = public_df[col].mean()
    internet_df[col] = public_mean

# 填充internet表中的其他缺失值为public表的均值
for col in internet_df.columns:
    if internet_df[col].isnull().any():
        internet_df[col].fillna(public_df[col].mean(), inplace=True)

# 分离特征和目标变量
X_public = public_df.drop(columns=[target_col])
y_public = public_df[target_col]
X_internet = internet_df.drop(columns=[target_col])

# 训练CatBoost模型
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    eval_metric='Accuracy',
    verbose=False
)

model.fit(X_public, y_public)

# 对internet表进行违约概率预测
predictions_proba = model.predict_proba(X_internet)[:, 1]

# 打印不同阈值下符合条件的项目数量
thresholds = np.arange(0.01, 0.101, 0.01)

for threshold in thresholds:
    selected_indices = internet_df[((predictions_proba < threshold) & (internet_df[target_col] == 0)) | ((predictions_proba > 1-threshold) & (internet_df[target_col] == 1))].index
    count = len(selected_indices)
    print(f"Threshold: {threshold:.2f}, Number of non-default samples: {count}")

    # 保存选中的行并与原始public表组合
    selected_df = internet_df.loc[selected_indices]
    combined_df = pd.concat([public_df, selected_df])
    output_csv_path = f'../assets/combined_{threshold:.2f}.csv'
    combined_df.to_csv(output_csv_path, index=False)
    print(f'Saved to {output_csv_path}')
'''

# 示例：使用某个阈值（例如 0.07）来筛选样本
threshold = 0.07
selected_indices = internet_df[(predictions_proba < threshold) & (internet_df[target_col] == 0)].index

# 从 internet_df 中筛选出被选中的样本
selected_internet_df = internet_df.loc[selected_indices]

# 去除选中样本中的 target 列，进行合并
selected_internet_df = selected_internet_df.assign(target=0)

# 将筛选结果与原始public表合并
combined_df = pd.concat([public_df, selected_internet_df], ignore_index=True)

# 保存最终的训练集
output_csv_path = '../assets/combined_train_public_internet.csv'
combined_df.to_csv(output_csv_path, index=False)
print(f"Combined dataset saved to {output_csv_path}")

'''