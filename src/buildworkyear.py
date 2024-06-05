import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

# 读取处理后的CSV文件
csv_file_path = '../assets/train_made_work_year.csv'
df = pd.read_csv(csv_file_path)

# 分离训练集和预测集
train_df = df[df['work_year'].notna()]
predict_df = df[df['work_year'].isna()]

# 特征和目标变量
X_train = train_df.drop(columns=['work_year'])
y_train = train_df['work_year']
X_predict = predict_df.drop(columns=['work_year'])

# 使用CatBoostClassifier模型进行训练
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    eval_metric='Accuracy',
    verbose=False
)

# 训练模型
model.fit(X_train, y_train)

# 预测缺失值
predictions = model.predict(X_predict)

# 打印被预测项的编号和预测值
#for idx, prediction in zip(predict_df.index, predictions):
    #print(f"Index: {idx}, Predicted work_year: {prediction}")

# 填补缺失值
df.loc[df['work_year'].isna(), 'work_year'] = predictions

# 验证补全效果
print(df.isnull().sum())

# 保存处理后的数据
output_csv_file_path = '../assets/train_built_work_year.csv'
df.to_csv(output_csv_file_path, index=False)
print(f"Processed data with filled work_year saved to {output_csv_file_path}")