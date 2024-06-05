import pandas as pd
from sklearn.metrics import roc_auc_score
import xgboost as xgb

def train_and_evaluate_xgboost(train_file, val_file):
    # 读取训练集和验证集
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    
    # 填补验证集中work_year的缺失值
    val_df['work_year'].fillna(10, inplace=True)

    # 分离特征和标签
    X_train = train_df.drop(columns=['isDefault'])
    y_train = train_df['isDefault']
    X_val = val_df.drop(columns=['isDefault'])
    y_val = val_df['isDefault']
    
    # 构建XGBoost DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # 设置XGBoost参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'eta': 0.05,
        'verbosity': 1,
        'colsample_bytree': 0.6,
    }
    
    # 训练XGBoost模型
    bst = xgb.train(params, dtrain, num_boost_round=100)
    
    # 在验证集上进行预测
    y_pred_prob = bst.predict(dval)
    
    # 计算AUC
    auc_score = roc_auc_score(y_val, y_pred_prob)
    print(f'AUC Score: {auc_score}')

# 使用示例
train_file = '../assets/combined_0.03.csv'
val_file = '../assets/val_made_work_year.csv'
train_and_evaluate_xgboost(train_file, val_file)
