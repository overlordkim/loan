import pandas as pd
import xgboost as xgb

def train_and_predict_xgboost(train_file, test_file, submission_file):
    # 读取训练集和测试集
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # 填补训练集和测试集中的缺失值
    train_df['work_year'].fillna(10, inplace=True)
    test_df['work_year'].fillna(10, inplace=True)

    # 分离训练集的特征和标签
    X_train = train_df.drop(columns=['isDefault'])
    y_train = train_df['isDefault']
    
    # 提取测试集的loan_id和特征
    loan_ids = test_df['loan_id']
    X_test = test_df.drop(columns=['loan_id'])
    
    # 构建XGBoost DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    
    # 设置XGBoost参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'eta': 0.1,
        'verbosity': 1
    }
    
    # 训练XGBoost模型
    bst = xgb.train(params, dtrain, num_boost_round=100)
    
    # 在测试集上进行预测
    y_pred_prob = bst.predict(dtest)
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'loan_id': loan_ids,
        'isDefault': y_pred_prob
    })
    
    # 保存提交文件
    submission_df.to_csv(submission_file, index=False)
    print(f'Submission file saved to {submission_file}')

# 使用示例
train_file = '../assets/combined_0.04.csv'
test_file = '../assets/test_public_to_predict.csv'
submission_file = '../assets/submission.csv'
train_and_predict_xgboost(train_file, test_file, submission_file)