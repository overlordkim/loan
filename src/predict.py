import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import xgboost as xgb

def replace_inf_with_max(df):
    for column in df.columns:
        max_value = df[column].replace([np.inf, -np.inf], np.nan).max()
        df[column].replace([np.inf, -np.inf, np.nan], max_value, inplace=True)

def check_for_nan(df, dataset_name):
    if df.isnull().any().any():
        nan_columns = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"NaN values detected in {dataset_name}. Columns with NaN: {nan_columns}")

def train_and_predict_xgboost(train_file, test_file, submission_file):
    # 读取训练集和测试集
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # 填补训练集和测试集中的缺失值
    #train_df['work_year'].fillna(10, inplace=True)
    #test_df['work_year'].fillna(10, inplace=True)

    # 分离训练集的特征和标签s
    X_train = train_df.drop(columns=['isDefault'])
    y_train = train_df['isDefault']
    
    # 提取测试集的loan_id和特征
    #loan_ids = test_df['loan_id']
    #X_test = test_df.drop(columns=['loan_id'])
    X_test = test_df
    
    # 处理无穷大值
    replace_inf_with_max(X_train)
    replace_inf_with_max(X_test)
    
    # 检查NaN值
    check_for_nan(X_train, "training dataset")
    check_for_nan(X_test, "test dataset")
    
    # 归一化特征
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 获取所有以'class_'开头的列
    class_columns = [col for col in X_train.columns if col.startswith('class_')]
    
    # 对每个类别应用KMeans聚类，并创建新的One-Hot编码列
    for class_col in class_columns:
        class_data = X_train_scaled[X_train[class_col] == 1]
        if class_data.shape[0] > 0:  # 确保至少有一条数据
            kmeans = KMeans(n_clusters=5, random_state=2021)
            kmeans.fit(class_data)
            
            # 为训练和测试数据集创建聚类结果作为新特征
            train_cluster_labels = kmeans.predict(X_train_scaled[X_train[class_col] == 1])
            test_cluster_labels = kmeans.predict(X_test_scaled[X_test[class_col] == 1])
            
            # 将整数聚类标签转换为One-Hot编码形式
            for i in range(5):
                X_train[class_col + str(i)] = 0
                X_train.loc[X_train[class_col] == 1, class_col + str(i)] = (train_cluster_labels == i).astype(int)
                
                X_test[class_col + str(i)] = 0
                X_test.loc[X_test[class_col] == 1, class_col + str(i)] = (test_cluster_labels == i).astype(int)
    
    # 构建XGBoost DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    
    # 设置XGBoost参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 5,
        'eta': 0.08,
        'verbosity': 1,
        'colsample_bylevel': 1,
        'alpha': 0.2,
        'reg_lambda': 0.2,
        'subsample': 0.8,
        'min_child_weight': 2,
        'tree_method': 'hist',
        'n_jobs': -1,
    }
    
    # 训练XGBoost模型
    bst = xgb.train(params, dtrain, num_boost_round=100)
    
    # 在测试集上进行预测
    y_pred_prob = bst.predict(dtest)
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'isDefault': y_pred_prob
    })
    
    # 保存提交文件
    submission_df.to_csv(submission_file, index=False)
    print(f'Submission file saved to {submission_file}')

# 使用示例
train_file = '../assets/combined_0.02.csv'
test_file = '../assets/test_made_work_year.csv'
submission_file = '../assets/submission.csv'
train_and_predict_xgboost(train_file, test_file, submission_file)
