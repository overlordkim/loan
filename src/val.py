import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
def train_and_evaluate_xgboost(train_file, val_file):
    # 读取训练集和验证集
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    
    # 分离特征和标签
    X_train = train_df.drop(columns=['isDefault'])
    y_train = train_df['isDefault']
    X_val = val_df.drop(columns=['isDefault'])
    y_val = val_df['isDefault']

    # 处理无穷大值
    def replace_inf_with_max(df):
        # 对于每一列，替换inf为该列的最大有限值
        for column in df.columns:
            max_value = df[column].replace([np.inf, -np.inf], np.nan).max()
            df[column].replace([np.inf, -np.inf, np.nan], max_value, inplace=True)
    
    # 对训练集和验证集应用无穷大值替换
    replace_inf_with_max(X_train)
    replace_inf_with_max(X_val)

    # 检查NaN值
    def check_for_nan(df, dataset_name):
        if df.isnull().any().any():
            nan_columns = df.columns[df.isnull().any()].tolist()
            raise ValueError(f"NaN values detected in {dataset_name}. Columns with NaN: {nan_columns}")

    # 检查训练集和验证集
    check_for_nan(X_train, "training dataset")
    check_for_nan(X_val, "validation dataset")

    # 归一化特征
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 获取所有以'class_'开头的列
    class_columns = [col for col in X_train.columns if col.startswith('class_')]
    
    # 对每个类别应用KMeans聚类，并创建新的One-Hot编码列
    for class_col in class_columns:
        class_data = X_train_scaled[X_train[class_col] == 1]
        if class_data.shape[0] > 0:  # 确保至少有一条数据
            kmeans = KMeans(n_clusters=5, random_state=2021)
            kmeans.fit(class_data)
            
            # 为训练和验证数据集创建聚类结果作为新特征
            train_cluster_labels = kmeans.predict(X_train_scaled[X_train[class_col] == 1])
            val_cluster_labels = kmeans.predict(X_val_scaled[X_val[class_col] == 1])
            
            # 将整数聚类标签转换为One-Hot编码形式
            for i in range(5):
                X_train[class_col + str(i)] = 0
                X_train.loc[X_train[class_col] == 1, class_col + str(i)] = (train_cluster_labels == i).astype(int)
                
                X_val[class_col + str(i)] = 0
                X_val.loc[X_val[class_col] == 1, class_col + str(i)] = (val_cluster_labels == i).astype(int)

    # 构建XGBoost DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
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
    
    # 在验证集上进行预测
    y_pred_prob = bst.predict(dval)
    
    # 计算AUC
    auc_score = roc_auc_score(y_val, y_pred_prob)
    print(f'AUC Score: {auc_score}')

        # 获取特征重要性并绘图
    xgb.plot_importance(bst, max_num_features=10)
    plt.show()

# 使用示例
train_file = '../assets/combined_0.02.csv'
val_file = '../assets/val_made_work_year.csv'
train_and_evaluate_xgboost(train_file, val_file)
