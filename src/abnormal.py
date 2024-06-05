import pandas as pd
import re
import matplotlib.pyplot as plt

# 读取CSV文件
csv_file_path = '../assets/train_public_processed.csv'  # 请替换为实际的CSV文件路径
df = pd.read_csv(csv_file_path)

# 定义四分位数处理函数
def detect_outliers_by_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return len(outliers), outliers

# 定义一个函数检查表项名称是否包含汉字
def contains_chinese(text):
    return bool(re.search('[\u4e00-\u9fff]', text))

# 遍历每一列，检查数据类型并处理数值型数据
for col in df.columns:
    if contains_chinese(col):
        continue

    # 跳过包含非数值型数据
    if pd.api.types.is_numeric_dtype(df[col]):
        num_outliers, outliers = detect_outliers_by_iqr(df[col])
        if num_outliers != 0:
            print(f"Column: {col}, Number of outliers: {num_outliers}")
    
            # 绘制箱线图
            plt.figure(figsize=(10, 6))
            plt.boxplot(df[col].dropna(), vert=False, patch_artist=True)
            plt.title(f"Boxplot of {col}")
            plt.xlabel(col)
            
            # 保存图像
            output_path = f'../graphs/{col}_boxplot.png'
            plt.savefig(output_path, format='png')
            plt.close()  # 关闭图像以节省内存

            print(f"Boxplot saved to {output_path}")