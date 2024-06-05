import pandas as pd
import matplotlib.pyplot as plt

def calculate_missing_percentage(csv_file_path):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    # 计算每列的缺失值数量
    missing_count = df.isnull().sum()
    
    # 计算每列的缺失率
    missing_percentage = (missing_count / len(df)) * 100
    
    # 创建一个数据框存储结果
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': missing_count,
        'Missing Percentage': missing_percentage
    })
    
    # 按缺失率降序排列
    missing_df = missing_df.sort_values(by='Missing Percentage', ascending=False)
    
    return missing_df

def plot_missing_data(missing_df, output_path):
    # 筛选出有缺失值的列
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    # 设置字体大小
    plt.rcParams.update({'font.size': 14})
    
    # 绘制直方图
    plt.figure(figsize=(16, 10))
    plt.barh(missing_df['Column'], missing_df['Missing Percentage'], color='skyblue')
    plt.xlabel('Missing Percentage', fontsize=16)
    plt.ylabel('Columns', fontsize=16)
    plt.title('Missing Data Percentage by Column', fontsize=20)
    plt.gca().invert_yaxis()  # 反转Y轴，使缺失最多的列在最上面
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, format='png')
    plt.close()  # 关闭图像显示

# 示例使用
csv_file_path = '../assets/test_public_processed.csv'  # 将 'your_file.csv' 替换为实际的CSV文件路径
output_path = '../graphs/internet_processed_missing.png'  # 输出路径
missing_df = calculate_missing_percentage(csv_file_path)

# 打印所有缺失值的数量和比例
print("Missing Values and Their Percentage:")
print(missing_df)

