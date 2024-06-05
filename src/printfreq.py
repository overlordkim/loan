import pandas as pd
import matplotlib.pyplot as plt

column = 'work_year'

# 读取CSV文件
csv_file_path = '../assets/train_public.csv'  # 请替换为实际的CSV文件路径
df = pd.read_csv(csv_file_path)

# 过滤掉 NaN 值
non_nan = df[column].dropna()

# 定义替换函数
def replace_years(x):
    if x == "10+ years":
        return "10+"
    elif x == "1 year":
        return "1"
    elif x == "< 1 year":
        return "<1"
    else:
        return x.replace(" years", "")

# 应用替换函数
years = non_nan.apply(replace_years)

# 计算频数并排序
frequency = years.value_counts().sort_values(ascending=False)

# 绘制柱状图
plt.figure(figsize=(12, 8))
bars = plt.bar(frequency.index, frequency.values, color='skyblue', edgecolor='black')

# 设置标题和标签
plt.title('Frequency Bar Chart of ' + column, fontsize=16)
plt.xlabel(column, fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# 旋转 x 轴标签
plt.xticks(rotation=0, ha='right')

# 显示网格
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存图像
output_path = '../graphs/' + column + '_bar_sorted_by_frequency.png'
plt.tight_layout()
plt.savefig(output_path, format='png')
plt.close()  # 关闭图像显示

# 打印提示信息
print(f"Bar chart saved to {output_path}")