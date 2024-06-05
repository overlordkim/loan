import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(input_file, train_output_file, val_output_file, train_size=0.9):
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 检查样本数是否为10000
    if len(df) != 10000:
        print(f"输入文件的样本数不是10000，而是{len(df)}")
        return
    
    # 分割数据集
    train_df, val_df = train_test_split(df, train_size=train_size, random_state=42)
    
    # 保存训练集和验证集到新的CSV文件
    train_df.to_csv(train_output_file, index=False)
    val_df.to_csv(val_output_file, index=False)
    print(f"训练集保存到 {train_output_file}，验证集保存到 {val_output_file}")

# 使用示例
input_file = '../assets/train_public.csv'
train_output_file = '../assets/train.csv'
val_output_file = '../assets/val.csv'
split_dataset(input_file, train_output_file, val_output_file)
