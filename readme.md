# Datamining Final Project


## 无用值删除

### loan_id

- 贷款编号？应该没有用，删掉了

### user_id

- 用户编号？也删掉了

### post_code

- 邮政编码，删掉了

### region

- 地区编号，大概问了一下gpt，看不出编码规则，也删掉了

### title

- 看值看不出什么意思，删掉了

## 日期转换

### earlies_credit_mon

- 最早贷款月份

- 格式类似`Jan-83`，是月份和年份，提取出了月份和完整年份，然后整合成年-月，这样可以当作interval变量使用（比如这个就对应198301）

### issue_date

- 完整的借款日期

- 格式类似`2014/9/1`，整合成了年-月-日，可以当作interval（20140901）

## 名义值映射

### class

- 贷款等级

- 分为七个等级ABCDEFG，分别映射到0123456

### employer_type

- 企业类型

- 用了自带的独热编码器，分成了以下六种，对应的值是True或者False

    - employer_type_上市企业
    - employer_type_世界五百强
    - employer_type_幼教与中小学校
    - employer_type_政府机构
    - employer_type_普通企业
    - employer_type_高等教育机构

### industry

- 行业类型

- 用了自带的独热编码器，分成了以下十四种，对应的值是True或者False

    - industry_交通运输、仓储和邮政业
    - industry_住宿和餐饮业
    - industry_信息传输、软件和信息技术服务业
    - industry_公共服务、社会组织
    - industry_农、林、牧、渔业
    - industry_制造业
    - industry_国际组织
    - industry_建筑业
    - industry_房地产业
    - industry_批发和零售业
    - industry_文化和体育业
    - industry_电力、热力生产供应业
    - industry_采矿业
    - industry_金融业

## 异常值处理

对每个表项做了异常值检测，有异常值的表项占大多数，但其中有一些表项的异常值非常离谱，远远超出了正常范围：

### debt_loan_ratio

- 贷款余额与贷款总额的比值

- 把>50的删掉了，删掉了25个

- 大多数小于40的情况下，有的是三位数，甚至有999（可能指的是无限大）

### earlies_credit_mon

- 最早贷款月份

- 把<197000的删掉了，也是刚好删掉25个

- 太久远的不考虑了，最老的有1952年贷过款的

### house_exist

- 拥有房产数量

- 把>2的删掉了，删掉了两个3和一个4

- 有三四套房子还来贷款的人实在是凤毛麟角

### known_outstanding_loan

- 已知未偿还贷款

- 把>36的删掉了，删掉了22个

### recircle_b

- 某种关于循环的量

- 把>174000的删掉了，也是25个

- 最大有800000的

## 缺失值处理

### pub_dero_bankrup

- 公开破产次数

- 缺失7个，非零1296个，是1的1223个（13%的人破过产，并且其中绝大多数只破产一次）

- 缺失值用0代替（贝叶斯？）

### f1

- 未知的废物变量

- 缺失858个，非零13个，全是1，并且其中只有2个和赖账重合（2/13）

- 直接把这项删掉了

### f0、f2、f3、f4

- 未知的有用整型变量

- 缺失498个

- 均值方差大，可能有丰富信息，不应该直接删除

- 均值填充6、8、15、8

### work_year

- ordinal变量，共11种，<1年、1-9年、10+年

- 缺失622个，10+年的占比超过1/3，大部分值具体未知，再用均值替代不合理

- 前述替换后，训练catboost分类器预测了work_year

# todo

## 重建特征

internet比public多出以下这些项目：

- offsprings：子女数量
- marriage：婚姻状况
- work_type：工作类型
- f5：未知变量
- house_loan_status：房贷情况
- sub_class：贷款子类型

其中sub_class是对public中的class的进一步拓展，每种大类型都拥有5中小类型，比如A类贷款有A1-A5这五种子类型贷款

在public中使用kmeans对于每种大类型贷款重建5种子类型贷款。

