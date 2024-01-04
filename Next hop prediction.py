import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pathlib

# 读取CSV文件到pandas DataFrame并对数据进行预处理
cwd = pathlib.Path.cwd()
data = pd.read_csv(cwd / 'data' / 'traj.csv', index_col=0)

data['time'] = pd.to_datetime(data['time'])
data['timestamp'] = data['time'].apply(lambda x: x.timestamp())

# 将 'coordinates' 列拆分为 'latitude' 和 'longitude' 两列
data['latitude'] = data['coordinates'].apply(lambda x: eval(x)[0])
data['longitude'] = data['coordinates'].apply(lambda x: eval(x)[1])

# 将轨迹点进行偏移以创建一个新的目标列进行预测
data['next_latitude'] = data['latitude'].shift(-1)
data['next_longitude'] = data['longitude'].shift(-1)

# 删除由偏移创建的包含NaN值的最后一行
data = data.dropna()

# 将空间位置离散化为类别（这里简化为经纬度所在的象限）
data['category'] = np.sign(data['next_latitude'] - data['latitude']).astype(str) + \
                  np.sign(data['next_longitude'] - data['longitude']).astype(str)

# 定义特征列和目标列
feature_columns = ['latitude', 'longitude', 'current_dis', 'speeds', 'holidays', 'timestamp']
target_column = 'category'

# 将数据划分为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 创建一个RandomForestClassifier模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 在训练集上训练模型
model.fit(train_data[feature_columns], train_data[target_column])

# 在测试集上预测
predictions = model.predict(test_data[feature_columns])

# 计算准确率
accuracy = accuracy_score(test_data[target_column], predictions)
print(f"准确率：{accuracy}")

# 打印混淆矩阵
conf_matrix = confusion_matrix(test_data[target_column], predictions)
print("混淆矩阵：")
print(conf_matrix)

# 预测新数据点的下一个轨迹点
new_data_point = np.array([[116.461418, 39.920624, 0.323132219, 48.7275, 0, 0]])
predicted_category = model.predict(new_data_point)

print(f'预测的下一个轨迹点类别：{predicted_category}')
predicted_category.to_csv('predict.txt', header=False, index=False)