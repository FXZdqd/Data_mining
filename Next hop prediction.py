import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pathlib
import numpy as np
from ast import literal_eval

def process_data(data):
    # Step 1: 处理缺失值
    data.ffill()  # 使用前一个值填充缺失值

    # Step 2: 特征工程
    data['time'] = pd.to_datetime(data['time'])
    data['timestamp'] = data['time'].apply(lambda x: x.timestamp())

    # 将 'coordinates' 列拆分为 'latitude' 和 'longitude' 两列
    data['latitude'] = data['coordinates'].apply(lambda x: eval(x)[0] if pd.notnull(x) else np.nan)
    data['longitude'] = data['coordinates'].apply(lambda x: eval(x)[1] if pd.notnull(x) else np.nan)


# 读取CSV文件到pandas DataFrame并对数据进行预处理
cwd = pathlib.Path.cwd()
train = pd.read_csv(cwd / 'data' / 'traj.csv', index_col=0)
process_data(train)

# 数据拆分
X = train[['timestamp', 'entity_id', 'traj_id', 'speeds', 'holidays']]

y_coordinates = train[['latitude', 'longitude']]
y_current_dis = train['current_dis']

X_train, X_test, y_coordinates_train, y_coordinates_test, y_current_dis_train, y_current_dis_test = train_test_split(
    X, y_coordinates, y_current_dis, test_size=0.2, random_state=42
)

# 选择模型
model_coordinates = RandomForestRegressor()
model_current_dis = RandomForestRegressor()

# 训练
model_coordinates.fit(X_train, y_coordinates_train)
model_current_dis.fit(X_train, y_current_dis_train)

# 测试
y_coordinates_pred = model_coordinates.predict(X_test)
y_current_dis_pred = model_current_dis.predict(X_test)

# 评估
mse_coordinates = mean_squared_error(y_coordinates_test, y_coordinates_pred)
mse_current_dis = mean_squared_error(y_current_dis_test, y_current_dis_pred)

print(f'Mean Squared Error (coordinates): {mse_coordinates}')
print(f'Mean Squared Error (current_dis): {mse_current_dis}')

# 对新数据进行预测
new_data = pd.read_csv(cwd / 'data' / 'jump_task.csv', index_col=0)
process_data(new_data)

X_new = new_data[['timestamp', 'entity_id', 'traj_id', 'speeds', 'holidays']]

new_coordinates_pred = model_coordinates.predict(X_new)

# Add the predicted coordinates to the new_data DataFrame
new_data['latitude'] = new_coordinates_pred[:, 0]
new_data['longitude'] = new_coordinates_pred[:, 1]

# Save the results to predict.csv
new_data.to_csv(cwd / 'predict.csv', index=False)
