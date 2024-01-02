from datetime import datetime
import pandas as pd
import pathlib
from xgboost import XGBRegressor

cwd = pathlib.Path.cwd()
data = pd.read_csv(cwd / 'data' / 'traj.csv', index_col=0)

def process_data(data):
    # 将time转化为时间格式，并生成相应的时间戳
    data['time'] = pd.to_datetime(data['time'])
    data['timestamp'] = data['time'].apply(lambda x: x.timestamp())
    # 将coordinates拆分为经纬度
    data['end_longitude'] = data['coordinates'].apply(lambda x: eval(x)[0])
    data['end_latitude'] = data['coordinates'].apply(lambda x: eval(x)[1])

def merge_data(group):
    group['start_longitude'] = group['end_longitude'].iloc[0]
    group['start_latitude'] = group['end_latitude'].iloc[0]
    group['avg_speed'] = (group['speeds'] + group['speeds'].iloc[0]) / 2
    group['time_diff'] = group['timestamp'] - group['timestamp'].iloc[0]
    group = group.iloc[1:]
    return group

grouped_train_data = data.groupby('traj_id')
process_data(data)
train = pd.DataFrame()
train = grouped_train_data.apply(merge_data)
#得到处理后的训练集数据train

features = [
    'entity_id'
    ,'avg_speed'
    , 'holidays'
    , 'start_longitude'
    , 'start_latitude'
    , 'end_longitude'
    , 'end_latitude'
    , 'current_dis'
    , 'timestamp'
    ]
target = 'time_diff'

#划分特征和标签
X_train = train[features]
y_train = train[target]

model = XGBRegressor(n_estimators=1000
                     ,subsample=0.9
                     ,reg_lambda=100
                     ,reg_alpha=100
                     ,min_child_weight=3
                     ,max_depth=10
                     ,learning_rate=0.3
                     ,gamma=0.1
                     ,colsample_bytree=0.6
                   )
#训练模型
model = model.fit(X_train, y_train)

test = pd.read_csv(cwd / 'data' /'eta_task.csv')

def process_test_data(data):
    data['time'] = pd.to_datetime(data['time']).fillna(method='ffill')
    data['timestamp'] = data['time'].apply(lambda x: x.timestamp())
    data['end_longitude'] = data['coordinates'].apply(lambda x: eval(x)[0])
    data['end_latitude'] = data['coordinates'].apply(lambda x: eval(x)[1])

def merge_test_data(group):
    group['start_id'] = group['id'].iloc[0]
    group['end_id'] = group['id']
    group['start_longitude'] = group['end_longitude'].iloc[0]
    group['start_latitude'] = group['end_latitude'].iloc[0]
    group['start_speeds'] = group['speeds'].iloc[0]
    group['end_speeds'] = group['speeds']
    group['avg_speed'] = (group['end_speeds'] + group['start_speeds']) / 2
    group = group.iloc[1:]
    return group

grouped_test_data = test.groupby('traj_id')

process_test_data(test)

test_data = grouped_test_data.apply(merge_test_data)
test_x = test_data[features]

#预测
test_pred = model.predict(test_x)
test_data['time_diff'] = test_pred

# 创建一个新的DataFrame来存储拆分后的数据
new_rows = []

# 遍历原始DataFrame的每一行
for _, row in test_data.iterrows():
    start_row = row.copy()  # 复制原始行数据
    end_row = row.copy()    # 复制原始行数据
    # 修改第一行的值
    start_row['id'] = start_row['start_id']
    start_row['coordinates'] = [start_row['start_longitude'], start_row['start_latitude']]
    start_row['current_dis'] = 0
    start_row['speeds'] = start_row['start_speeds']
    start_row['time'] = datetime.utcfromtimestamp(start_row['timestamp']).strftime('%Y-%m-%dT%H:%M:%SZ')
    # 修改第二行的值
    end_row['id'] = end_row['end_id']
    end_row['coordinates'] = [end_row['end_longitude'], end_row['end_latitude']]
    end_row['speeds'] = end_row['end_speeds']
    end_row['timestamp'] = end_row['timestamp'] + end_row['time_diff']
    end_row['time'] = datetime.utcfromtimestamp(end_row['timestamp']).strftime('%Y-%m-%dT%H:%M:%SZ')
    # 添加两行到新的DataFrame
    new_rows.append(start_row)
    new_rows.append(end_row)

# 创建新的DataFrame
new_df = pd.DataFrame(new_rows, columns=['id', 'traj_id', 'time', 'entity_id', 'coordinates', 'current_dis',
                                         'speeds', 'holidays'])
#输出最终预测结果
new_df.to_csv("ETA_pred.csv",index=False)

