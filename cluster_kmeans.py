import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import dtw
import numpy as np

df = pd.read_csv('./data/match_output.csv')

df['coordinates'] = df['matched_coordinates'].apply(lambda x: eval(x))
df['longitude'] = df['coordinates'].apply(lambda x: x[0])
df['latitude'] = df['coordinates'].apply(lambda x: x[1])

#构建时间序列数据集
time_series_dataset = df.groupby('traj_id')[['longitude', 'latitude']].apply(lambda x: x.values.tolist()).tolist()

print("计算DTW距离矩阵……")
len_ = len(time_series_dataset)
dtw_distance_matrix = np.zeros((len_, len_))
for i in range(len_):
    for j in range(i+1, len_):
        dtw_distance_matrix[i, j] = dtw(time_series_dataset[i], time_series_dataset[j],distance_only=True)
    progress = (i / len_) * 100
    print(f"\r进度：{progress:.2f}%", end='', flush=True)

dtw_distance_matrix += dtw_distance_matrix.T

#选取特征并操纵
dtw = dtw_distance_matrix.mean(axis=1)
df = df.merge(dtw.reset_index(name='dtw'), on='traj_id')
df['time'] = pd.to_datetime(df['time'])
df['time'] = pd.to_datetime(df['time'])
df['timestamp'] = df['time'].apply(lambda x: x.timestamp())
traj_group_mean_speeds = df.groupby('traj_id')['speeds'].mean()
df = df.merge(traj_group_mean_speeds.reset_index(name='mean_speed'), on='traj_id')

features = ['longitude', 'latitude', 'dtw', 'mean_speed', 'timestamp']

#标准化特征
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

n_clusters = 50
X = df[features]
print("\n开始聚类……")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster_label'] = kmeans.fit_predict(X)

print("开始绘制……")
plt.figure(figsize=(12, 8))
for cluster in range(n_clusters):
    cluster_data = df[df['cluster_label'] == cluster]
    for idx, row in cluster_data.iterrows():
        plt.plot(row['longitude'], row['latitude'], label=f'Cluster {cluster}', color=f'C{cluster}', linewidth=0.5)

plt.title('Trajectories Clustered by K-Means using DTW')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.legend()
plt.show()
