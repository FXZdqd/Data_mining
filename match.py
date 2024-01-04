import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


def load_and_parse(filepath, nrows=None):
    df = pd.read_csv(filepath, nrows=nrows)
    df['coordinates'] = df['coordinates'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    return df


def calculate_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


roads_df = load_and_parse('data/road.csv')
traj_df = load_and_parse('data/traj.csv')


distances = []
for i in range(1, len(traj_df)):
    if traj_df.at[i, 'traj_id'] == traj_df.at[i - 1, 'traj_id']:
        coord1 = traj_df.at[i - 1, 'coordinates']
        coord2 = traj_df.at[i, 'coordinates']
        distances.append(calculate_distance(coord1, coord2))


distance_threshold = np.percentile(distances, 95)


print("开始噪声处理...")
for i in range(1, len(traj_df) - 1):
    if traj_df.at[i, 'traj_id'] == traj_df.at[i - 1, 'traj_id'] == traj_df.at[i + 1, 'traj_id']:
        coord1 = traj_df.at[i - 1, 'coordinates']
        coord2 = traj_df.at[i, 'coordinates']
        coord3 = traj_df.at[i + 1, 'coordinates']

        dist_to_prev = calculate_distance(coord1, coord2)
        dist_to_next = calculate_distance(coord2, coord3)

        if dist_to_prev > distance_threshold and dist_to_next > distance_threshold:
            new_coord = [(coord1[0] + coord3[0]) / 2, (coord1[1] + coord3[1]) / 2]
            traj_df.at[i, 'coordinates'] = new_coord

    if i % 100 == 0:
        print(f"\r噪声处理进度：{i / len(traj_df):.2%}", end='', flush=True)

print("\n噪声处理完成。")


road_points = np.vstack(roads_df['coordinates'].values)
road_tree = KDTree(road_points)


matched_road_indices = road_tree.query(traj_df['coordinates'].tolist(), return_distance=False)
matched_coordinates = [list(road_points[i[0]]) for i in matched_road_indices]


traj_df['matched_coordinates'] = matched_coordinates


traj_df.to_csv("match_output.csv", index=False)


plt.figure(figsize=(10, 6))


print("开始绘制路网...")
for idx, row in enumerate(roads_df.iterrows()):
    coords = np.array(row[1]['coordinates']).T
    plt.plot(coords[0], coords[1], 'b', linewidth=0.5)
    progress = (idx + 1) / len(roads_df) * 100
    print(f"\r路网绘制进度：{progress:.2f}%", end='', flush=True)
print("\n路网绘制完成。")


print("开始绘制匹配的轨迹点...")

coords = traj_df['matched_coordinates'].tolist()
x_coords, y_coords = zip(*coords)
plt.scatter(x_coords, y_coords, c='r', s=1)

print("\n轨迹点绘制完成。")

plt.title('Road Network Matching')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()