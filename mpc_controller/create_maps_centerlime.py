import yaml
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
import csv
import matplotlib.pyplot as plt

# === LOAD YAML ===
yaml_path = "/home/giorgos/sim_ws/src/f1tenth_gym_ros/maps/columbia_small.yaml"
with open(yaml_path, 'r') as f:
    map_metadata = yaml.safe_load(f)

resolution = map_metadata['resolution']
origin = map_metadata['origin']  # [x, y, theta]

# === LOAD IMAGE ===
img = np.array(Image.open("/home/giorgos/sim_ws/src/f1tenth_gym_ros/maps/columbia_small.png").convert("L"))

# === THRESHOLD ===
binary = img > 200

# clean noise
binary = remove_small_objects(binary, min_size=500)

# keep largest component (track only)
labels = label(binary)
counts = np.bincount(labels.flat)
counts[0] = 0
binary = labels == np.argmax(counts)

# === DISTANCE MAP ===
dist_map = distance_transform_edt(binary) * resolution

# === SKELETON ===
skeleton = skeletonize(binary)

# === GET POINTS ===
points = np.column_stack(np.where(skeleton))

# === ORDER POINTS (nearest neighbor) ===
points_list = points.tolist()
ordered = [points_list.pop(0)]

while points_list:
    last = np.array(ordered[-1])
    pts = np.array(points_list)
    dists = np.linalg.norm(pts - last, axis=1)
    idx = np.argmin(dists)
    ordered.append(points_list.pop(idx))

ordered = np.array(ordered)

# === DOWNSAMPLE (IMPORTANT) ===
def downsample_by_distance(points, min_dist=3):
    new_pts = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - new_pts[-1]) > min_dist:
            new_pts.append(p)
    return np.array(new_pts)

ordered = downsample_by_distance(ordered, min_dist=3)

# === PIXEL → WORLD + WIDTH ===
height = img.shape[0]
world_points = []

for y_pix, x_pix in ordered:
    x = x_pix * resolution + origin[0]
    y = (height - y_pix) * resolution + origin[1]

    d = dist_map[y_pix, x_pix]

    world_points.append([x, y, d, d])  # right, left

# === SAVE CSV ===
with open("centerline.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"])
    writer.writerows(world_points)

print("✅ centerline.csv created!")

# === PLOTS ===

# walls + centerline
plt.figure(figsize=(8,8))
walls = img < 100
plt.imshow(walls, cmap='gray')

ys = ordered[:,0]
xs = ordered[:,1]
plt.plot(xs, ys, 'r.', markersize=2)

plt.title("Walls + Centerline")
plt.gca().invert_yaxis()
plt.axis('equal')
plt.show()

# track area
plt.figure(figsize=(8,8))
plt.imshow(binary, cmap='gray')
plt.title("Detected Track Area")
plt.show()

# distance map
plt.figure(figsize=(8,8))
plt.imshow(dist_map, cmap='jet')
plt.colorbar(label="Distance (m)")
plt.plot(xs, ys, 'w.', markersize=1)
plt.title("Distance Transform + Centerline")
plt.show()