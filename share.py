import os
import matplotlib
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#%% utility functions
def show_cloud(points_plt):
    ax = plt.axes(projection='3d')
    ax.scatter(points_plt[:,0], points_plt[:,1], points_plt[:,2], s=0.01)
    plt.show()

def show_scatter(x,y):
    plt.scatter(x, y)
    plt.show()

def get_ground_level(pcd):
    z_values = pcd[:,2]

    hist, bin_edges = np.histogram(z_values, bins=100)

    max_bin_index = np.argmax(hist)

    ground_level = (bin_edges[max_bin_index] + bin_edges[max_bin_index+1]) / 2

    return ground_level

# create folder for images
if not os.path.exists("images"):
    os.makedirs("images")

# read file containing point cloud data
pcd = np.load("dataset1.npy")

pcd.shape

# remove ground plane (Task 1 )

est_ground_level = get_ground_level(pcd)
print("dataset1 ground level:", est_ground_level)

pcd_above_ground = pcd[pcd[:,2] > est_ground_level]

pcd_above_ground.shape

# histogram plot for dataset1
plt.figure(figsize=(10,5))
plt.hist(pcd[:,2], bins=100, color='skyblue', edgecolor='black')
plt.axvline(est_ground_level, color='red', linestyle='--', label=f'Ground level = {est_ground_level:.3f}')
plt.title('Histogram of z-values - dataset1')
plt.xlabel('z value')
plt.ylabel('Number of points')
plt.legend()
plt.tight_layout()
plt.show()


unoptimal_eps = 10
# find the elbow
clustering = DBSCAN(eps = unoptimal_eps, min_samples=5).fit(pcd_above_ground)

clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, max(clusters,1))]

# Plotting resulting clusters
plt.figure(figsize=(10,10))
plt.scatter(pcd_above_ground[:,0], 
            pcd_above_ground[:,1],
            c=clustering.labels_,
            cmap=matplotlib.colors.ListedColormap(colors),
            s=2)

plt.title('DBSCAN: %d clusters' % clusters,fontsize=20)
plt.xlabel('x axis',fontsize=14)
plt.ylabel('y axis',fontsize=14)
plt.show()


# Task 2 

tree = KDTree(pcd_above_ground)
distances, indices = tree.query(pcd_above_ground, k=5)

k_distances = np.sort(distances[:,4])

# elbow plot
plt.figure(figsize=(10,5))
plt.plot(k_distances)
plt.title("Elbow plot - dataset1", fontsize=16)
plt.xlabel("Points sorted by distance", fontsize=12)
plt.ylabel("5-NN distance", fontsize=12)
plt.tight_layout()
plt.show()

# Simple automatic estimate of elbow
x = np.arange(len(k_distances))
start_point = np.array([0, k_distances[0]])
end_point = np.array([len(k_distances)-1, k_distances[-1]])

line_vec = end_point - start_point
line_vec = line_vec / np.linalg.norm(line_vec)

point_vecs = np.vstack((x, k_distances)).T - start_point
proj_lengths = point_vecs @ line_vec
proj_points = np.outer(proj_lengths, line_vec) + start_point

dist_to_line = np.linalg.norm(np.vstack((x, k_distances)).T - proj_points, axis=1)
elbow_index = np.argmax(dist_to_line)

optimal_eps = k_distances[elbow_index]
print("dataset1 optimal eps:", optimal_eps)

# Apply DBSCAN again with optimal eps
clustering = DBSCAN(eps = optimal_eps, min_samples=5).fit(pcd_above_ground)

clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, max(clusters,1))]

# Plotting resulting clusters with optimized eps
plt.figure(figsize=(10,10))
plt.scatter(pcd_above_ground[:,0], 
            pcd_above_ground[:,1],
            c=clustering.labels_,
            cmap=matplotlib.colors.ListedColormap(colors),
            s=2)

plt.title('DBSCAN with optimal eps: %d clusters' % clusters, fontsize=20)
plt.xlabel('x axis', fontsize=14)
plt.ylabel('y axis', fontsize=14)
plt.tight_layout()
plt.show()

#%% Dataset 2
# read file containing point cloud data
pcd = np.load("dataset2.npy")

pcd.shape

# remove ground plane Task 1

est_ground_level = get_ground_level(pcd)
print("dataset2 ground level:", est_ground_level)

pcd_above_ground = pcd[pcd[:,2] > est_ground_level]

pcd_above_ground.shape

# histogram plot for dataset2
plt.figure(figsize=(10,5))
plt.hist(pcd[:,2], bins=100, color='skyblue', edgecolor='black')
plt.axvline(est_ground_level, color='red', linestyle='--', label=f'Ground level = {est_ground_level:.3f}')
plt.title('Histogram of z-values - dataset2')
plt.xlabel('z value')
plt.ylabel('Number of points')
plt.legend()
plt.tight_layout()
plt.show()

unoptimal_eps = 10
# find the elbow
clustering = DBSCAN(eps = unoptimal_eps, min_samples=5).fit(pcd_above_ground)

clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, max(clusters,1))]

# Plotting resulting clusters
plt.figure(figsize=(10,10))
plt.scatter(pcd_above_ground[:,0], 
            pcd_above_ground[:,1],
            c=clustering.labels_,
            cmap=matplotlib.colors.ListedColormap(colors),
            s=2)


plt.title('DBSCAN: %d clusters' % clusters,fontsize=20)
plt.xlabel('x axis',fontsize=14)
plt.ylabel('y axis',fontsize=14)
plt.show()


#% Task 2

tree = KDTree(pcd_above_ground)
distances, indices = tree.query(pcd_above_ground, k=5)

k_distances = np.sort(distances[:,4])

# elbow plot
plt.figure(figsize=(10,5))
plt.plot(k_distances)
plt.title("Elbow plot - dataset2", fontsize=16)
plt.xlabel("Points sorted by distance", fontsize=12)
plt.ylabel("5-NN distance", fontsize=12)
plt.tight_layout()
plt.show()

# Simple automatic estimate of elbow
x = np.arange(len(k_distances))
start_point = np.array([0, k_distances[0]])
end_point = np.array([len(k_distances)-1, k_distances[-1]])

line_vec = end_point - start_point
line_vec = line_vec / np.linalg.norm(line_vec)

point_vecs = np.vstack((x, k_distances)).T - start_point
proj_lengths = point_vecs @ line_vec
proj_points = np.outer(proj_lengths, line_vec) + start_point

dist_to_line = np.linalg.norm(np.vstack((x, k_distances)).T - proj_points, axis=1)
elbow_index = np.argmax(dist_to_line)

optimal_eps = k_distances[elbow_index]
print("dataset2 optimal eps:", optimal_eps)

# Apply DBSCAN again with optimal eps
clustering = DBSCAN(eps = optimal_eps, min_samples=5).fit(pcd_above_ground)
clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, max(clusters,1))]

# Plotting resulting clusters with optimized eps
plt.figure(figsize=(10,10))
plt.scatter(pcd_above_ground[:,0], 
            pcd_above_ground[:,1],
            c=clustering.labels_,
            cmap=matplotlib.colors.ListedColormap(colors),
            s=2)

plt.title('DBSCAN with optimal eps: %d clusters' % clusters, fontsize=20)
plt.xlabel('x axis', fontsize=14)
plt.ylabel('y axis', fontsize=14)
plt.tight_layout()
plt.show()