LiDAR Point Cloud Processing – Assignment 5

This project analyzes LiDAR point cloud data using Python.
Each point in the dataset represents a 3D coordinate:  (x, y, z)

where:
x – horizontal position
y – horizontal position
z – height

The goal of this assignment is to process LiDAR data and perform clustering using the DBSCAN algorithm.

Two datasets were provided:
1. dataset1.npy
2. dataset2.npy

The tasks include estimating the ground level, removing ground points, and determining the optimal epsilon parameter for DBSCAN clustering.

Task 1 – Ground Level Estimation

The first task was to estimate the ground level of the point cloud.
Since the ground usually contains the largest number of points, the ground level can be estimated using a histogram of the z-values. The bin with the highest frequency represents the most common height in the dataset, which corresponds to the ground level.
The ground level was calculated using the NumPy function:
np.histogram()

- Dataset 1

Estimated ground level: 64.32

Histogram of z-values:
<img width="1425" height="705" alt="image" src="https://github.com/user-attachments/assets/0417cc7a-aceb-46cf-9b7d-79e3c2a624b5" />

- Dataset 2

Estimated ground level: 63.91

Histogram of z-values:
<img width="1425" height="705" alt="image" src="https://github.com/user-attachments/assets/6d3b4836-3e16-4fa1-8075-5c5c4ec8280c" />


Task 2 – DBSCAN Epsilon Optimization

The second task was to determine the optimal epsilon value (eps) for the DBSCAN clustering algorithm.
DBSCAN requires two main parameters:
1- eps – maximum distance between two points for them to be considered neighbors
2- min_samples – minimum number of points required to form a cluster

To determine a suitable value for eps, the k-distance method was used.
Steps used in the method:

- Compute the distance to the 5th nearest neighbor for each point using a KDTree.
- Sort the distances.
- Plot the k-distance graph.
- Identify the elbow point in the graph, which represents a suitable eps value.

Dataset 1

Optimal eps value: 1.82

Elbow plot:
<img width="1425" height="705" alt="image" src="https://github.com/user-attachments/assets/07d1aaef-e35e-4149-a816-d0d29520537d" />

Cluster visualization:
<img width="1233" height="1249" alt="image" src="https://github.com/user-attachments/assets/06548f57-3d24-4a6b-bc82-d4a7c08f8d4c" />

- Dataset 2

Optimal eps value: 1.76

Elbow plot:
<img width="1425" height="705" alt="image" src="https://github.com/user-attachments/assets/7c17bbb5-aad9-400e-8dda-b67a19c9b096" />

Cluster visualization:
<img width="1220" height="1249" alt="image" src="https://github.com/user-attachments/assets/d7398b45-52bc-46af-9de1-edd403d3ee4a" />

Tools and Libraries

The following Python libraries were used in this project:

1- NumPy – numerical computation and data handling

2- Matplotlib – visualization and plotting

3- SciPy (KDTree) – nearest neighbor search

4- Scikit-learn (DBSCAN) – clustering algorithm


Summary

In this assignment:

The ground level was estimated using histogram analysis of the z-values.

Ground points were removed from the datasets.

DBSCAN clustering was applied to the remaining points.

The optimal epsilon parameter was determined using the elbow method.

This workflow allows efficient processing and clustering of LiDAR point cloud data.
