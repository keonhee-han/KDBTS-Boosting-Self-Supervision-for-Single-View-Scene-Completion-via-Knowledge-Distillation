# import numpy as np
# import open3d as o3d


# # Read the binary file | label file
# print("specify: /path/to/your/pointcloud.bin")
# path = input()
# points = np.fromfile(f"{path}", dtype=np.float32).reshape(-1, 4)

# print("specify correspodning label file if existed. type z if none.")
# path_label = input()
# if path_label != "z":
#     labels = np.fromfile(f"{path_label}", dtype=np.uint32)
    
#     # Create a colormap (replace this with your actual colormap)
#     colormap = {
#         0: [1, 0, 0],  # Red for label 0
#         1: [0, 1, 0],  # Green for label 1
#         2: [0, 0, 1],  # Blue for label 2
#         # Add more colors as needed
#     }
    
#     # Assign each point a color based on its label
#     colors = []
#     for label in labels:
#         if label not in colormap:
#             # If label is not in colormap, assign a random color
#             colormap[label] = np.random.rand(3).tolist()
#         colors.append(colormap[label])
#     colors = np.array(colors)

#     # Assign each point a color based on its label
#     #colors = np.array([colormap[label] for label in labels])
# pcd = o3d.geometry.PointCloud()
# # Create a point cloud from the points
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # use only the first 3 columns (X, Y, Z)
# pcd.colors = o3d.utility.Vector3dVector(colors)

# print("specify correspodning PCD file to compared with. e.g. /path/to/your/predicted.pcd type z if none.")
# path_pcd = input()
# if path_pcd != "z":
#     # Load the predicted point cloud from the PCD file
#     predicted_pcd_path = path_pcd
#     predicted_pcd = o3d.io.read_point_cloud(predicted_pcd_path)

#     # Extract points and colors from the PCD file
#     points = np.asarray(predicted_pcd.points)
#     colors = np.asarray(predicted_pcd.colors)

#     # Flatten colors to a single dimension
#     colors_flattened = colors.flatten()


# # Visualize the point cloud
# o3d.visualization.draw_geometries([pcd])