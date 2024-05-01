# """
# Code reference: http://www.open3d.org/docs/latest/tutorial/Advanced/voxelization.html
# visualize the bin format to voxel purpose, and maybe video teaser
# """

# # print('input')
# # N = 2000
# # pcd = o3dtut.get_armadillo_mesh().sample_points_poisson_disk(N)
# # # fit to unit cube
# # pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
# #           center=pcd.get_center())
# # pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
# # o3d.visualization.draw_geometries([pcd])

# # print('voxelization')
# # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
# #                                                             voxel_size=0.05)
# # o3d.visualization.draw_geometries([voxel_grid])

# # input

# import numpy as np
# import open3d as o3d



# def visualize_occupancy_voxel_array(occupancy_array):
#   """Visualizes a NumPy array with boolean values representing occupancy values.

#   Args:
#     occupancy_array: The NumPy array with boolean values representing occupancy values.

#   Returns:
#     An Open3D point cloud.
#   """

#   # Create a new Open3D point cloud.
#   open3d_point_cloud = o3d.geometry.PointCloud()

#   # Set the point cloud's fields.
#   open3d_point_cloud.points = o3d.utility.Vector3dVector(np.where(occupancy_array)[1:])
#   print(np.where(occupancy_array)[1:])

#   # Set the point cloud's colors.
#   open3d_point_cloud.colors = o3d.utility.Vector3dVector(np.full((open3d_point_cloud.points.shape[0], 3), [0, 255, 0]))

#   return open3d_point_cloud


# # Load the NumPy array.
# print("__give input npy file")
# input_str = input()
# occupancy_array = np.load(input_str)

# # Visualize the NumPy array.
# open3d_point_cloud = visualize_occupancy_voxel_array(occupancy_array)

# # Create a visualization window.
# vis = o3d.visualization.Visualizer()

# # Add the point cloud to the visualization window.
# vis.add_geometry(open3d_point_cloud)

# # Show the visualization window.
# vis.run()