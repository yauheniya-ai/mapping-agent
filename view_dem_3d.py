import numpy as np
import open3d as o3d
import rasterio

with rasterio.open("output_files/TX23_Nelson_terrain_map.tif") as src:
    z = src.read(1)
    rows, cols = np.indices(z.shape)
    x = cols * src.transform[0] + src.transform[2]
    y = rows * src.transform[4] + src.transform[5]
    mask = ~np.isnan(z)

    # Get percentile thresholds
    vmin = np.percentile(z[mask], 0.5)
    vmax = np.percentile(z[mask], 99.5)

    # Filter points within percentile range
    valid = (z >= vmin) & (z <= vmax) & mask

    points = np.column_stack((x[valid], y[valid], z[valid]))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([pcd], window_name="DEM 3D Viewer")
