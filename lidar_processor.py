#!/usr/bin/env python3
"""
LiDAR Data Processor for Mapping Automation Agent
Processes and visualizes USGS LiDAR .laz files for drone mapping applications
"""

from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from tqdm import tqdm
import textwrap

# Optional: set style after imports
plt.style.use('dark_background')

DATASET_NAME = "TX23_Nelson"

SOURCE_TEXT = """
        Data Source:

        Nelson, M. (2024). 
        Characterizing Alluvial Channel Bank Erosion with Lidar, TX 2023. 
        National Center for Airborne Laser Mapping (NCALM). 
        Distributed by OpenTopography. 
        Survey Date: 09/19/2023
        https://doi.org/10.5069/G9DZ06JZ. 
        
        Accessed 2025-07-29
        """

class LiDARProcessor:
    """Main class for processing LiDAR data for drone mapping applications"""
    
    def __init__(self, data_dir=DATASET_NAME, output_dir="output_files", max_files=10):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.max_files = max_files
        self.processed_data = {}
        
    def get_local_laz_files(self):
        """Load local .laz files from the data directory"""
        laz_files = sorted(self.data_dir.glob("*.laz"))
        if self.max_files:
            laz_files = laz_files[:self.max_files]
        return laz_files
    
    def analyze_laz_file(self, filepath):
        """Analyze a single .laz file and extract key information"""
            
        try:
            las = laspy.read(filepath)
            
            info = {
                'filename': filepath.name,
                'num_points': len(las.points),
                'x_min': las.header.x_min,
                'x_max': las.header.x_max,
                'y_min': las.header.y_min,
                'y_max': las.header.y_max,
                'z_min': las.header.z_min,
                'z_max': las.header.z_max,
                'point_format': las.header.point_format,
                'scale_x': las.header.x_scale,
                'scale_y': las.header.y_scale,
                'scale_z': las.header.z_scale,
            }
            
            # Extract point data (sample for large files)
            sample_size = min(100000, len(las.points))
            indices = np.random.choice(len(las.points), sample_size, replace=False)
            
            points_data = {
                'x': las.x[indices],
                'y': las.y[indices],
                'z': las.z[indices],
                'intensity': las.intensity[indices] if hasattr(las, 'intensity') else None,
                'classification': las.classification[indices] if hasattr(las, 'classification') else None,
                'return_number': las.return_number[indices] if hasattr(las, 'return_number') else None,
            }
            
            info['points_sample'] = points_data
            return info
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None
    
    def process_all_files(self, filepaths):
        """Process all downloaded .laz files"""
        print(f"\nProcessing {len(filepaths)} LiDAR files...")
        
        for filepath in tqdm(filepaths, desc="Processing"):
            info = self.analyze_laz_file(filepath)
            if info:
                self.processed_data[filepath.stem] = info
                
        return self.processed_data
    
    def create_overview_visualization(self):
        """Create comprehensive visualization of the LiDAR dataset"""
        if not self.processed_data:
            print("No processed data available for visualization")
            return
            
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Dataset Overview Statistics
        ax1 = plt.subplot(3, 3, 1)
        file_info = []
        for name, data in self.processed_data.items():
            file_info.append({
                'File': name,
                'Points (M)': data['num_points'] / 1e6,
                'Area (km²)': ((data['x_max'] - data['x_min']) * (data['y_max'] - data['y_min'])) / 1e6,
                'Elevation Range (m)': data['z_max'] - data['z_min']
            })
        
        df_info = pd.DataFrame(file_info)
        bars = ax1.bar(range(len(df_info)), df_info['Points (M)'], color='#52BCFF', edgecolor='black')

        total_files = len(self.processed_data)
        total_points_m = sum(data['num_points'] for data in self.processed_data.values()) / 1e9
        avg_points_m = np.mean([data['num_points'] for data in self.processed_data.values()]) / 1e6

        ax1.set_title(
            f"Points (pts) per File\nFiles: {total_files:,.0f}, ∑ pts: {total_points_m:,.1f}B, Ø pts: {avg_points_m:,.1f}M per file",
            fontsize=12, fontweight='bold'
        );
        
        ax1.set_xlabel('File Index')
        ax1.set_ylabel('Points (Millions)')
        
        # 2. Coverage Map
        ax2 = plt.subplot(3, 3, 2)
        for name, data in self.processed_data.items():
            # Draw bounding boxes
            x_coords = [data['x_min'], data['x_max'], data['x_max'], data['x_min'], data['x_min']]
            y_coords = [data['y_min'], data['y_min'], data['y_max'], data['y_max'], data['y_min']]
            ax2.plot(x_coords, y_coords, alpha=0.7, linewidth=2)
        
        total_area = sum(
            ((d['x_max'] - d['x_min']) * (d['y_max'] - d['y_min'])) / 1e6
            for d in self.processed_data.values()
        )

        ax2.set_title(f'Spatial Coverage\nTotal Area: {total_area:,.2f} km²',
                    fontsize=12, fontweight='bold')
            
        ax2.set_xlabel('Longitude (UTM)')
        ax2.set_ylabel('Latitude (UTM)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Elevation Distribution
        ax3 = plt.subplot(3, 3, 3)
        all_elevations = []
        for data in self.processed_data.values():
            if data['points_sample']['z'] is not None:
                all_elevations.extend(data['points_sample']['z'])

        if all_elevations:
            min_elev = min(data['z_min'] for data in self.processed_data.values())
            max_elev = max(data['z_max'] for data in self.processed_data.values())

            ax3.hist(all_elevations, bins=50, alpha=0.7, color='#94e047', edgecolor='black')
            ax3.set_title(
                f"Elevation Distribution\nMin: {min_elev:,.1f} m, Max: {max_elev:,.1f} m",
                fontsize=12, fontweight='bold'
            )
            ax3.set_xlabel('Elevation (m)')
            ax3.set_ylabel('Frequency')
        
        # 4. 3D Terrain Sample
        ax4 = plt.subplot(3, 3, 4, projection='3d')

        all_x, all_y, all_z = [], [], []

        for data in self.processed_data.values():
            pts = data['points_sample']
            if pts['x'] is not None:
                all_x.extend(pts['x'])
                all_y.extend(pts['y'])
                all_z.extend(pts['z'])

        # Subsample to avoid overplotting
        if all_x:
            all_x = np.array(all_x)
            all_y = np.array(all_y)
            all_z = np.array(all_z)

            total_points = len(all_x)

            # Remove extreme percentiles BEFORE sampling
            z_min, z_max = np.percentile(all_z, [0.05, 99.5])
            mask = (all_z >= z_min) & (all_z <= z_max)

            filtered_x = all_x[mask]
            filtered_y = all_y[mask]
            filtered_z = all_z[mask]

            # Sample from the filtered data
            sample_size = min(200_000, len(filtered_x))
            idx = np.random.choice(len(filtered_x), sample_size, replace=False)

            ax4.scatter(filtered_x[idx], filtered_y[idx], filtered_z[idx],
                        c=filtered_z[idx], cmap='terrain', s=0.5)
            
            percent = (sample_size / total_points) * 100

            ax4.set_title(f'3D Terrain Sample\n{sample_size:,} Points ({percent:.2f}%)',
                        fontsize=12, fontweight='bold')
            ax4.set_xlabel('X (UTM)')
            ax4.set_ylabel('Y (UTM)')
            ax4.set_zlabel('Elevation (m)')
        
        # 5. Classification Analysis (if available)
        ax5 = plt.subplot(3, 3, 5)
        all_classifications = []
        for data in self.processed_data.values():
            if data['points_sample']['classification'] is not None:
                all_classifications.extend(data['points_sample']['classification'])

        if all_classifications:
            unique_classes, counts = np.unique(all_classifications, return_counts=True)
            x_positions = np.arange(len(unique_classes))  # categorical x positions

            bars = ax5.bar(x_positions, counts, align='center', color='#ffce00', edgecolor='black')
            ax5.bar_label(bars, labels=[f'{c:,}' for c in counts], fontsize=10)  
            ax5.set_xticks(x_positions)
            ax5.set_xticklabels(unique_classes)

            ax5.set_title(f'Point Classifications\n1 - Unclassified, 2 - Ground, 7 - Low Point, 18 - High Noise',
                          fontsize=12, fontweight='bold')
            ax5.set_xlabel('Classification Code')
            ax5.set_ylabel('Count')
            ymax = counts.max() * 1.1  # 10% higher than max count
            ax5.set_ylim(0, ymax)
        
        # 6. Intensity Distribution (if available)
        ax6 = plt.subplot(3, 3, 6)
        all_intensities = []

        for data in self.processed_data.values():
            if data['points_sample']['intensity'] is not None:
                all_intensities.extend(data['points_sample']['intensity'])

        if all_intensities:
            min_intensity = np.min(all_intensities)
            max_intensity = np.max(all_intensities)
            
            ax6.hist(all_intensities, bins=50, alpha=0.7, color='#ea2081', edgecolor='black')
            ax6.set_title(f'Intensity Distribution\nMin: {min_intensity:,.0f}, Max: {max_intensity:,.0f}', 
                        fontsize=12, fontweight='bold')
            ax6.set_xlabel('Intensity')
            ax6.set_ylabel('Frequency')
        else:
            ax6.text(0.5, 0.5, 'No intensity data available', ha='center', va='center', fontsize=12, color='red')

        
        # 7. Elevation Map 
        ax7 = plt.subplot(3, 3, 7)

        all_x = []
        all_y = []
        all_z = []

        for tile_data in self.processed_data.values():
            pts = tile_data['points_sample']
            if pts['x'] is not None:
                all_x.extend(pts['x'])
                all_y.extend(pts['y'])
                all_z.extend(pts['z'])

        if all_x:
            all_x = np.array(all_x)
            all_y = np.array(all_y)
            all_z = np.array(all_z)
            
            vmin = np.percentile(all_z, 0.5)
            vmax = np.percentile(all_z, 99.5)

            sc = ax7.scatter(all_x, all_y, c=all_z, cmap='terrain', s=0.1, vmin=vmin, vmax=vmax)

            ax7.set_title("Elevation Map", fontsize=12, fontweight='bold')
            ax7.axis('equal')
            ax7.axis('off')
            fig.colorbar(sc, ax=ax7, shrink=0.7, label="Elevation (m)")

        # 8. DEM Visualization
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')

        dem_path = self.output_dir / f"{DATASET_NAME}_terrain_map.tif"

        if dem_path.exists():
            with rasterio.open(dem_path) as src:
                elevation = src.read(1)
                elevation_masked = np.ma.masked_invalid(elevation)  # Mask NaNs properly

                if elevation_masked.count() == 0:
                    ax8.text(0.1, 0.5, "No valid elevation data", fontsize=12, color="red")
                else:
                    vmin = np.percentile(elevation_masked.compressed(), 0.5)
                    vmax = np.percentile(elevation_masked.compressed(), 99.5)

                    img = ax8.imshow(elevation_masked, cmap='terrain', vmin=vmin, vmax=vmax)
                    ax8.set_title('Digital Elevation Model', fontsize=12, fontweight='bold')
                    plt.colorbar(img, ax=ax8, shrink=0.7, label="Elevation (m)")
        else:
            ax8.text(0.1, 0.5, "DEM not available", fontsize=12, color="red")

        # 9. Data Source Information
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        wrapped_lines = [
            textwrap.fill(line.strip(), width=60) if line.strip() else ""
            for line in SOURCE_TEXT.strip().splitlines()
        ]
        wrapped_text = "\n".join(wrapped_lines)

        ax9.text(0.05, 0.95, wrapped_text, transform=ax9.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='square,pad=0.8', facecolor='#8b76e9', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{DATASET_NAME}_lidar_analysis.png', dpi=300, bbox_inches='tight')

        plt.show()
        
        return fig
    
    def generate_ai_training_data(self):
        """Generate processed data suitable for AI training"""
        if not self.processed_data:
            return None
            
        training_data = {}
        
        for name, data in self.processed_data.items():
            points = data['points_sample']
            if points['x'] is not None:
                # Create structured training data
                training_data[name] = {
                    'point_cloud': np.column_stack((points['x'], points['y'], points['z'])),
                    'features': {
                        'elevation_stats': {
                            'min': np.min(points['z']),
                            'max': np.max(points['z']),
                            'mean': np.mean(points['z']),
                            'std': np.std(points['z'])
                        },
                        'spatial_extent': {
                            'x_range': data['x_max'] - data['x_min'],
                            'y_range': data['y_max'] - data['y_min'],
                            'area': (data['x_max'] - data['x_min']) * (data['y_max'] - data['y_min'])
                        },
                        'point_density': data['num_points'] / ((data['x_max'] - data['x_min']) * (data['y_max'] - data['y_min'])),
                        'terrain_roughness': np.std(points['z']),
                        'classification': points['classification'],
                        'intensity': points['intensity']
                    }
                }
        
        # Save training data
        np.save(self.output_dir / f'{DATASET_NAME}_processed_data.npy', training_data)
        print(f"AI training data saved to {self.output_dir / f'{DATASET_NAME}_processed_data.npy'}")
        
        return training_data
    
    def get_utm_epsg(self, x, y):
        """Estimate UTM EPSG code from approximate lat/lon"""
        lon = x / 111320  # rough meter-to-degree
        lat = y / 110540
        zone = int((lon + 180) / 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone  # 326xx for north, 327xx for south
        return epsg

    def generate_dem_from_points(self, output_file="dem.tif", resolution=1.0):
        """Generate a DEM from LiDAR point clouds and export to GeoTIFF"""
        all_x, all_y, all_z = [], [], []

        for data in self.processed_data.values():
            if data['points_sample']['x'] is None:
                continue
            all_x.extend(data['points_sample']['x'])
            all_y.extend(data['points_sample']['y'])
            all_z.extend(data['points_sample']['z'])

        if not all_x:
            print("❌ No point data available for DEM generation.")
            return

        print("🛠️ Generating DEM...")

        all_x = np.array(all_x)
        all_y = np.array(all_y)
        all_z = np.array(all_z)

        # Define grid
        xi = np.arange(all_x.min(), all_x.max(), resolution)
        yi = np.arange(all_y.min(), all_y.max(), resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        # Interpolate Z values onto grid
        zi_grid = griddata(
            (all_x, all_y),
            all_z,
            (xi_grid, yi_grid),
            method='linear'
        )

        # Infer EPSG code
        epsg_code = self.get_utm_epsg(np.mean(all_x), np.mean(all_y))
        crs = CRS.from_epsg(epsg_code)

        # Rasterio transform
        transform = from_origin(
            west=xi.min(),
            north=yi.max(),
            xsize=resolution,
            ysize=resolution
        )

        # Save GeoTIFF
        output_path = self.output_dir / output_file
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=zi_grid.shape[0],
            width=zi_grid.shape[1],
            count=1,
            dtype=zi_grid.dtype,
            crs=crs,
            transform=transform,
            nodata=np.nan
        ) as dst:
            dst.write(zi_grid, 1)

        print(f"✅ DEM saved: {output_path}")

def main():
    """Main execution function"""
    print("🚁 AI Mapping Agent – LiDAR Data Processor")
    print("=" * 50)
    
    # Initialize processor (points to local 'data' directory)
    processor = LiDARProcessor(data_dir=DATASET_NAME, max_files=None)
    
    # Step 1: Load local files
    print("\n📂 Step 1: Loading local LiDAR files...")
    laz_files = processor.get_local_laz_files()
    
    if not laz_files:
        print("❌ No .laz files found")
        return
    
    print(f"✅ Found {len(laz_files)} .laz files")
    
    # Step 2: Process files
    print("\n🔄 Step 2: Processing LiDAR data...")
    processed_data = processor.process_all_files(laz_files)
    print(f"✅ Processed {len(processed_data)} files")
    
    # Step 3: Generate AI training data
    print("\n🤖 Step 3: Generating AI training data...")
    training_data = processor.generate_ai_training_data()
    
    # Step 4: Generate DEM map
    print("\n🗺️ Step 4: Generating Digital Elevation Model (DEM)...")
    processor.generate_dem_from_points(output_file=f"{DATASET_NAME}_terrain_map.tif", resolution=5.0)
    
    # Step 5: Create visualizations
    print("\n📊 Step 5: Creating visualizations...")
    processor.create_overview_visualization()
    
    print("\n✅ Analysis complete! Check the generated files:")
    print(f"   - Visualization: {processor.output_dir}/{DATASET_NAME}_lidar_analysis.png")
    print(f"   - Training data: {processor.output_dir}/{DATASET_NAME}_processed_data.npy")


if __name__ == "__main__":
    main()