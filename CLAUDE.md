# Facial Asymmetry Research Project

## Project Overview
This project focuses on 3D facial mesh deformation and symmetry analysis. The main goal is to generate controlled facial asymmetries for research purposes using various mathematical displacement methods, and to analyze real patient data from multiple datasets (bijian, kedian, ppdh, headspace).

## Environment Setup
- **Python Version**: 3.9.7 (Conda environment)
- **Key Dependencies**:
  - `open3d==0.16.1` - 3D mesh processing and visualization
  - `trimesh==4.10.1` - Alternative mesh tools
  - `numpy==1.22.4` - Numerical computation (downgraded for compatibility)
  - `scipy` - Spatial algorithms (KDTree)
  - `matplotlib` - Plotting and analysis
  - `pandas` - Data manipulation
  - `openpyxl` - Excel file generation

## Core Files

### Input Files
- **`Template.obj`** - Original 3D facial mesh template
  - 7,160 vertices
  - 14,050 triangular faces
  - Perfectly symmetric about x=0 plane (left-right symmetry)

- **`template landmark.csv`** - 5 landmark points on the face
  - Format: `x,y,z` (one point per line)
  - Landmark 2 (index 2): Nose tip at [0.0, 15.934288, 54.339256]

- **`region.csv`** - 7 landmark points for face region partition
  - Used for bijian/kedian datasets (coordinate-based partition)

- **`region_labels.txt`** - Pre-computed region labels by vertex ID
  - Format: `vertex_index,region_label` (one per line)
  - Used for ppdh/headspace datasets (ID-based partition)
  - Region distribution: R1=850, R2=667, R3=2318, R4=760, R5=400, R6=2165

- **`vertex_mapping.npz`** - Vertex index mapping between Open3D and original OBJ order
  - Open3D reorders vertices when loading OBJ files
  - Contains `o3d_to_orig` and `orig_to_o3d` arrays

### Analysis Scripts

#### `view_template.py`
- **Purpose**: Interactive 3D visualization
- **Features**:
  - Load and display OBJ mesh
  - Overlay landmark points in red
  - Supports rotation, zoom, pan
- **Usage**: `python view_template.py`

#### `check_symmetry.py`
- **Purpose**: Symmetry verification and pairing generation
- **Key Functions**:
  - `analyze_symmetry(obj_path, plane='yz')` - Analyze mirror symmetry
  - `generate_symmetry_pairs(obj_path, plane='yz', output_csv='pairs.csv', tolerance=0.01)` - Generate vertex pair correspondences
- **Results**:
  - Template.obj is 100% symmetric about x=0 plane
  - Mean mirror distance: 0.000001 mm
- **Output**:
  - `symmetry_analysis.pdf` - Distance distribution histogram
  - `pairs.csv` - 3,633 symmetric vertex pairs (all 7,160 vertices)

#### `compute_normals.py`
- **Purpose**: Demonstrate vertex normal calculation
- **Key Functions**:
  - `compute_face_normal(v1, v2, v3)` - Calculate face normal using cross product
  - `compute_vertex_normals_manual(mesh)` - Average adjacent face normals
  - `get_landmark_normal(obj_path, csv_path, landmark_idx)` - Get normal at specific landmark
- **Example**: Landmark 2 normal = [0.0, 0.114726, 0.993397]

#### `verify_displacement_symmetry.py`
- **Purpose**: Compare symmetry before/after displacement
- **Usage**: Analyze why biased Gaussian method creates asymmetry
- **Findings**: Biased kernel causes 0.032-0.049mm X-direction asymmetry

#### `mirror_registration.py`
- **Purpose**: Self-registration through mirroring
- **Method**:
  - Mirror Template.obj across x=0 plane
  - Use ICP to register mirrored → original
  - Output vertex correspondences
- **Output**: `registration_mirror/mirror_pairs.csv`

#### `standard_icp.py`
- **Purpose**: Standard ICP registration for asymmetry analysis
- **Method**:
  - Mirror the deformed mesh across x=0 plane
  - Apply ICP to align mirrored mesh to original deformed mesh
  - Output vertex correspondences with distances
- **Output**: CSV files with columns: `source_id,target_id,distance`
- **Usage**: Batch processing of all deformed meshes

#### `visualize_asymmetry.py`
- **Purpose**: Visualize facial asymmetry using heat maps
- **Method**:
  - Compare original mesh with mirrored+registered mesh
  - Calculate point-to-point distances using KDTree
  - Generate colored heat map (distance-based coloring)
- **Features**:
  - Interactive 3D viewer (optional)
  - PDF report with heat map and statistical analysis
  - MSE and RMSE calculation
  - Uses Open3D rendering for consistent visualization
  - **Summary statistics** with region breakdown (R1-R6)
- **Output**: Multi-page PDF with heat map and distribution plots
- **Key Metrics**: count, Mean, Median, Std, Min, Max, RMSE, pct_within
- **Summary Function**: `save_summary_statistics()` generates CSV with overall and per-region stats

#### `visualize_pairing_quality.py`
- **Purpose**: Visualize ICP pairing quality from CSV results (Ground Truth method)
- **Method**:
  - Load pairing results from ICP output CSV
  - Color vertices by pairing distance
  - Generate heat map and statistical plots
- **Features**:
  - Interactive 3D viewer
  - PDF report with heat map and analysis
  - MSE and RMSE calculation
  - Handles missing vertex pairs (shown in gray)
  - **Summary statistics** with region breakdown (R1-R6)
- **Input**: OBJ file + CSV file (from ICP registration)
- **Output**: Multi-page PDF with visualization and statistics
- **Key Metrics**: Valid pairs count, Mean, Median, Std, Min, Max, MSE, RMSE

#### `face_region_partition.py`
- **Purpose**: Partition face mesh into 6 regions based on landmarks
- **Key Functions**:
  - `load_landmarks_from_csv(csv_path)` - Load 7 landmark points
  - `partition_face(obj_path, landmarks, reorder=False)` - Assign region labels to vertices
- **Regions**:
  - R1: Forehead (y > v1.y) - 850 vertices
  - R2: Upper Cheeks - 667 vertices
  - R3: Lower Cheeks - 2318 vertices
  - R4: Upper Nose Bridge - 760 vertices
  - R5: Lower Nose Bridge - 400 vertices
  - R6: Chin (y < v5.y) - 2165 vertices
- **Note**: `reorder` parameter converts Open3D vertex order to original OBJ order

#### `vertex_reorder.py`
- **Purpose**: Handle vertex index mapping between Open3D and original OBJ order
- **Key Functions**:
  - `load_mapping(npz_path)` - Load vertex mapping from npz file
  - `reorder_values_o3d_to_orig(values, o3d_to_orig)` - Convert O3D order to original
  - `reorder_values_orig_to_o3d(values, orig_to_o3d)` - Convert original to O3D order

#### `batch_asymmetry_analysis.py`
- **Purpose**: Batch process ppdh/headspace datasets with heatmaps and statistics
- **Method**:
  - Read OBJ files from `mapped_templates/` subdirectory
  - Read FA results from `fa_results/` subdirectory (CSV with vertex distances)
  - Use `region_labels.txt` for ID-based region partition (not coordinate-based)
  - Generate heatmap PDFs with cutoff_distance=0.5mm
  - Calculate overall and per-region statistics
- **Key Functions**:
  - `load_region_labels_from_file(path)` - Load pre-computed region labels
  - `process_single_pair(obj_path, csv_path, region_labels, ...)` - Process one file
  - `batch_process_directory(input_dir, region_labels_path, output_dir, ...)` - Batch process
- **Usage**: `python batch_asymmetry_analysis.py --input_dirs output_ppdh output_headspace`
- **Output**:
  - `analysis_output_ppdh_mapped_templates/` - heatmap PDFs and summary CSV
  - `analysis_output_headspace_mapped_templates/` - heatmap PDFs and summary CSV

#### `add_menton_distance.py`
- **Purpose**: Calculate menton point distance to midsagittal plane
- **Method**:
  - Midsagittal plane defined by 3 vertex indices: 3, 91, 3664 (found in Template.obj)
  - For each OBJ file, get these 3 vertices' coordinates to define the plane
  - Calculate signed distance from menton point to the plane
  - Menton coordinates from external CSV files
- **Data Sources**:
  - headspace: `/Users/lqy/Desktop/research/lele/summary.csv` ('new sequence' column)
  - ppdh: `/Users/lqy/Desktop/research/lele/asymmetry_ppdh_folders.csv` ('FolderName' column)
- **Output**: Adds `Menton_Distance` column to summary_statistics.csv
- **Usage**: `python add_menton_distance.py`

#### `merge_summary_to_excel.py`
- **Purpose**: Merge all summary CSV files into one Excel workbook
- **Input**: Summary CSVs from bijian, kedian, ppdh, headspace (GT, ICP, MeshMonk methods)
- **Output**: `summary_statistics_all.xlsx` with 9 sheets:
  - All_Results (combined)
  - bijian_GT, bijian_ICP, bijian_MeshMonk
  - kedian_GT, kedian_ICP, kedian_MeshMonk
  - ppdh_MeshMonk, headspace_MeshMonk
- **Usage**: `python merge_summary_to_excel.py`

### Deformation Scripts

All deformation scripts use:
- **Center**: Landmark 2 (nose tip)
- **Direction**: +X axis (1, 0, 0) - rightward displacement
- **Sigma**: +1 (positive displacement)

#### `gaussian_displacement.py`
Implements two Gaussian-weighted deformation methods:

**Method 1: Normal Displacement (法向位移)**
- Each vertex moves along its surface normal
- Formula: `Δv_i = σ * A * w_i * n_i`
- Gaussian weight: `w_i = exp(-0.5 * (d_i/r)²)`
- Biased kernel: `c* = c + u_0 * e_hat` where `u_0 = k * r`
- **Parameters**:
  - `A`: Amplitude (0, 1, 2, 3, 4, 6 mm)
  - `r`: Kernel radius (10, 15, 20, 25 mm)
  - `k`: Bias ratio (0.1, 0.15, 0.2, 0.25, 0.3)
- **Output**: `displaced_normal/A_r_k.obj` (120 files total)
- **Note**: Creates slight asymmetry due to biased kernel

**Method 2: Directional Displacement (方向性位移)**
- All vertices move in fixed direction (1, 0, 0)
- Formula: `Δv_i = σ * w_i * (r*k) * direction`
- Displacement magnitude: `r * k`
- No biased kernel: `c* = c` (maintains perfect symmetry)
- **Parameters**:
  - `r`: Kernel radius (10, 15, 20, 25 mm)
  - `k`: Displacement ratio (0.1, 0.15, 0.2, 0.25, 0.3)
- **Output**: `displaced_directional/r_k.obj` (20 files total)

**Functions**:
- `gaussian_normal_displacement()` - Method 1
- `gaussian_directional_displacement()` - Method 2
- `batch_generate()` - Batch generation for Method 1
- `batch_generate_directional()` - Batch generation for Method 2

#### `rbf_wendland_displacement.py`
Implements RBF-based deformation using Wendland C2 basis function:

- **Weight function**: `w(r) = (1-r)⁴ * (4r+1)` for r ∈ [0,1]
- **Properties**: C² continuous (smoother than Gaussian)
- **Formula**: `Δv_i = σ * w_i * (r*k) * direction`
- **Parameters**:
  - `r`: Influence radius (10, 15, 20, 25 mm)
  - `k`: Displacement ratio (0.1, 0.15, 0.2, 0.25, 0.3)
- **Output**: `displaced_rbf/r_k.obj` (20 files total)

**Functions**:
- `rbf_wendland_displacement()` - Core deformation
- `batch_generate_rbf()` - Batch generation

## Generated Data Structure

```
facial-asymmetry/
├── Template.obj                    # Original mesh
├── template landmark.csv           # Landmark coordinates
├── region.csv                      # 7 landmarks for region partition
├── region_labels.txt               # Pre-computed region labels (ID-based)
├── vertex_mapping.npz              # Open3D <-> OBJ vertex mapping
├── pairs.csv                       # Symmetric vertex pairs (3,633 pairs)
├── symmetry_analysis.pdf           # Symmetry verification plot
├── summary_statistics_all.xlsx     # Combined Excel with all summary stats
│
├── displaced_normal/               # Gaussian normal displacement
│   └── A_r_k.obj                  # 120 files (6×4×5 combinations)
├── displaced_directional/          # Gaussian directional displacement
│   └── r_k.obj                    # 20 files (4×5 combinations)
├── displaced_rbf/                  # RBF Wendland displacement
│   └── r_k.obj                    # 20 files (4×5 combinations)
│
├── bijian/                         # Bijian dataset
│   ├── ground_truth/              # Ground truth analysis
│   │   └── ground_truth_summary_statistics.csv
│   ├── icp_result/                # ICP method results
│   │   └── icp_summary_statistics.csv
│   └── meshmonk/                  # MeshMonk method results
│       └── meshmonk_summary_statistics.csv
│
├── kedian/                         # Kedian dataset (same structure as bijian)
│   ├── ground_truth/
│   ├── icp_result/
│   └── meshmonk/
│
├── output_ppdh/                    # PPDH dataset input
│   ├── mapped_templates/          # OBJ files
│   └── fa_results/                # FA CSV files
├── analysis_output_ppdh_mapped_templates/  # PPDH analysis output
│   ├── *_mapped_asymmetry.pdf     # Heatmap PDFs
│   ├── *_stats.txt                # Individual stats
│   └── summary_statistics.csv     # With Menton_Distance column
│
├── output_headspace/               # Headspace dataset input
│   ├── mapped_templates/
│   └── fa_results/
└── analysis_output_headspace_mapped_templates/  # Headspace analysis output
    ├── *_mapped_asymmetry.pdf
    ├── *_stats.txt
    └── summary_statistics.csv     # With Menton_Distance column
```

## Key Concepts

### Coordinate System
- **X-axis**: Left-right (negative = left side, positive = right side)
- **Y-axis**: Up-down
- **Z-axis**: Front-back
- **Symmetry plane**: x=0 (YZ plane)

### Deformation Parameters
- **r (radius/kernel)**: Size of influence region (mm)
- **k (ratio)**: Controls displacement magnitude
- **A (amplitude)**: Direct displacement scale for normal method (mm)
- **σ (sigma)**: Direction sign (+1 = outward/rightward, -1 = inward/leftward)

### File Naming Convention
- Normal displacement: `A_r_k.obj` (e.g., `2_15_0.2.obj` = A=2mm, r=15mm, k=0.2)
- Directional/RBF: `r_k.obj` (e.g., `20_0.3.obj` = r=20mm, k=0.3)

### Region Partition Methods
1. **Coordinate-based** (bijian/kedian): Uses Y-coordinates from `region.csv` landmarks
2. **ID-based** (ppdh/headspace): Uses pre-computed `region_labels.txt` by vertex ID
   - Different datasets have different coordinate systems
   - ID-based method ensures consistent region assignment across datasets

### Midsagittal Plane
- Defined by 3 vertex indices in Template.obj: **3, 91, 3664**
- These correspond to points near:
  - Pt1 (0.44, 51.63, 25.11) → vertex 3
  - Pt2 (-0.09, 3.42, 44.43) → vertex 91
  - Pt3 (-0.18, -26.38, 47.89) → vertex 3664
- Used for calculating menton distance to midsagittal plane

## Mathematical Formulas

### Gaussian Normal Displacement
```
c* = c + k*r*e_hat          # Biased kernel center
d_i = ||v_i - c*||          # Distance from biased center
w_i = exp(-0.5*(d_i/r)²)    # Gaussian weight
Δv_i = σ * A * w_i * n_i    # Displacement along normal
```

### Gaussian Directional Displacement
```
c* = c                      # No bias (symmetric)
d_i = ||v_i - c*||         # Distance from center
w_i = exp(-0.5*(d_i/r)²)   # Gaussian weight
Δv_i = σ * w_i * (r*k) * d # Displacement along direction d
```

### RBF Wendland C2
```
r_norm = d_i / r                            # Normalized distance [0,1]
w_i = (1 - r_norm)⁴ * (4*r_norm + 1)       # Wendland C2 weight
Δv_i = σ * w_i * (r*k) * direction         # Weighted displacement
```

### Point-to-Plane Distance
```
# Plane defined by 3 points p1, p2, p3
normal = normalize(cross(p2-p1, p3-p1))
d = -dot(normal, p1)
distance = |dot(normal, point) + d|
```

## Common Operations

### View a mesh with landmarks
```python
python view_template.py
# Edit the script to specify obj_file and csv_file
```

### Check symmetry of a mesh
```python
python check_symmetry.py
# Output: pairs.csv, symmetry_analysis.pdf
```

### Generate deformed meshes
```python
# Gaussian normal displacement (法向位移)
python gaussian_displacement.py
# Choose option 1

# Gaussian directional displacement (方向性位移)
python gaussian_displacement.py
# Choose option 2

# RBF Wendland displacement
python rbf_wendland_displacement.py
```

### Batch process ppdh/headspace datasets
```python
# Generate heatmaps and statistics
python batch_asymmetry_analysis.py --input_dirs output_ppdh output_headspace

# Add menton distance to summaries
python add_menton_distance.py

# Merge all summaries to Excel
python merge_summary_to_excel.py
```

### Standard ICP registration and analysis
```python
# Run ICP on all deformed meshes
python standard_icp.py
# Output: standard_icp_result/*.csv files

# Visualize asymmetry heat map
python visualize_asymmetry.py
# Edit script to specify original_obj and mirrored_registered_obj
# Output: asymmetry_analysis.pdf

# Visualize pairing quality
python visualize_pairing_quality.py
# Edit script to specify obj_file and csv_file
# Output: pairing_quality.pdf with heat map and statistics
```

## Known Issues & Notes

1. **Asymmetry in Normal Displacement**: The biased Gaussian normal method (`gaussian_normal_displacement`) creates slight asymmetry (~0.04% relative error) due to the offset kernel center `c* = c + k*r*e_hat`. This is intentional for creating directional bias but results in non-symmetric deformations.

2. **Symmetry Preservation**: Use `gaussian_directional_displacement` or `rbf_wendland_displacement` if perfect bilateral symmetry must be maintained.

3. **ICP Registration**: The mirror registration uses identity matrix as initial transformation since Template.obj is already perfectly symmetric.

4. **Matplotlib Chinese Font**: All plots use English to avoid Chinese character rendering issues.

5. **NumPy Version**: Must use numpy 1.22.4 (not 2.x) for compatibility with trimesh 4.10.1 in Python 3.9.7.

6. **Visualization Rendering**: Both `visualize_asymmetry.py` and `visualize_pairing_quality.py` use Open3D's `capture_screen_image()` method for consistent face orientation in PDF reports. Earlier versions used matplotlib 3D rendering which had incorrect viewing angles.

7. **Open3D Vertex Reordering**: Open3D reorders vertices when loading OBJ files. Use `vertex_mapping.npz` and `vertex_reorder.py` functions to convert between orders.

8. **Different Coordinate Systems**: ppdh/headspace datasets have different coordinate systems from Template.obj. Use ID-based region partition (`region_labels.txt`) instead of coordinate-based partition for these datasets.

9. **Open3D PNG Warnings**: Harmless warnings "Read PNG failed: unable to parse header" appear when loading OBJ files without textures.

## Research Context

This project appears to be exploring different mathematical approaches to generating controlled facial asymmetries, potentially for:
- Studying perception of facial asymmetry
- Medical simulation (e.g., facial paralysis, developmental asymmetries)
- Computer graphics and animation
- Morphometric analysis
- **Orthodontic research** (analyzing menton deviation from midsagittal plane)

The three deformation methods provide different characteristics:
- **Gaussian Normal**: Creates natural "bulging" or "sunken" deformations
- **Gaussian Directional**: Creates lateral shifts (simulates bone displacement)
- **RBF Wendland**: Smoother, more controlled deformations with C² continuity

## Workflow Summary

**Typical analysis workflow for simulated data:**
1. Generate deformed meshes using `gaussian_displacement.py` or `rbf_wendland_displacement.py`
2. Run `standard_icp.py` to perform ICP registration on all meshes
3. Visualize results:
   - Use `visualize_pairing_quality.py` to see ICP pairing quality
   - Use `visualize_asymmetry.py` to analyze asymmetry heat maps
4. Extract metrics (MSE, RMSE, mean distance) from PDF reports or return dictionaries

**Typical analysis workflow for patient data (ppdh/headspace):**
1. Ensure OBJ files in `output_xxx/mapped_templates/` and FA CSVs in `output_xxx/fa_results/`
2. Run `python batch_asymmetry_analysis.py` to generate heatmaps and statistics
3. Run `python add_menton_distance.py` to add menton distance to summaries
4. Run `python merge_summary_to_excel.py` to create combined Excel report

## Next Steps (Potential)

- [ ] Apply deformations to other landmarks (currently only landmark 2)
- [ ] Test negative sigma values (leftward/inward displacement)
- [ ] Compare ICP registration results with geometry-based pairs
- [ ] Analyze how different deformation methods affect facial features
- [x] Generate asymmetry metrics (comparing displaced models to original) - **COMPLETED**
- [x] Visualize deformation vector fields using heat maps - **COMPLETED**
- [x] Batch process all meshes and generate comparison tables - **COMPLETED**
- [x] Add menton distance to midsagittal plane calculation - **COMPLETED**
- [ ] Test on non-symmetric input meshes
- [ ] Automate PDF report generation for all test cases

## Quick Start for Next Session

1. Read this file to understand project context
2. Check `Template.obj` and `template landmark.csv` exist
3. Verify generated data directories exist
4. Run `python view_template.py` to visualize current state
5. Continue with specific deformation or analysis tasks
