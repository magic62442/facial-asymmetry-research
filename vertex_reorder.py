"""
Vertex Reorder Module

This module provides functions to compute vertex correspondence between
the original OBJ file vertex order and Open3D's reordered vertices.

Open3D may reorder vertices when reading OBJ files, so we need to find
the mapping to correctly apply per-vertex data (like distances from CSV).
"""

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


def read_obj_vertices_original(obj_path):
    """
    Read vertices from OBJ file in original order (without reordering).

    Parameters:
        obj_path: Path to OBJ file

    Returns:
        numpy array of shape (N, 3) with vertex coordinates
    """
    vertices = []
    with open(obj_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
    return np.array(vertices)


def read_obj_vertices_open3d(obj_path):
    """
    Read vertices from OBJ file using Open3D (may reorder vertices).

    Parameters:
        obj_path: Path to OBJ file

    Returns:
        numpy array of shape (N, 3) with vertex coordinates
    """
    mesh = o3d.io.read_triangle_mesh(obj_path)
    return np.asarray(mesh.vertices)


def compute_vertex_mapping(obj_path, tolerance=1e-4):
    """
    Compute the mapping between original OBJ vertex order and Open3D vertex order.

    Parameters:
        obj_path: Path to OBJ file
        tolerance: Maximum distance to consider vertices as matching

    Returns:
        o3d_to_orig: numpy array where o3d_to_orig[i] = original index for Open3D vertex i
        orig_to_o3d: numpy array where orig_to_o3d[i] = Open3D index for original vertex i

    Raises:
        ValueError: If vertices cannot be matched one-to-one
    """
    # Read vertices in both orders
    vertices_orig = read_obj_vertices_original(obj_path)
    vertices_o3d = read_obj_vertices_open3d(obj_path)

    return compute_vertex_mapping_from_vertices(vertices_orig, vertices_o3d, tolerance)


def compute_vertex_mapping_from_vertices(vertices_orig, vertices_o3d, tolerance=1e-4):
    """
    Compute the mapping between two sets of vertices.

    Parameters:
        vertices_orig: numpy array of shape (N, 3) - original vertex order
        vertices_o3d: numpy array of shape (N, 3) - Open3D vertex order
        tolerance: Maximum distance to consider vertices as matching

    Returns:
        o3d_to_orig: numpy array where o3d_to_orig[i] = original index for Open3D vertex i
        orig_to_o3d: numpy array where orig_to_o3d[i] = Open3D index for original vertex i

    Raises:
        ValueError: If vertices cannot be matched one-to-one
    """
    n_orig = len(vertices_orig)
    n_o3d = len(vertices_o3d)

    if n_orig != n_o3d:
        raise ValueError(f"Vertex count mismatch: original={n_orig}, Open3D={n_o3d}")

    print(f"Computing vertex mapping for {n_orig} vertices...")

    # Build KDTree from original vertices
    tree = KDTree(vertices_orig)

    # For each Open3D vertex, find the closest original vertex
    distances, indices = tree.query(vertices_o3d)

    # Check for one-to-one correspondence
    # Each original index should appear exactly once
    unique_indices = np.unique(indices)
    if len(unique_indices) != n_orig:
        raise ValueError(f"Not a one-to-one mapping: {len(unique_indices)} unique matches for {n_orig} vertices")

    # Check all distances are within tolerance
    max_dist = distances.max()
    if max_dist > tolerance:
        raise ValueError(f"Maximum matching distance {max_dist} exceeds tolerance {tolerance}")

    # o3d_to_orig[i] = original vertex index corresponding to Open3D vertex i
    o3d_to_orig = indices.copy()

    # orig_to_o3d[i] = Open3D vertex index corresponding to original vertex i
    orig_to_o3d = np.zeros(n_orig, dtype=int)
    for o3d_idx, orig_idx in enumerate(o3d_to_orig):
        orig_to_o3d[orig_idx] = o3d_idx

    print(f"  Mapping computed successfully")
    print(f"  Maximum matching distance: {max_dist:.10f}")
    print(f"  All {n_orig} vertices matched one-to-one")

    return o3d_to_orig, orig_to_o3d


def reorder_values_orig_to_o3d(values, orig_to_o3d):
    """
    Reorder values from original OBJ order to Open3D order.

    Use this when you have per-vertex values indexed by original OBJ order
    and want to apply them to an Open3D mesh.

    Parameters:
        values: numpy array of shape (N,) or (N, k) indexed by original vertex order
        orig_to_o3d: mapping array from compute_vertex_mapping

    Returns:
        Reordered values indexed by Open3D vertex order
    """
    n_vertices = len(orig_to_o3d)

    if values.ndim == 1:
        reordered = np.zeros(n_vertices, dtype=values.dtype)
        for orig_idx in range(n_vertices):
            o3d_idx = orig_to_o3d[orig_idx]
            reordered[o3d_idx] = values[orig_idx]
    else:
        reordered = np.zeros((n_vertices, values.shape[1]), dtype=values.dtype)
        for orig_idx in range(n_vertices):
            o3d_idx = orig_to_o3d[orig_idx]
            reordered[o3d_idx] = values[orig_idx]

    return reordered


def reorder_values_o3d_to_orig(values, o3d_to_orig):
    """
    Reorder values from Open3D order to original OBJ order.

    Parameters:
        values: numpy array of shape (N,) or (N, k) indexed by Open3D vertex order
        o3d_to_orig: mapping array from compute_vertex_mapping

    Returns:
        Reordered values indexed by original vertex order
    """
    n_vertices = len(o3d_to_orig)

    if values.ndim == 1:
        reordered = np.zeros(n_vertices, dtype=values.dtype)
        for o3d_idx in range(n_vertices):
            orig_idx = o3d_to_orig[o3d_idx]
            reordered[orig_idx] = values[o3d_idx]
    else:
        reordered = np.zeros((n_vertices, values.shape[1]), dtype=values.dtype)
        for o3d_idx in range(n_vertices):
            orig_idx = o3d_to_orig[o3d_idx]
            reordered[orig_idx] = values[o3d_idx]

    return reordered


def save_mapping(o3d_to_orig, orig_to_o3d, output_path='vertex_mapping.npz'):
    """
    Save vertex mapping to file.

    Parameters:
        o3d_to_orig: mapping array
        orig_to_o3d: mapping array
        output_path: output file path (.npz format)
    """
    np.savez(output_path, o3d_to_orig=o3d_to_orig, orig_to_o3d=orig_to_o3d)
    print(f"Mapping saved to {output_path}")


def load_mapping(input_path='vertex_mapping.npz'):
    """
    Load vertex mapping from file.

    Parameters:
        input_path: input file path (.npz format)

    Returns:
        o3d_to_orig, orig_to_o3d: mapping arrays
    """
    data = np.load(input_path)
    return data['o3d_to_orig'], data['orig_to_o3d']


if __name__ == "__main__":
    # Test with Template.obj
    obj_path = "Template.obj"

    print("=" * 60)
    print("Vertex Reorder - Compute and Save Mapping")
    print("=" * 60)

    # Compute mapping
    o3d_to_orig, orig_to_o3d = compute_vertex_mapping(obj_path)

    # Save mapping to file
    save_mapping(o3d_to_orig, orig_to_o3d, 'vertex_mapping.npz')

    # Verify the mapping is correct
    vertices_orig = read_obj_vertices_original(obj_path)
    vertices_o3d = read_obj_vertices_open3d(obj_path)

    print("\nVerification:")

    # Test: reordered original vertices should match Open3D vertices
    reordered_orig = reorder_values_orig_to_o3d(vertices_orig, orig_to_o3d)
    diff = np.abs(reordered_orig - vertices_o3d).max()
    print(f"  Max difference after reordering (orig -> o3d): {diff:.10f}")

    # Test: reordered Open3D vertices should match original vertices
    reordered_o3d = reorder_values_o3d_to_orig(vertices_o3d, o3d_to_orig)
    diff2 = np.abs(reordered_o3d - vertices_orig).max()
    print(f"  Max difference after reordering (o3d -> orig): {diff2:.10f}")

    # Show some example mappings
    print("\nExample mappings (first 10 vertices):")
    print("  Original idx -> Open3D idx:")
    for i in range(10):
        print(f"    {i} -> {orig_to_o3d[i]}")

    print("\n  Open3D idx -> Original idx:")
    for i in range(10):
        print(f"    {i} -> {o3d_to_orig[i]}")

    print("\n" + "=" * 60)
    print("Complete! Mapping saved to vertex_mapping.npz")
    print("=" * 60)
