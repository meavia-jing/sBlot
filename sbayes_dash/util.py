#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import io
import time
import csv
import os
import sys
import traceback
import warnings
from pathlib import Path
from math import sqrt
from itertools import permutations
from typing import Sequence, Union, Iterator

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
import pandas as pd
import scipy
import scipy.spatial as spatial
from scipy.special import betaln, expit, gammaln
from scipy.sparse import csr_matrix
from unidecode import unidecode


EPS = np.finfo(float).eps
COLOR_0 = "rgba(150, 150, 150, 0.4)"
COLOR_1 = "#990055"


PathLike = Union[str, Path]
"""Convenience type for cases where `str` or `Path` are acceptable types."""


def encode_cluster(cluster: NDArray[bool]) -> str:
    """Format the given cluster as a compact bit-string."""
    cluster_s = cluster.astype(int).astype(str)
    return ''.join(cluster_s)


def decode_cluster(cluster_str: str) -> NDArray[bool]:
    """Read a bit-string and parse it into an area array."""
    return np.array(list(cluster_str)).astype(int).astype(bool)


def format_cluster_columns(clusters: NDArray[bool]) -> str:
    """Format the given array of clusters as tab separated strings."""
    clusters_encoded = map(encode_cluster, clusters)
    return '\t'.join(clusters_encoded)


def parse_cluster_columns(clusters_encoded: str) -> NDArray[bool]:
    """Read tab-separated area encodings into a two-dimensional area array."""
    clusters_decoded = map(decode_cluster, clusters_encoded.split('\t'))
    return np.array(list(clusters_decoded))


def compute_distance(a, b):
    """ This function computes the Euclidean distance between two points a and b

    Args:
        a (list): The x and y coordinates of a point in a metric CRS.
        b (list): The x and y coordinates of a point in a metric CRS.

    Returns:
        float: Distance between a and b
    """

    a = np.asarray(a)
    b = np.asarray(b)
    ab = b-a
    dist = sqrt(ab[0]**2 + ab[1]**2)

    return dist


def bounding_box(points):
    """ This function retrieves the bounding box for a set of 2-dimensional input points

    Args:
        points (numpy.array): Point tuples (x,y) for which the bounding box is computed
    Returns:
        (dict): the bounding box of the points
    """
    x = [x[0] for x in points]
    y = [x[1] for x in points]
    box = {'x_max': max(x),
           'y_max': max(y),
           'x_min': min(x),
           'y_min': min(y)}

    return box


def get_neighbours(cluster, already_in_cluster, adjacency_matrix):
    """This function returns the neighbourhood of a cluster as given in the adjacency_matrix, excluding sites already
    belonging to this or any other cluster.

    Args:
        cluster (np.array): The current cluster (boolean array)
        already_in_cluster (np.array): All sites already assigned to a cluster (boolean array)
        adjacency_matrix (np.array): The adjacency matrix of the sites (boolean)

    Returns:
        np.array: The neighborhood of the cluster (boolean array)
    """

    # Get all neighbors of the current zone, excluding all vertices that are already in a zone

    neighbours = np.logical_and(adjacency_matrix.dot(cluster), ~already_in_cluster)
    return neighbours


def compute_delaunay(locations):
    """Computes the Delaunay triangulation between a set of point locations

    Args:
        locations (np.array): a set of locations
            shape (n_sites, n_spatial_dims = 2)
    Returns:
        (np.array) sparse matrix of Delaunay triangulation
            shape (n_edges, n_edges)
    """
    n = len(locations)

    if n < 4:
        # scipy's Delaunay triangulation fails for <3. Return a fully connected graph:
        return csr_matrix(1-np.eye(n, dtype=int))

    delaunay = spatial.Delaunay(locations, qhull_options="QJ Pp")

    indptr, indices = delaunay.vertex_neighbor_vertices
    data = np.ones_like(indices)

    return csr_matrix((data, indices, indptr), shape=(n, n))


def gabriel_graph_from_delaunay(delaunay, locations):
    delaunay = delaunay.toarray()
    # converting delaunay graph to boolean array denoting whether points are connected
    delaunay = delaunay > 0

    # Delaunay indices and locations
    delaunay_connections = []
    delaunay_locations = []

    for index, connected in np.ndenumerate(delaunay):
        if connected:
            # getting indices of points in area
            i1, i2 = index[0], index[1]
            if [i2, i1] not in delaunay_connections:
                delaunay_connections.append([i1, i2])
                delaunay_locations.append(locations[[*[i1, i2]]])
    delaunay_connections = np.sort(np.asarray(delaunay_connections), axis=1)
    delaunay_locations = np.asarray(delaunay_locations)

    # Find the midpoint on all Delaunay edges
    m = (delaunay_locations[:, 0, :] + delaunay_locations[:, 1, :]) / 2

    # Find the radius sphere between each pair of nodes
    r = np.sqrt(np.sum((delaunay_locations[:, 0, :] - delaunay_locations[:, 1, :]) ** 2, axis=1)) / 2

    # Use the kd-tree function in Scipy's spatial module
    tree = spatial.cKDTree(locations)
    # Find the nearest point for each midpoint
    n = tree.query(x=m, k=1)[0]
    # If nearest point to m is at a distance r, then the edge is a Gabriel edge
    g = n >= r * 0.999  # The factor is to avoid precision errors in the distances

    return delaunay_connections[g]


def n_smallest_distances(a, n, return_idx: bool):
    """ This function finds the n smallest distances in a distance matrix

    >>> n_smallest_distances([
    ... [0, 2, 3, 4],
    ... [2, 0, 5, 6],
    ... [3, 5, 0, 7],
    ... [4, 6, 7, 0]], 3, return_idx=False)
    array([2, 3, 4])

    >>> n_smallest_distances([
    ... [0, 2, 3, 4],
    ... [2, 0, 5, 6],
    ... [3, 5, 0, 7],
    ... [4, 6, 7, 0]], 3, return_idx=True)
    (array([1, 2, 3]), array([0, 0, 0]))

    Args:
        a (np.array): The distane matrix
        n (int): The number of distances to return
        return_idx (bool): return the indices of the points (True) or rather the distances (False)

    Returns:
        (np.array): the n_smallest distances
    or
        (np.array, np.array): the indices between which the distances are smallest
    """
    a_tril = np.tril(a)
    a_nn = a_tril[np.nonzero(a_tril)]
    smallest_n = np.sort(a_nn)[: n]
    a_idx = np.isin(a_tril, smallest_n)

    if return_idx:
        return np.where(a_idx)
    else:
        return smallest_n


def clusters_autosimilarity(cluster, t):
    """
    This function computes the similarity of consecutive cluster in a chain
    Args:
        cluster (list): cluster
        t (integer): lag between consecutive cluster in the chain

    Returns:
        (float) : mean similarity between cluster in the chain with lag t
    """
    z = np.asarray(cluster)
    z = z[:, 0, :]
    unions = np.maximum(z[t:], z[:-t])
    intersections = np.minimum(z[t:], z[:-t])
    sim_norm = np.sum(intersections, axis=1) / np.sum(unions, axis=1)

    return np.mean(sim_norm)


def range_like(a):
    """Return a list of incrementing integers (range) with same length as `a`."""
    return list(range(len(a)))


# Encoding
def encode_states(features_raw, feature_states):
    # Define shapes
    n_states, n_features = feature_states.shape
    features_bin_shape = features_raw.shape + (n_states,)
    n_sites, _ = features_raw.shape
    assert n_features == _

    # Initialize arrays and counts
    features_bin = np.zeros(features_bin_shape, dtype=int)
    applicable_states = np.zeros((n_features, n_states), dtype=bool)
    state_names = []
    na_number = 0

    # Binary vectors used for encoding
    one_hot = np.eye(n_states)

    for f_idx in range(n_features):
        f_name = feature_states.columns[f_idx]
        f_states = feature_states[f_name]

        # Define applicable states for feature f
        applicable_states[f_idx] = ~f_states.isna()

        # Define external and internal state names
        s_ext = f_states.dropna().to_list()
        s_int = range_like(s_ext)
        state_names.append(s_ext)

        # Map external to internal states for feature f
        ext_to_int = dict(zip(s_ext, s_int))
        f_raw = features_raw[f_name]
        f_enc = f_raw.map(ext_to_int)
        if not (set(f_raw.dropna()).issubset(set(s_ext))):
            print(set(f_raw.dropna()) - set(s_ext))
            print(s_ext)
        assert set(f_raw.dropna()).issubset(set(s_ext))  # All states should map to an encoding

        # Binarize features
        f_applicable = ~f_enc.isna().to_numpy()
        f_enc_applicable = f_enc[f_applicable].astype(int)

        features_bin[f_applicable, f_idx] = one_hot[f_enc_applicable]

        # Count NA
        na_number += np.count_nonzero(f_enc.isna())

    features = {
        'values': features_bin.astype(bool),
        'states': applicable_states,
        'state_names': state_names
    }

    return features, na_number


def normalize_str(s: str) -> str:
    if pd.isna(s):
        return s
    return str.strip(unidecode(s))


def read_data_csv(csv_path: PathLike | io.StringIO) -> pd.DataFrame:
    na_values = ["", " ", "\t", "  "]
    data: pd.DataFrame = pd.read_csv(csv_path, na_values=na_values, keep_default_na=False, dtype=str)
    data.columns = [unidecode(c) for c in data.columns]
    return data.map(normalize_str)


def write_languages_to_csv(features, sites, families, file):
    """This is a helper function to export features as a csv file
    Args:
        features (np.array): features
            shape: (n_sites, n_features, n_categories)
        sites (dict): sites with unique id
        families (np.array): families
            shape: (n_families, n_sites)
        file(str): output csv file
    """
    families = families.transpose(1, 0)

    with open(file, mode='w', encoding='utf-8') as csv_file:
        f_names = list(range(features.shape[1]))
        csv_names = ['f' + str(f) for f in f_names]
        csv_names = ["name", "x", "y", "family"] + csv_names
        writer = csv.writer(csv_file)
        writer.writerow(csv_names)

        for i in sites['id']:
            # name
            name = "site_" + str(i)
            # location
            x, y = sites['locations'][i]
            # features
            f = np.where(features[i] == 1)[1].tolist()
            # family
            fam = np.where(families[i] == 1)[0].tolist()
            if not fam:
                fam = ""
            else:
                fam = "family_" + str(fam[0])
            writer.writerow([name] + [x] + [y] + [fam] + f)


def touch(fname):
    """Create an empty file at path `fname`."""
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a').close()


def mkpath(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isdir(path):
        touch(path)


def linear_rescale(value, old_min, old_max, new_min, new_max):
    """
    Function to linear rescale a number to a new range

    Args:
         value (float): number to rescale
         old_min (float): old minimum of value range
         old_max (float): old maximum of value range
         new_min (float): new minimum of value range
         new_max (float): new maximum of vlaue range
    """

    return (new_max - new_min) / (old_max - old_min) * (value - old_max) + old_max


def normalize(x, axis=-1):
    """Normalize ´x´ s.t. the last axis sums up to 1.

    Args:
        x (np.array): Array to be normalized.
        axis (int): The axis to be normalized (will sum up to 1).

    Returns:
         np.array: x with normalized s.t. the last axis sums to 1.

    == Usage ===
    >>> normalize(np.ones((2, 4)))
    array([[0.25, 0.25, 0.25, 0.25],
           [0.25, 0.25, 0.25, 0.25]])
    >>> normalize(np.ones((2, 4)), axis=0)
    array([[0.5, 0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5, 0.5]])
    """
    assert np.all(np.sum(x, axis=axis) > 0)
    return x / np.sum(x, axis=axis, keepdims=True)


def decompose_config_path(config_path: PathLike) -> (Path, Path):
    """Extract the base directory of `config_path` and return the path itself as an absolute path."""
    abs_config_path = Path(config_path).absolute()
    base_directory = abs_config_path.parent
    return base_directory, abs_config_path


def fix_relative_path(path: PathLike, base_directory: PathLike) -> Path:
    """Make sure that the provided path is either absolute or relative to the config file directory.

    Args:
        path: The original path (absolute or relative).
        base_directory: The base directory

    Returns:
        The fixed path.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    else:
        return base_directory / path


def timeit(units='s'):
    SECONDS_PER_UNIT = {
        'h': 3600.,
        'm': 60.,
        's': 1.,
        'ms': 1E-3,
        'µs': 1E-6,
        'ns': 1E-9
    }
    unit_scaler = SECONDS_PER_UNIT[units]

    def timeit_decorator(func):

        def timed_func(*args, **kwargs):


            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            passed = (end - start) / unit_scaler

            print(f'Runtime {func.__name__}: {passed:.2f}{units}')

            return result

        return timed_func

    return timeit_decorator


def get_permutations(n: int) -> Iterator[tuple[int]]:
    return permutations(range(n))


def get_best_permutation(
        areas: NDArray[bool],  # shape = (n_areas, n_sites)
        prev_area_sum: NDArray[int],  # shape = (n_areas, n_sites)
) -> NDArray[int]:
    """Return a permutation of areas that would align the areas in the new sample with previous ones."""
    cluster_agreement_matrix = np.matmul(prev_area_sum, areas.T)
    return linear_sum_assignment(cluster_agreement_matrix, maximize=True)[1]


def min_and_max_with_padding(x, pad=0.05):
    lower = np.min(x)
    upper = np.max(x)
    diff = upper - lower
    return lower - pad * diff, upper + pad * diff


def reproject_locations(locations, data_proj, map_proj):
    if data_proj == map_proj:
        return locations
    loc = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*locations.T), crs=data_proj)
    loc_re = loc.to_crs(map_proj).geometry
    return np.array([loc_re.x, loc_re.y]).T


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    # traceback.print_stack(file=log)
    warning_trace = traceback.format_stack()
    warning_trace_str = "".join(["\n\t|" + l for l in "".join(warning_trace).split("\n")])
    message = str(message) + warning_trace_str
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def activate_verbose_warnings():
    warnings.showwarning = warn_with_traceback


if __name__ == "__main__":
    import doctest
    doctest.testmod()

