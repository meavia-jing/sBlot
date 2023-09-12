import os
from pathlib import Path

os.environ['USE_PYGEOS'] = '0'  # Fix for Pandas deprecation warning

import re
from fnmatch import fnmatch
from functools import lru_cache
from shutil import copyfile
import colorsys
import math

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib import colors

from random import random
import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from scipy.optimize import linear_sum_assignment
import seaborn as sns

from shapely import geometry
from shapely.geometry import Polygon
from shapely.prepared import prep
from shapely.ops import cascaded_union, polygonize

from sblot.align_clusters_across_logs import write_clusters, cluster_agreement, get_permuted_params
from sblot.results import Results
from sblot.util import add_edge


def get_datapath(datapath: str | Path):
    """
    This helper function gets the file path of the cluster and stats file.
    Args
        datapath: the path to the sBayes results

    Returns:
        a dict that contains the  path of cluster and stats file under the datapath folder seperately.
    """
    namelist = []
    pattern = "*.txt"
    for item in os.listdir(datapath):
        if re.match("[nK][0-9]+",item) :
            rawdatapath = os.path.join(datapath,item)
            if os.path.exists(rawdatapath):
                for filename in os.listdir(rawdatapath):
                    if (fnmatch(filename, pattern)):
                        namelist.append(os.path.join(rawdatapath, filename))

    stats = []
    cluster = []

    for item in namelist:
        if "stats" in item and "operator_stats" not in item:
            stats.append(item)
        elif "area" in item or "cluster" in item:
            cluster.append(item)

    cluster = sorted(cluster)
    stats = sorted(stats)
    detailed_path = {"clusters": cluster,
                     "stats": stats}

    return detailed_path

def get_cluster_files(all_cluster_paths,folder_name):
    """
    This function gets a path list for all cluster file in one folder
    Args:
        all_cluster_paths: parameter of Plot: store all the cluster file path under one experiment
        folder_name: the name of folder

    Returns:
        a list of cluster txt file path  for one

    """
    return [x for x in all_cluster_paths if os.path.basename(x).split("_")[1] == folder_name]

def get_stats_files(all_stats_paths,folder_name) :
    """
        This function gets a path list for all stats file in one folder
        Args:
            all_stats_paths: parameter of Plot: store all the stats file path under one experiment
            folder_name: the name of folder

        Returns:
            a list of stats txt file path  for one

        """
    return [x for x in all_stats_paths if os.path.basename(x).split("_")[1] == folder_name]


def align_files(all_cluster_paths,all_stats_paths,folder_names,backupdir):
    """
    This funciton is to align files before combining files
    Args:
        all_cluster_paths: parameter of Plot: store all the cluster file path under one experiment
        all_stats_paths: parameter of Plot: store all the stats file path under one experiment
        folder_names: the names of folders that store different number of clusters
        backupdir:  folders that store backup files

    """
    print("Aligning cluster across logs...")
    for item in set(folder_names):
        # get all the files under each folder
        one_expcluster = get_cluster_files(all_cluster_paths,item)
        one_expstats = get_stats_files(all_stats_paths,item)

        ## align txt for different runs

        result_list =[]
        mean_clusters_list = []
        clusters_backup_list = []
        parameters_backup_list= []

        for i,j in zip(one_expcluster,one_expstats):
            #if not str(j).endswith('stats_n1_.txt'):
            if not re.match("^stats_n1_.*",os.path.basename(j)):
                # Load results
                result = Results.from_csv_files(i, j, burn_in=0)
                # Compute the best permutation
                mean_clusters = np.mean(result.clusters, axis=1)

                result_list.append(result)
                mean_clusters_list.append(mean_clusters)

                clusters_backup_path =  os.path.join(backupdir,os.path.basename(i).partition(".")[0]+".txt" )
                parameters_backup_path = os.path.join(backupdir,os.path.basename(j).partition(".")[0]+".txt")

                clusters_backup_list.append(clusters_backup_path)
                parameters_backup_list.append(parameters_backup_path)

        for i in range(1,len(result_list)):
                # Backup the original files of experiment bigger than 1 (which are overwritten by aligned version)
                copyfile(one_expcluster[i], clusters_backup_list[i])
                copyfile(one_expstats[i], parameters_backup_list[i])

                # Compute the best permutation
                d = cluster_agreement(mean_clusters_list[0], mean_clusters_list[i])
                perm = linear_sum_assignment(d, maximize=True)[1]

                # Permute the clusters and parameters
                clusters_aligned = result_list[i].clusters[perm].transpose((1, 0, 2))
                params_aligned = get_permuted_params(result_list[i], perm)

                write_clusters(one_expcluster[i], clusters_aligned)
                params_aligned.to_csv(one_expstats[i], index=False, sep="\t")


def extract_lines_with_equal_intervals(one_expfiles, output_file, num_lines, has_header=True):
    """
    This function is to extract lines with equal intervals when combining files
    Args:
        one_expfiles:  files that used to combine
        output_file: newly generate combined file name
        num_lines: total number of lines that required by users.
        has_header:

    Returns:

    """
    header = None
    all_lines = []

    # Combine all input files into a single file
    for i, file_path in enumerate(one_expfiles):
        with open(file_path, 'r') as input_file:
            file_lines = input_file.readlines()
            if has_header:
                header = file_lines.pop(0)
            all_lines += file_lines

    # Extract lines with equal intervals from the combined file and write to output file
    total_lines_count = len(all_lines)
    interval = (total_lines_count - 1) // (num_lines - 1)

    with open(output_file, 'w') as output:
        if header:
            output.write(header)
        output.writelines(all_lines[::interval])


def combine_files(acq_length,all_cluster_paths,all_stats_paths,input_main_paths):
    """

    This function is to  combine cluster data selected from differenet experiments and save in a new file

    Returns:
        a new fold that contains newly generate data
    """

    ## get all unique folders: n1, n2, n3
    folder_names = []
    for item in all_cluster_paths:
        label = os.path.basename(item).split("_")[1]
        folder_names.append(label)

    #generate new folder for combined files
    # input_main_paths = fix_relative_path(path=self.config['results']['path_in'],
    #                                      base_directory=self.base_directory)
    # input_main_paths = fix_relative_path(path=path_in,
    #                                      base_directory=base_directory)
    newdir = os.path.join(input_main_paths, "combined_results")
    if not os.path.exists(newdir):
        os.makedirs(newdir)

    ### create new folder to store backup file
    backupdir = os.path.join(input_main_paths, "backup")
    if not os.path.exists(backupdir):
        os.makedirs(backupdir)

     ### align all the others files according to the first files
    align_files(all_cluster_paths,all_stats_paths,folder_names,backupdir)

    ### start generate combined cluster file
    ##  for-loop cluster file in each fold
    print("Combining files...")
    for item in set(folder_names):
        one_expcluster = get_cluster_files(all_cluster_paths,item)
        one_expstats = get_stats_files(all_stats_paths,item)

        ### generate name for combined cluster file
        output_cluster_name = os.path.basename(one_expcluster[0]).rpartition("_")[0]
        output_cluster_file = os.path.join(newdir,output_cluster_name+".txt")

        ### generate name for combined stats file
        output_stats_name = os.path.basename(one_expstats[0]).rpartition("_")[0]
        output_stats_file = os.path.join(newdir, output_stats_name + ".txt")


        ## combine cluster files and generate a new cluster files
        has_header = False
        extract_lines_with_equal_intervals(one_expcluster,output_cluster_file,acq_length,has_header)

        ## combine stats files and generate a new stats file
        has_header = True
        extract_lines_with_equal_intervals(one_expstats, output_stats_file, acq_length,has_header)


# def decompose_config_path(config_path) :
#     """
#
#     Args:
#         config_path: Path like
#
#     Returns:
#          -> tuple[Path, Path]
#     """
#     abs_config_path = Path(config_path).absolute()
#     base_directory = abs_config_path.parent
#     return base_directory, abs_config_path

def decompose_config_path(config_path):
    """
    This helper function gets parent path and absolute path for a file

    Args:
        config_path: -> tuple[Path, Path]:

    Returns:

    """

    abs_config_path = Path(config_path).absolute()
    base_directory = abs_config_path.parent
    return base_directory, abs_config_path


def Hex_to_RGB(hex):
    """
    This function converts hex to RGB
    "#1b9e77" to "27, 158, 119"
    """
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    #     rgb = str(r)+','+str(g)+','+str(b)
    rgb = [r, g, b]

    return rgb

def rgb_color(colors_area):
    """
    convert of list of hex color to rgb color
    Args:
        colors_area: a list of hex color
    """
    rgb = np.array(list((map(Hex_to_RGB, colors_area))))
    red = rgb[:,0]
    green = rgb[:,1]
    blue = rgb[:,2]
    return red, green, blue


def rgb_to_hex(rgb: tuple) -> str:
    """Convert rgb color to hex"""
    return '#%02x%02x%02x' % rgb


def grid_bounds(geom: Polygon, delta: float) -> list[Polygon]:
    """
     This function generates grid for a polygon
    Args:
        geom: shapely polygon want to convert to grid
        delta: resolution for the grid
    Returns:
        a grid polygon
    """
    minx, miny, maxx, maxy = geom.bounds
    nx = int((maxx - minx) / delta)
    ny = int((maxy - miny) / delta)
    gx, gy = np.linspace(minx, maxx, nx), np.linspace(miny, maxy, ny)
    grid = []
    for i in range(len(gx) - 1):
        for j in range(len(gy) - 1):
            poly_ij = Polygon([[gx[i], gy[j]], [gx[i], gy[j + 1]], [gx[i + 1], gy[j + 1]], [gx[i + 1], gy[j]]])
            grid.append(poly_ij)
    return grid


def partition(geom: Polygon, delta: float) -> list[Polygon]:
    """
     This function is to clip the gird with a polygon
     Args:
        geom: shapely polygon want to convert to grid
        delta: resolution for the grid
    Returns:
        a grid polygon clipped by geom
    """
    prepared_geom = prep(geom)
    grid = [cell for cell in grid_bounds(geom, delta)
            if prepared_geom.contains(cell.centroid)]
    return grid


def polygon_width(polygon: Polygon) -> float:
    """This funcitons gets the width for a polygon."""
    return polygon.bounds[2] - polygon.bounds[0]


def polygon_height(polygon: Polygon) -> float:
    """This function gets the height for a polygon"""
    return polygon.bounds[3] - polygon.bounds[1]


def get_cluster_colors(n_clusters: int, custom_colors=None):
    """
    This function return a list of color at a given number
    Args:
        n_clusters: number of colors
        custom_colors: custom colors
    Returns:
        a list of colors
    """
    if custom_colors is None:
        clrs = []
        for i, x in enumerate(np.linspace(0, 1, n_clusters, endpoint=False)):
            b = i % 2
            h = x % 1
            s = 0.6 + 0.4 * (1 - b)
            v = 0.5 + 0.3 * (b)
            clrs.append(
                colors.to_hex(colors.hsv_to_rgb((h, s, v)))
            )
        return clrs
    else:
        provided = np.array([colors.to_rgba(c) for c in custom_colors])
        additional = get_cluster_colors(n_clusters - len(custom_colors))
        return list(np.concatenate((provided, additional), axis=0))


def annotate_label(xy, label, color, offset_x, offset_y, ax, fontsize=10):
    """
    This function is to annotate label in figure
    """
    x = xy[0]+offset_x
    y = xy[1]+offset_y
    anno_opts = dict(xy=(x, y), fontsize=fontsize, color=color)
    ax.annotate(label, **anno_opts)

def lighten_color(color, amount=0.2):
    """
    This functon is to lighten color

    """
    c = colorsys.rgb_to_hls(*color)
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def scientific(x):
    """
    using scientific notation to note a number

    """
    b = int(np.log10(x))
    a = x / 10 ** b
    return '%.2f \cdot 10^{%i}' % (a, b)


def add_log_likelihood_legend(likelihood_single_clusters: dict):
    """
    This function is to sort cluster accroding to its likehood

    Returns:
        a list of cluster lables and an empty list contianing patch.Rectangle like objects

    """

    # Legend for cluster labels
    cluster_labels = ["      log-likelihood per cluster"]

    lh_per_cluster = np.array(list(likelihood_single_clusters.values()), dtype=float)
    to_rank = np.mean(lh_per_cluster, axis=1)
    p = to_rank[np.argsort(-to_rank)]

    for i, lh in enumerate(p):
        cluster_labels.append(f'$Z_{i + 1}: \, \;\;\; {int(lh)}$')

    extra = patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    # Line2D([0], [0], color=None, lw=6, linestyle='-')

    return cluster_labels, [extra]

@lru_cache(maxsize=128)
def get_corner_points(n, offset=0.5 * np.pi):
    """
    Generate corner points of a equal sided ´n-eck´.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + offset
    return np.array([np.cos(angles), np.sin(angles)]).T


def fill_outside(polygon, color, ax=None):
    """
    Fill the area outside the given polygon with ´color´.
    Args:
        polygon (np.array): The polygon corners in a numpy array.
            shape: (n_corners, 2)
        ax (plt.Axis): The pyplot axis.
        color (str or tuple): The fill color.
    """
    if ax is None:
        ax = plt.gca()

    n_corners = polygon.shape[0]
    if n_corners <= 2:
        raise ValueError('Can only plot polygons with >2 corners')

    i_left = np.argmin(polygon[:, 0])
    i_right = np.argmax(polygon[:, 0])

    # Find corners of bottom face
    i = i_left
    bot_x = [polygon[i, 0]]
    bot_y = [polygon[i, 1]]
    while i % n_corners != i_right:
        i += 1
        bot_x.append(polygon[i, 0])
        bot_y.append(polygon[i, 1])

    # Find corners of top face
    i = i_left
    top_x = [polygon[i, 0]]
    top_y = [polygon[i, 1]]
    while i % n_corners != i_right:
        i -= 1
        top_x.append(polygon[i, 0])
        top_y.append(polygon[i, 1])

    ymin, ymax = ax.get_ylim()
    ax.fill_between(bot_x, ymin, bot_y, color=color)
    ax.fill_between(top_x, ymax, top_y, color=color)


def get_family_shapes(n_family, custom_shapes=None):
    """
    This funcition gets a list of shape with a given number
    Args:
        n_family: int
        custom_shapes: custom shape

    Returns:
        List of shape

    """
    # markerlist 15 ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    usefulmarkers = list(Line2D.filled_markers)
    ## '8' and 'o' is too similar in map
    usefulmarkers.remove('8')
    if custom_shapes is None and n_family<= len(usefulmarkers):
        print(f'No colors for clusters provided in featuremap>graphic>clusters>mark'
              f'in the config plot file. I am using default colors instead.')
        return list(usefulmarkers)[:n_family]
    elif len(custom_shapes) <= n_family and n_family <= len(usefulmarkers):
        left_markers = [item for item in usefulmarkers if item not in custom_shapes]
        addition_num = n_family - len(custom_shapes)
        additional = random.sample(left_markers,addition_num)
        # print("custom_shapes",custom_shapes)
        # print("additional",additional)
        return custom_shapes+additional
    elif len(custom_shapes) >= n_family and n_family <= len(usefulmarkers):
        return custom_shapes
    else:
        print("Can not provide enough shapes")



def compute_bbox(extent):
    """
    This function compute bounding box for extent
    Args:
        extent:

    Returns:

    """
    bbox = geometry.box(extent['x_min'], extent['y_min'],
                        extent['x_max'], extent['y_max'])
    return bbox


def get_cluster_freq(cluster):
    """Computes the frequency at which each object occurs in each area."""
    cluster = np.asarray(cluster)
    return np.mean(cluster, axis=0)


def standard_idw(
    grid_lon: NDArray[float],
    grid_lat: NDArray[float],
    longs: NDArray[float],
    lats: NDArray[float],
    d_values: NDArray[float],
    id_power: int = 2,
    background_weight: float = 0.0,
    background_value: float = 0.0
):
    """
    calculating inverse distance weights
    """

    # The grid has more than one dimension. We flatten it, but remember the shape so
    # that we can bring it back to the original shape in the end.
    grid_shape = grid_lon.shape
    grid_lon = grid_lon.flatten()
    grid_lat = grid_lat.flatten()

    # Compute distances from the grid cell (lon, lat) to the values (longs, lats)
    dists = np.sqrt((longs[np.newaxis, :] - grid_lon[:, np.newaxis]) ** 2 +
                    (lats[np.newaxis, :] - grid_lat[:, np.newaxis]) ** 2)

    # The IDW weights are given by " w = 1 / (d(x, x_i)^power + 1)"
    # >> constant 1 is to prevent int divide by zero when distance is zero.
    weights = 1 / (dists ** id_power + 1)

    # Divide sum of weighted values by sum of weights to get IDW interpolation.
    # We add a constant background_value with background_weight, so that the
    # interpolation decays to this background value when no points are nearby.
    sum_weighted_values = (background_value * background_weight +
                           np.sum(d_values[None, :] * weights, axis=1))
    sum_weights = background_weight + np.sum(weights, axis=1)
    idw = sum_weighted_values / sum_weights

    # Reshape the idw values to match the original grid shape
    return idw.reshape(grid_shape)


def cal_idw(extentpoly, point_rgb, delta, idw_power, background_weight):
    """
    This function is to generate grid based a base polygon and calculate IDW value for each cell
    Args:
        extentpoly: base polygon used to generate grid
        point_rgb: point with x,y and hex color
        delta: resolution for grid
        idw_power: control the weights of distance
        background_weight:  a constant background_value; the interpolation decays to this background value when no points are nearby.
    Returns:

    """
    grid = partition(extentpoly, delta)
    grid = gpd.GeoDataFrame(geometry=gpd.GeoSeries(grid))

    bbox_width = polygon_width(extentpoly)
    bbox_height = polygon_height(extentpoly)

    # Compute weights for the background color.
    # I decided to weigh the background the same as 1 sample that is 1/8 of the
    # diagonal of the whole map away from each grid cell. Seems to give visually
    # pleasing results in the Balkans example.
    dist_diag = math.sqrt(bbox_width ** 2 + bbox_height ** 2)
    background_weight = background_weight / ((dist_diag / 8) ** idw_power + 1)

    # calculating the idw color for the entire grid
    grid_centroids = grid.geometry.centroid
    grid_x = grid_centroids.x.to_numpy()
    grid_y = grid_centroids.y.to_numpy()
    x = point_rgb.x.to_numpy()
    y = point_rgb.y.to_numpy()

    # Apply IDW interpolation to each color channel separately
    for c in ['red', 'green', 'blue']:
        grid[c] = standard_idw(
            grid_lon=grid_x,
            grid_lat=grid_y,
            longs=x,
            lats=y,
            d_values=point_rgb[c].to_numpy(),
            id_power=idw_power,
            background_weight=background_weight,
            background_value=255
        ).astype(int)

    grid['idw_hex'] = [rgb_to_hex(rgb) for rgb in zip(grid.red, grid.green, grid.blue)]

    return grid


def compute_alpha_shapes(points, alpha_shape):

    """
    Compute the alpha shape (concave hull) of a set of sites
    Args:
        points (np.array): subset of locations around which to create the alpha shapes (e.g. family, cluster, ...)
        alpha_shape (float): parameter controlling the convexity of the alpha shape
    Returns:
        (polygon): the alpha shape"""

    tri = Delaunay(points, qhull_options="QJ Pp")

    edges = set()
    edge_nodes = []

    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

        # Semiperimeter of triangle
        s = (a + b + c) / 2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)

        "alpha value to influence the shape of the convex hull Smaller numbers don't fall inward "
        "as much as larger numbers. Too large, and you lose everything!"

        if circum_r < 1.0 / alpha_shape:
            add_edge(edges, edge_nodes, points, ia, ib)
            add_edge(edges, edge_nodes, points, ib, ic)
            add_edge(edges, edge_nodes, points, ic, ia)

    m = geometry.MultiLineString(edge_nodes)

    triangles = list(polygonize(m))
    polygon = cascaded_union(triangles)

    return polygon


def style_axes(extent, ax):
    """
    set extent for a plot
    """

    # setting axis limits
    ax.set_xlim([extent['x_min'], extent['x_max']])
    ax.set_ylim([extent['y_min'], extent['y_max']])

    # Removing axis labels
    ax.set_xticks([])
    ax.set_yticks([])


def min_and_max_with_padding(x: list[float], pad=0.05) -> (float, float):
    """Compute the minimum and maximum of a list of numbers and add a padding that is
    given relative to the total range: `pad*(max-min)`."""
    lower = np.min(x)
    upper = np.max(x)
    diff = upper - lower
    return lower - pad * diff, upper + pad * diff


def compute_dic(lh):
    """This function computes the deviance information criterion
    (see for example Celeux et al. 2006) using the posterior mode as a point estimate
    Args:
        lh: (dict): log-likelihood of samples in hte posterior
        burn_in(float): percentage of samples, which are discarded as burn-in
        """
    # max(ll) likelihood evaluated at the posterior mode
    d_phi_pm = -2 * np.max(lh)
    mean_d_phi = -4 * np.mean(lh)
    dic = mean_d_phi + d_phi_pm
    return dic


def add_mean_line_to_kde(x: NDArray[float], ax: plt.Axes, color, lw):
    print(ax.lines)
    kdeline = ax.lines[0]
    mean = np.mean(x)
    xs = kdeline.get_xdata()
    ys = kdeline.get_ydata()
    height = np.interp(mean, xs, ys)
    ax.vlines(mean, 0, height, color=color, lw=lw, ls=':')


def kdeplot(x: NDArray[float], ax: plt.Axes, color=None, lw=1, alpha=0.2, clip=None, zorder=None):
    sns.kdeplot(x, fill=False, color=color, ax=ax, lw=lw, clip=clip, zorder=zorder)
    kdeline = ax.lines[-1]
    mean = np.mean(x)
    xs = kdeline.get_xdata()
    ys = kdeline.get_ydata()
    height = np.interp(mean, xs, ys)
    ax.vlines(mean, 0, height, color=color, lw=lw, ls=':', zorder=zorder)
    ax.fill_between(xs, 0, ys, facecolor=color, alpha=alpha, zorder=zorder)

