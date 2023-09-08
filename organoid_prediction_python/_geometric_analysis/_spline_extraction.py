from scipy import interpolate
import numpy as np
from scipy.optimize import minimize
from toska_functions import *
# TODO import from updated toska library, Credit Allyson

# TODO Docstring
def extract_spine_coordinates(skeleton, return_edge_points=False):
    # Extract initial spine data from the skeleton
    coords, e_pts, b_pts, brnch, brnch_ids, brnch_lengths = n26_parse_skel_3d(skeleton.astype(bool), 0, 1, 2)
    
    # Relabel branch points
    bp_img, n_bp = n26_relabel_brnch_pts(b_pts, brnch.shape, brnch.dtype)
    
    # Generate adjacency matrix
    adj_mat = n26_adjacency_matrix(e_pts, bp_img, n_bp, brnch, len(brnch_ids))
    
    # Construct a skeleton network
    nodes, weighted_edges, graph = skeleton_network(adj_mat, brnch_lengths)
    
    # Search for spine path within the skeleton
    spine_path, spine_length = skeleton_spine_search(nodes, graph)
    
    # Mapping and creating spine image
    spine_edges_list = spine_edges(spine_path)
    mapped_spine_edges = map_spine_edges(spine_edges_list, adj_mat, brnch_lengths, brnch_ids)
    img_spine = create_spine_img(brnch, mapped_spine_edges)
    
    # Convert spine image to coordinates
    coords_spine = np.asarray(np.where(img_spine)).T

    # If edge points are requested, extract and return them
    if return_edge_points:
        edge_point_idxs = np.argwhere(np.sum(adj_mat[:, spine_path], 1)[20:] == 1)
        edge_points = np.array(e_pts)[np.ravel(edge_point_idxs)]
        points_in_spine = set(map(tuple, edge_points)).intersection(map(tuple, coords_spine))
        spine_edge_points = np.array(list(points_in_spine))
        
        return coords_spine, spine_edge_points

    return coords_spine

# TODO Docstring
def fit_bspline_to_data(spline_point_list, start_coord, knots_count=3, spline_degree=2):
    ordered_point_list = order_points_from_start(spline_point_list,start_coord)
    
    # Normalize the parameter t for the range [0, 1]
    t_normalized = np.linspace(0, 1, len(ordered_point_list))
    
    # Define knots for the B-spline
    knots = np.linspace(0, 1, knots_count)[1:-1]
    
    # Fit quadratic B-spline to the ordered data
    tck_x = interpolate.splrep(t_normalized, ordered_point_list[:, 2], t=knots, k=spline_degree)
    tck_y = interpolate.splrep(t_normalized, ordered_point_list[:, 1], t=knots, k=spline_degree)
    tck_z = interpolate.splrep(t_normalized, ordered_point_list[:, 0], t=knots, k=spline_degree)
    
    # Function to evaluate the B-spline for a given parameter t
    def bspline_eval(t_param):
        x = interpolate.splev(t_param, tck_x)
        y = interpolate.splev(t_param, tck_y)
        z = interpolate.splev(t_param, tck_z)
        return z, y, x
    
    return bspline_eval

# TODO Docstring
def find_parametric_curve_mesh_intersects(mesh, parametric_curve_function, search_space = [-1,2]):
    # Define search space
    space = np.linspace(search_space[0], search_space[1], 1000)
    intersects = []

    # Iterating over a range of t values
    for t in range(len(space)-1):
        point1 = parametric_curve_function(space[t])
        point2 = parametric_curve_function(space[t+1])
        
        # Check if point is inside mesh
        intersect = mesh.intersect_with_line(point1,point2)
        if len(intersect)!= 0:
            intersects.append(intersect[0])

    output = ParamIntersections(coordinates=np.array(intersects))

    def objective_function(t, intersection_point):
        curve_point = parametric_curve_function(t[0])  # t is an array, so we extract its first value
        return distance(curve_point, intersection_point)

    param_1 = minimize(objective_function, x0=search_space[0], args=(intersects[0])).x[0]
    param_2 = minimize(objective_function, x0=search_space[1], args=(intersects[1])).x[0]

    output.parameters = np.array([param_1,param_2])
    output.coordinates = parametric_curve_function(output.parameters)

    return output


# TODO Docstring
def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# TODO Docstring
class ParamIntersections():
    def __init__(self,coordinates = None, parameters = None):
        self.coordinates = coordinates
        self.parameters = parameters

# TODO Docstring
def order_points_from_start(points, start_coord):
    # Find the index of the start coordinate
    start_idx = np.where((points == start_coord).all(axis=1))[0][0]
    
    # Initialize with the start point
    ordered_points = [points[start_idx]]
    points = np.delete(points, start_idx, axis=0)
    
    while len(points) > 0:
        # Compute distances from the last point in ordered_points to all points
        distances = np.linalg.norm(points - ordered_points[-1], axis=1)
        
        # Find the index of the nearest point
        nearest_idx = np.argmin(distances)
        
        # Append this point to ordered_points
        ordered_points.append(points[nearest_idx])
        
        # Remove this point from points
        points = np.delete(points, nearest_idx, axis=0)
    
    return np.array(ordered_points)

