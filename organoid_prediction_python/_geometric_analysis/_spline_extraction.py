from scipy import interpolate, integrate
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from functools import partial
from .toska_functions import *
import numpy as np
import napari_process_points_and_surfaces as nppas
import vedo
from scipy.optimize import minimize, differential_evolution
from functools import partial
import concurrent.futures as cf
import time
from tqdm import tqdm

# TODO import from updated toska library, Credit Allyson

# DETERMINING SPLINE FUNCTIONS
#---------------------------------------------------------------------------------------------------------------

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
def fit_bspline_to_data(ordered_point_list, knots_count=3, spline_degree=2, return_ticks = False):
    # Normalize the parameter t for the range [0, 1]
    t_normalized = np.linspace(0, 1, len(ordered_point_list))
    
    # Define knots for the B-spline
    knots = np.linspace(0, 1, knots_count)[1:-1]
    
    # Fit quadratic B-spline to the ordered data
    tck_x = interpolate.splrep(t_normalized, ordered_point_list[:, 2], t=knots, k=spline_degree)
    tck_y = interpolate.splrep(t_normalized, ordered_point_list[:, 1], t=knots, k=spline_degree)
    tck_z = interpolate.splrep(t_normalized, ordered_point_list[:, 0], t=knots, k=spline_degree)
    
    # Function to evaluate the B-spline for a given parameter t
    bspline_eval = make_bspline_from_ticks([tck_x,tck_y,tck_z])

    if return_ticks:
        return bspline_eval, [tck_x,tck_y,tck_z]
    return bspline_eval

# TODO Docstring
def make_bspline_from_ticks(ticks):
    # Function to evaluate the B-spline for a given parameter t
    return partial(
        bspline_eval, 
        **{
            "tck_x":ticks[0], 
            "tck_y":ticks[1], 
            "tck_z":ticks[2]
        }
    )


def bspline_eval(t_param,tck_x, tck_y, tck_z):
    x = interpolate.splev(t_param, tck_x)
    y = interpolate.splev(t_param, tck_y)
    z = interpolate.splev(t_param, tck_z)
    if is_list_like(t_param):
        return np.array([z, y, x]).T
    return np.array([z, y, x])


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
def create_translation_function(bspline_eval, ticks, curve_mesh_intersects, num_points=1000, inverse = False):
    """
    Create a function that maps [0, 1] to [t1, t2] such that even t values give 
    evenly spaced points on the curve. t=0 maps to t1 and t=1 maps to t2.
    """
    t1, t2 = curve_mesh_intersects
    # Sample t-values densely between t1 and t2
    t_values_dense = np.linspace(t1, t2, num_points)
    
    # Compute arc lengths for each t-value from t1 to the current t-value
    cum_arc_lengths = [compute_arc_length_between_points(ticks[0], ticks[1], ticks[2], t1, t) for t in t_values_dense]
    
    # Normalize arc lengths to [0, 1]
    total_length = cum_arc_lengths[-1]
    normalized_arc_lengths = np.array(cum_arc_lengths) / total_length
    
    if inverse:
        # Interpolation function for the inverse (this will map [t1, t2] back to [0, 1])
        inverse_translation_func = interp1d(t_values_dense, normalized_arc_lengths, kind='linear', fill_value="extrapolate")
        return inverse_translation_func
    
    # Interpolation function (this will map [0, 1] to [t1, t2])
    translation_func = interp1d(normalized_arc_lengths, t_values_dense, kind='linear', fill_value="extrapolate")
    return translation_func 

# TODO Docstring
def smooth_resample_curve(points, window_length=61, poly_order=2):
    """
    Smooth and resample a curve using arc-length parameterization and Savitzky-Golay filter.
    
    Parameters:
    - points: An array of shape (n, 3) representing the x, y, z coordinates.
    - window_length: Window length for Savitzky-Golay filter.
    - poly_order: Polynomial order for Savitzky-Golay filter.
    
    Returns:
    - resampled_smoothed_points: An array of shape (n, 3) representing the smoothed x, y, z coordinates.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Compute the cumulative distance for parameterization
    s = compute_cumulative_distance(x, y, z)
    
    # Interpolate the data using the cumulative distance as the parameter
    interp_x = interp1d(s, x, kind='linear')
    interp_y = interp1d(s, y, kind='linear')
    interp_z = interp1d(s, z, kind='linear')
    
    # Resample the curve at regular intervals of s
    s_resampled = np.linspace(s.min(), s.max(), len(s))
    x_resampled = interp_x(s_resampled)
    y_resampled = interp_y(s_resampled)
    z_resampled = interp_z(s_resampled)
    
    # Applying Savitzky-Golay filter to the resampled data
    z_resampled_smooth = savgol_filter(z_resampled, window_length, poly_order)
    y_resampled_smooth = savgol_filter(y_resampled, window_length, poly_order)
    x_resampled_smooth = savgol_filter(x_resampled, window_length, poly_order)
    
    return np.column_stack((x_resampled_smooth, y_resampled_smooth, z_resampled_smooth))

# TODO Docstring
def add_spline_end_points(ordered_spline_coordinates, surface_mesh, n_points=2, n_points_averaging=10):
    start_points = make_intermediate_spline_end_points(ordered_spline_coordinates, surface_mesh, n_points, n_points_averaging, start = True)
    end_points = make_intermediate_spline_end_points(ordered_spline_coordinates, surface_mesh, n_points, n_points_averaging, start = False)
    
    return np.concatenate((start_points,ordered_spline_coordinates,end_points),axis=0)

# PROCESSING WITH CURVE
# --------------------------------------------------------------------------------------------

# TODO Docstring
def slice_mesh_along_curve(
        mesh, 
        curve_function, 
        ticks, 
        n_segments = 10, 
        search_space = [-1,2],
        translation_func = None
    ):
    
    intersect_obj = find_parametric_curve_mesh_intersects(mesh, curve_function,search_space)
    
    if translation_func is None:
        translation_func = create_translation_function(curve_function,ticks,intersect_obj.parameters)

    planes = find_planes_along_parametric_curve(
        curve_function,
        n_segments+1,
        translation_function=translation_func,
    )
    
    slices = []
    for i in range(len(planes)-1):
        if i == 0:
            slice = slice_mesh_with_planes(mesh,[planes[i+1]],normal_factors=[-1])
        elif i == len(planes)-2:
            slice = slice_mesh_with_planes(mesh,[planes[i]])
        else:
            slice = slice_mesh_with_planes(mesh, planes[i:i+2])
        slices.append(slice)

    return slices

# TODO Docstring
def find_planes_along_parametric_curve(
        curve_function,
        n_planes,
        translation_function = None,
        plane_size = (400,400)
    ):
    planes = []
    
    for t in np.linspace(0,1,n_planes):
        if translation_function is not None:
            t = translation_function(t)
        
        # Assuming curve_point and tangent_at_point are already defined
        plane = vedo.Plane(
            pos=curve_function(t), 
            normal=curve_tangent(curve_function, t), 
            s = plane_size
        )
        planes.append(plane)

    return planes

# TODO Docstring
def slice_mesh_with_planes(mesh,planes, normal_factors = [1,-1]):
    sliced = vedo.mesh.Mesh(mesh)

    for plane, factor in zip(planes,normal_factors):
        sliced.cut_with_plane(plane.pos(),plane.normal * factor)
        sliced.fill_holes(size=10000)

    print(f"Closed: {sliced.is_closed()}")

    return sliced

# TODO Docstring
def measure_normalised_distances(coordinates,curve_ticks,intersect_paramameters,surface_mesh):
    curve_function = make_bspline_from_ticks(curve_ticks)
    inverse_translation_func = create_translation_function(curve_function,curve_ticks, intersect_paramameters, inverse=True)
    array_1, array_2 = nppas.to_napari_surface_data(surface_mesh)

    start = time.perf_counter()

    numbers_tqdm = tqdm(coordinates)
    # Initialize pool and run computations, collecting results in a list
    results = []
    with cf.ProcessPoolExecutor() as pool:
        results = pool.map(
            partial(
                measure_normalised_point_distance,
                **{"array1":array_1,
                   "array2":array_2,
                   "curve_function":curve_function,
                   "search_space":intersect_paramameters,
                   "parameter_translation_func":inverse_translation_func}
            ),
            numbers_tqdm
        )
    result_grabbed = [res for res in results]
    end = time.perf_counter()
    print(f"processing took {int((end-start)/60)} min, {(end-start)%60} s")

    return np.array(result_grabbed)

# HELPER FUNCTIONS TODO Decide which ones can go into utils
# ---------------------------------------------------------------------------------------------------------------------

# TODO Docstring
def curve_tangent(parametric_curve_function, t):
    # Calculate the derivative of your curve function
    # For now, I'll use a numerical approximation:
    delta = 1e-5
    tangent = (
        np.array(parametric_curve_function(t + delta)) - np.array(parametric_curve_function(t))
    ) / delta
    return tangent / np.linalg.norm(tangent)  # Normalize the tangent

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
    start_idx = do_kdtree(points, [start_coord])[0]
    
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

# TODO Docstring, maybe move somewhere else
def is_list_like(obj):
    return np.iterable(obj) and not isinstance(obj, (str, bytes))

# TODO Docstring
def do_kdtree(coordinates,point_list):
    mytree = cKDTree(coordinates)
    dist, indexes = mytree.query(point_list)
    return indexes

# TODO Docstring
def compute_cumulative_distance(x, y, z):
    """Compute the cumulative distance along a curve defined by x, y, z coordinates."""
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    return np.insert(np.cumsum(distances), 0, 0)

# TODO Docstring
def compute_arc_length_between_points(tck_x, tck_y, tck_z, t_start, t_end):
    """Compute the arc length of a B-spline between two parameter values."""
    
    # Compute the derivative of the B-spline
    def derivative(t):
        dx = interpolate.splev(t, tck_x, der=1)
        dy = interpolate.splev(t, tck_y, der=1)
        dz = interpolate.splev(t, tck_z, der=1)
        return np.array([dx, dy, dz])
    
    # Compute the speed (magnitude of the derivative) of the B-spline
    def speed(t):
        deriv = derivative(t)
        return np.sqrt(np.sum(deriv**2))
    
    # Integrate the speed between t_start and t_end to get the arc length
    arc_length, _ = integrate.quad(speed, t_start, t_end)
    return arc_length

# TODO Docstring
def make_norm_vector(point1, point2):
    vector = point2 - point1
    return vector / np.linalg.norm(vector)

# TODO Docstring
def make_intermediate_spline_end_points(ordered_spline_coordinates, surface_mesh, n_points=2, n_points_averaging=10, start = True):
    point1 = ordered_spline_coordinates[0]
    point2 = np.mean(ordered_spline_coordinates[1:n_points_averaging+1], axis=0)
    if not start:
        point1 = ordered_spline_coordinates[-1]
        point2 = np.mean(
            ordered_spline_coordinates[len(ordered_spline_coordinates)-n_points_averaging-1:-1], 
            axis=0
        )
    
    vector = make_norm_vector(point2,point1)
    intersect = surface_mesh.intersect_with_line(point1,point1 + vector*1000)
    
    distance = np.linalg.norm(point1-intersect)
    print(distance)
    
    points = []
    for i in range(n_points):
        points.append(point1 + vector*(i+1)*(distance/ (n_points+1)))
    print(points)    
    return points

#TODO DOcstring
def curve_distance(t, P, C):
    return np.linalg.norm(C(t) - P)

# Distance to surface as objective function
# TODO Docstring
def surface_distance(t, P, C, surface_mesh: vedo.mesh.Mesh):
    # Note: Update this function to compute the distance to the surface along vector V
    V = P - C(t)
    #print(np.squeeze(C(t)),np.squeeze(C(t) + V*1000))
    surface_intersect = surface_mesh.intersect_with_line(np.squeeze(C(t)),np.squeeze(C(t) + V*1000000))
    #print(f"intersect: {surface_intersect}")

    # This takes care of the case that for the given point no surface intersect exists
    if len(np.squeeze(surface_intersect)) == 0:
        return np.inf
    
    # This takes care of the case when we have multiple intersects
    elif len (surface_intersect) > 1:
        #print("possibly problematic")
        lengths = np.array([np.linalg.norm(inters - C(t)) for inters in surface_intersect])
        
        # Find the intersection closest to the curve
        surface_intersect = surface_intersect[np.argmin(lengths)]

    dist_surf = np.linalg.norm(P - np.squeeze(surface_intersect))
    
    # handles the case that our intersect is closer to curve than our point
    # if curve_distance(t,surface_intersect,C) < curve_distance(t,P,C):
    #    return np.inf
    
    return dist_surf

# TODO decide if needed. This function can handle the case that there are two surface intersects
# and it will ignore the one which is closer to the curve than the point of interest
def surface_distance_more_complex(t, P, C, surface_mesh: vedo.mesh.Mesh):
    # Note: Update this function to compute the distance to the surface along vector V
    V = P - C(t)
    #print(np.squeeze(C(t)),np.squeeze(C(t) + V*1000))
    surface_intersect = surface_mesh.intersect_with_line(np.squeeze(C(t)),np.squeeze(C(t) + V*1000000))
    #print(f"intersect: {surface_intersect}")

    # This takes care of the case that for the given point no surface intersect exists
    if len(np.squeeze(surface_intersect)) == 0:
        return np.inf
    
    # This takes care of the case when we have multiple intersects
    elif len (surface_intersect) > 1:
        #print("possibly problematic")
        lengths = np.array([np.linalg.norm(inters - C(t)) for inters in surface_intersect])

        lengths_to_curve = np.array([curve_distance(t,inters,C) for inters in surface_intersect])

        # Remove any intersections that are closer to the curve than our point of interest
        surface_intersect_no_impossible = np.array(surface_intersect)[lengths_to_curve>curve_distance(t,P,C)]

        # If we removed the only intersection this cannot be the point we are looking for
        if len(np.squeeze(surface_intersect_no_impossible)):
            return np.inf
        
        # Find the intersection closest to the curve but not closer than our point
        surface_intersect = surface_intersect_no_impossible[
            np.argmin(lengths[lengths_to_curve>curve_distance(t,P,C)])
        ]

    dist_surf = np.linalg.norm(P - np.squeeze(surface_intersect))
    
    # handles the case that our intersect is closer to curve than our point
    if curve_distance(t,surface_intersect,C) < curve_distance(t,P,C):
        return np.inf
    
    return dist_surf


# TODO Docstring
def combined_distance(t,P,C,surface_mesh,w1 = 0.5, w2 = 0.5):
    return w1 *curve_distance(t, P, C) + w2 * surface_distance(t, P, C, surface_mesh)

# TODO Docstring
def find_projected_point_combined(curve_func, point, surface_mesh, bounds): 
    result = differential_evolution(combined_distance, [bounds], args=(point, curve_func, surface_mesh))
    return result.x[0]

# TODO Docstring
def measure_normalised_point_distance(
    point,
    array1,
    array2,
    curve_function,
    search_space,
    parameter_translation_func = None,
    line_length = 1000000
):
    complex_obj = nppas.to_vedo_mesh((array1,array2))
    curve_parameter = find_projected_point_combined(curve_function,point,complex_obj,search_space)
    projected_point = np.squeeze(np.array(curve_function(curve_parameter)))
    
    curve_surf_vector = point - projected_point
    curve_surf_vector = curve_surf_vector / np.linalg.norm(curve_surf_vector)

    surface_intersect = complex_obj.intersect_with_line(projected_point,projected_point + curve_surf_vector*line_length)
    if len (surface_intersect) > 1:
        lengths = [np.linalg.norm(inters - projected_point) for inters in surface_intersect]
        surface_intersect = surface_intersect[np.argmin(lengths)]
    surface_intersect = np.squeeze(surface_intersect)
    
    dist_point = np.linalg.norm(projected_point - point)
    dist_surf = np.linalg.norm(projected_point - surface_intersect)

    normalised_dist = dist_point/dist_surf
    
    ap_dist = curve_parameter
    if parameter_translation_func is not None:
        ap_dist = parameter_translation_func(curve_parameter)
    
    return ap_dist, normalised_dist    


#tipps simon: distances vorprogrammieren (not python)
# weirdes coordinatensystem auf der surface womit man nur noch distance zur surface brauch
# bounds einschraenken mit volume slices

