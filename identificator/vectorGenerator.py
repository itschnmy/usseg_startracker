import numpy as np
from numpy.linalg import norm

def compute_body_vectors_from_centroids(matched_centroids, image_size, fov):
    # format & unit: yx & pixel, height width & pixel, degree
    # convention of axes: according to Tetra3 Plate Solving algorithm, vec is in the form of 1, delta x, delta y
    # therefore, component 0 or x-axis is the camera boresight direction, pointing forward. The rest points horizontally and vertically, respectively

    centroids = np.asarray(matched_centroids, dtype=np.float32) #transform to a numpy array
    height, width = image_size[:2]
    fov = np.deg2grad(float.fov)

    scale_factor = np.tan(fov/2.0)/width*2.0 #convert pixel displacement in the image to angular displacement -> angle per pixel

    # generate vector array & normalize vectors
    vec = np.ones((len(centroids), 3), dtype=np.float64)
    img_centre = np.array([height/2.0, width/2.0], dtype=np.float64)
    vec[:, 2:0:-1] = (img_centre - centroids) * scale_factor # make sure the order of vector components is y, x
    vec /= norm(v, axis=1)[:, None]

    return vec


def compute_inertial_vectors_from_solution(solution_dict, plate_solver):
    matched_stars = solution_dict.get("matched_stars", None)
    if matched_stars is None:
            raise ValueError("Missing solution")
    
    # right ascension and declination in rad
    ra = np.deg2rad(np.array([row[0] for row in matched_stars], dtype=np.float64))
    dec = np.deg2rad(np.array([row[1] for row in matched_stars], dtype=np.float64))

    # compute vector components in cartesian coordinate system
    x = cos(ra) * cos(dec)
    y = sin(ra) * cos(dec)
    z = sin(dec)

    # normalize
    vec = np.column_stack([x, y, z])
    v /= norm(v, axis=1)[:, None]

    return v


def generate_pairs(solution_dict, image_size):
    fov = solution_dict.get("FOV", None)
    if fov is None:
         raise ValueError("Missing FOV")
    
    matched_centroids = solution_dict.get("matched_centroids", None)
    if matched_centroids is None:
         raise ValueError("Missing matched centroids")
    
    body_vecs = compute_body_vectors_from_centroids(matched_centroids, image_size, fov)
    inertial_vecs = compute_inertial_vectors_from_solution(solution_dict, plate_solver=None)