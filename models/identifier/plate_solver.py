# Standard imports:
from pathlib import Path
import logging
import itertools
from time import perf_counter as precision_timestamp
from numbers import Number
import math

# External imports:
import numpy as np
from numpy.linalg import norm, lstsq
import scipy.stats
import scipy
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist

_MAGIC_RAND = np.uint64(2654435761)
_supported_databases = ('bsc5', 'hip_main', 'tyc_main')

# Helper functions:

def _insert_at_index(pattern, hash_index, table):
    """Inserts to table with quadratic probing. Returns table index where pattern was inserted."""
    max_ind = np.uint64(table.shape[0])
    hash_index = np.uint64(hash_index)
    for c in itertools.count():
        c = np.uint64(c)
        i = (hash_index + c*c) % max_ind
        if all(table[i, :] == 0):
            table[i, :] = pattern
            return i

def _get_table_index_from_hash(hash_index, table):
    """Gets from table with quadratic probing, returns list of all possibly matching indices."""
    max_ind = np.uint64(table.shape[0])
    hash_index = np.uint64(hash_index)
    found = []
    for c in itertools.count():
        c = np.uint64(c)
        i = (hash_index + c*c) % max_ind
        if all(table[i, :] == 0):
            return np.array(found)
        else:
            found.append(i)

def _key_to_index(key, bin_factor, max_index):
    """Get hash index for a given key. Can be length p list or n by p array."""
    key = np.uint64(key)
    bin_factor = np.uint64(bin_factor)
    max_index = np.uint64(max_index)
    # If p is the length of the key (default 5) and B is the number of bins (default 50,
    # calculated from max error), this will first give each key a unique index from
    # 0 to B^p-1, then multiply by large number and modulo to max index to randomise.
    if key.ndim == 1:
        hash_indices = np.sum(key*bin_factor**np.arange(len(key), dtype=np.uint64),
                              dtype=np.uint64)
    else:
        hash_indices = np.sum(key*bin_factor**np.arange(key.shape[1], dtype=np.uint64)[None, :],
                              axis=1, dtype=np.uint64)
    with np.errstate(over='ignore'):
        hash_indices = (hash_indices*_MAGIC_RAND) % max_index
    return hash_indices

def _compute_vectors(centroids, size, fov):
    """Get unit vectors from star centroids (pinhole camera)."""
    # compute list of (i,j,k) vectors given list of (y,x) star centroids and
    # an estimate of the image's field-of-view in the x dimension
    # by applying the pinhole camera equations
    centroids = np.array(centroids, dtype=np.float32)
    (height, width) = size[:2]
    scale_factor = np.tan(fov/2)/width*2
    star_vectors = np.ones((len(centroids), 3))
    # Pixel centre of image
    img_center = [height/2, width/2]
    # Calculate normal vectors
    star_vectors[:, 2:0:-1] = (img_center - centroids) * scale_factor
    star_vectors = star_vectors / norm(star_vectors, axis=1)[:, None]
    return star_vectors

def _compute_centroids(vectors, size, fov, trim=True):
    """Get (undistorted) centroids from a set of (derotated) unit vectors
    vectors: Nx3 of (i,j,k) where i is boresight, j is x (horizontal)
    size: (height, width) in pixels.
    fov: horizontal field of view in radians.
    trim: only keep ones within the field of view, also returns list of indices kept
    """
    (height, width) = size[:2]
    scale_factor = -width/2/np.tan(fov/2)
    centroids = scale_factor*vectors[:, 2:0:-1]/vectors[:, [0]]
    centroids += [height/2, width/2]
    if not trim:
        return centroids
    else:
        keep = np.flatnonzero(np.logical_and(
            np.all(centroids > [0, 0], axis=1),
            np.all(centroids < [height, width], axis=1)))
        return (centroids[keep, :], keep)

def _undistort_centroids(centroids, size, k):
    """Apply r_u = r_d(1 - k'*r_d^2)/(1 - k) undistortion, where k'=k*(2/width)^2,
    i.e. k is the distortion that applies width/2 away from the centre.
    centroids: Nx2 pixel coordinates (y, x), (0.5, 0.5) top left pixel centre.
    size: (height, width) in pixels.
    k: distortion, negative is barrel, positive is pincushion
    """
    centroids = np.array(centroids, dtype=np.float32)
    (height, width) = size[:2]
    # Centre
    centroids -= [height/2, width/2]
    # Scale
    scale = (1 - k*(norm(centroids, axis=1)/width*2)**2)/(1 - k)
    centroids *= scale[:, None]
    # Decentre
    centroids += [height/2, width/2]
    return centroids

def _distort_centroids(centroids, size, k, tol=1e-6, maxiter=30):
    """Distort centroids corresponding to r_u = r_d(1 - k'*r_d^2)/(1 - k),
    where k'=k*(2/width)^2 i.e. k is the distortion that applies
    width/2 away from the centre.

    Iterates with Newton-Raphson until the step is smaller than tol
    or maxiter iterations have been exhausted.
    """
    centroids = np.array(centroids, dtype=np.float32)
    (height, width) = size[:2]
    # Centre
    centroids -= [height/2, width/2]
    r_undist = norm(centroids, axis=1)/width*2
    # Initial guess, distorted are the same positon
    r_dist = r_undist.copy()
    for i in range(maxiter):
        r_undist_est = r_dist*(1 - k*r_dist**2)/(1 - k)
        dru_drd = (1 - 3*k*r_dist**2)/(1 - k)
        error = r_undist - r_undist_est
        r_dist += error/dru_drd

        if np.all(np.abs(error) < tol):
            break

    centroids *= (r_dist/r_undist)[:, None]
    centroids += [height/2, width/2]
    return centroids

def _find_rotation_matrix(image_vectors, catalog_vectors):
    """Calculate the least squares best rotation matrix between the two sets of vectors.
    image_vectors and catalog_vectors both Nx3. Must be ordered as matching pairs.
    """
    # find the covariance matrix H between the image and catalog vectors
    H = np.dot(image_vectors.T, catalog_vectors)
    # use singular value decomposition to find the rotation matrix
    (U, S, V) = np.linalg.svd(H)
    return np.dot(U, V)

def _find_centroid_matches(image_centroids, catalog_centroids, r):
    """Find matching pairs, unique and within radius r
    image_centroids: Nx2 (y, x) in pixels
    catalog_centroids: Mx2 (y, x) in pixels
    r: radius in pixels

    returns Kx2 list of matches, first colum is index in image_centroids,
        second column is index in catalog_centroids
    """
    dists = cdist(image_centroids, catalog_centroids)
    matches = np.argwhere(dists < r)
    # Make sure we only have unique 1-1 matches
    matches = matches[np.unique(matches[:, 0], return_index=True)[1], :]
    matches = matches[np.unique(matches[:, 1], return_index=True)[1], :]
    return matches

# The main algorithm, including plate solving & managing databases
class PlateSolver():
    """Solve star patterns and manage databases. For more information, please visit the open source https://github.com/esa/tetra3/blob/master/tetra3/tetra3.py"""

    def __init__(self, load_database='default_database', debug_folder=None):
        # Logger setup
        self._debug_folder = None
        self._logger = logging.getLogger('plateSolver.PlateSolver')
        if not self._logger.hasHandlers():
            # Add new handlers to the logger if there are none
            self._logger.setLevel(logging.DEBUG)
            # Console handler at INFO level
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            # Format and add
            formatter = logging.Formatter('%(asctime)s:%(name)s-%(levelname)s: %(message)s')
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)
            if debug_folder is not None:
                self.debug_folder = debug_folder
                # File handler at DEBUG level
                fh = logging.FileHandler(self.debug_folder / 'tetra3.txt')
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(formatter)
                self._logger.addHandler(fh)

        self._logger.debug('Tetra3 Constructor called with load_database=' + str(load_database))
        self._star_table = None
        self._star_catalog_IDs = None
        self._pattern_catalog = None
        self._pattern_largest_edge = None
        self._verification_catalog = None
        self._db_props = {'pattern_mode': None, 'pattern_size': None, 'pattern_bins': None,
                          'pattern_max_error': None, 'max_fov': None, 'min_fov': None,
                          'star_catalog': None, 'epoch_equinox': None, 'epoch_proper_motion': None,
                          'pattern_stars_per_fov': None, 'verification_stars_per_fov': None,
                          'star_max_magnitude': None, 'simplify_pattern': None,
                          'range_ra': None, 'range_dec': None, 'presort_patterns': None}

        if load_database is not None:
            self._logger.debug('Trying to load database')
            self.load_database(load_database)

    @property
    def debug_folder(self):
        """pathlib.Path: Get or set the path for debug logging. Will create folder if not existing.
        """
        return self._debug_folder

    @debug_folder.setter
    def debug_folder(self, path):
        # Do not do logging in here! This will be called before the logger is set up
        assert isinstance(path, Path), 'Must be pathlib.Path object'
        if path.is_file():
            path = path.parent
        if not path.is_dir():
            path.mkdir(parents=True)
        self._debug_folder = path

    @property
    def has_database(self):
        """bool: True if a database is loaded."""
        return not (self._star_table is None or self._pattern_catalog is None)

    @property
    def star_table(self):
        """numpy.ndarray: Table of stars in the database.

        The table is an array with six columns:
            - Right ascension (radians)
            - Declination (radians)
            - x = cos(ra) * cos(dec)
            - y = sin(ra) * cos(dec)
            - z = sin(dec)
            - Apparent magnitude
        """
        return self._star_table

    @property
    def pattern_catalog(self):
        """numpy.ndarray: Catalog of patterns in the database."""
        return self._pattern_catalog

    @property
    def pattern_largest_edge(self):
        """numpy.ndarray: Catalog of largest edges for each pattern in milliradian."""
        return self._pattern_largest_edge

    @property
    def star_catalog_IDs(self):
        """numpy.ndarray: Table of catalogue IDs for each entry in the star table.

        The table takes different format depending on the source catalogue used
        to build the database. See the `star_catalog` key of
        :meth:`database_properties` to find the source catalogue.
            - bsc5: A numpy array of size (N,) with datatype uint16. Stores the 'BSC' number.
            - hip_main: A numpy array of size (N,) with datatype uint32. Stores the 'HIP' number.
            - tyc_main: A numpy array of size (N, 3) with datatype uint16. Stores the
              (TYC1, TYC2, TYC3) numbers.

        Is None if no database is loaded or an older database without IDs stored.
        """
        return self._star_catalog_IDs

    @property
    def database_properties(self):
        """dict: Dictionary of database properties. Please visit the open source for the details: https://github.com/esa/tetra3/blob/master/tetra3/tetra3.py"""
        return self._db_props

    def load_database(self, path='default_database'):
        """Load database from file.

        Args:
            path (str or pathlib.Path): The file to load. If given a str, the file will be looked
                for in the tetra3/data directory. If given a pathlib.Path, this path will be used
                unmodified. The suffix .npz will be added.
        """
        self._logger.debug('Got load database with: ' + str(path))
        if isinstance(path, str):
            self._logger.debug('String given, append to tetra3 directory')
            path = (Path(__file__).parent / 'data' / path).with_suffix('.npz')
        else:
            self._logger.debug('Not a string, use as path directly')
            path = Path(path).with_suffix('.npz')

        self._logger.info('Loading database from: ' + str(path))
        with np.load(path) as data:
            self._logger.debug('Loaded database, unpack files')
            self._pattern_catalog = data['pattern_catalog']
            self._star_table = data['star_table']
            props_packed = data['props_packed']
            try:
                self._pattern_largest_edge = data['pattern_largest_edge']
            except KeyError:
                self._logger.debug('Database does not have largest edge stored, set to None.')
                self._pattern_largest_edge = None
            try:
                self._star_catalog_IDs = data['star_catalog_IDs']
            except KeyError:
                self._logger.debug('Database does not have catalogue IDs stored, set to None.')
                self._star_catalog_IDs = None

        self._logger.debug('Unpacking properties')
        for key in self._db_props.keys():
            try:
                self._db_props[key] = props_packed[key][()]
                self._logger.debug('Unpacked ' + str(key)+' to: ' + str(self._db_props[key]))
            except ValueError:
                if key == 'verification_stars_per_fov':
                    self._db_props[key] = props_packed['catalog_stars_per_fov'][()]
                    self._logger.debug('Unpacked catalog_stars_per_fov to: ' \
                        + str(self._db_props[key]))
                elif key == 'star_max_magnitude':
                    self._db_props[key] = props_packed['star_min_magnitude'][()]
                    self._logger.debug('Unpacked star_min_magnitude to: ' \
                        + str(self._db_props[key]))
                elif key == 'presort_patterns':
                    self._db_props[key] = False
                    self._logger.debug('No presort_patterns key, set to False')
                elif key == 'star_catalog':
                    self._db_props[key] = 'unknown'
                    self._logger.debug('No star_catalog key, set to unknown')
                else:
                    self._db_props[key] = None
                    self._logger.warning('Missing key in database (likely version difference): ' + str(key))
        if self._db_props['min_fov'] is None:
            self._logger.debug('No min_fov key, copy from max_fov')
            self._db_props['min_fov'] = self._db_props['max_fov']


    def solve_from_centroids(self, star_centroids, size, fov_estimate=None, fov_max_error=None,
                             pattern_checking_stars=8, match_radius=.01, match_threshold=1e-3,
                             solve_timeout=None, target_pixel=None, distortion=None,
                             return_matches=True, return_visual=False):
        """Solve for the sky location using a list of centroids. For the details and examples, please visit the open source: https://github.com/esa/tetra3/blob/master/tetra3/tetra3.py"""

        assert self.has_database, 'No database loaded'
        self._logger.debug('Got solve from centroids with input: '
                           + str((len(star_centroids), size, fov_estimate, fov_max_error,
                                 pattern_checking_stars, match_radius, match_threshold,
                                 solve_timeout, target_pixel, distortion,
                                 return_matches, return_visual)))

        image_centroids = np.asarray(star_centroids)
        image_centroids = image_centroids[:, [1, 0]] # convert x,y -> y,x 
        if fov_estimate is None:
            # If no FOV given at all, guess middle of the range for a start
            fov_initial = np.deg2rad((self._db_props['max_fov'] + self._db_props['min_fov'])/2)
        else:
            fov_estimate = np.deg2rad(float(fov_estimate))
            fov_initial = fov_estimate
        if fov_max_error is not None:
            fov_max_error = np.deg2rad(float(fov_max_error))
        match_radius = float(match_radius)
        num_patterns = self.pattern_catalog.shape[0] // 2
        match_threshold = float(match_threshold) / num_patterns
        self._logger.debug('Set threshold to: ' + str(match_threshold) + ', have '
            + str(num_patterns) + ' patterns.')
        pattern_checking_stars = int(pattern_checking_stars)
        if solve_timeout is not None:
            # Convert to seconds to match timestamp
            solve_timeout = float(solve_timeout) / 1000
        if target_pixel is not None:
            target_pixel = np.array(target_pixel)
            if target_pixel.ndim == 1:
                # Make shape (2,) array to (1,2), to match (N,2) pattern
                target_pixel = target_pixel[None, :]
        return_matches = bool(return_matches)

        # extract height (y) and width (x) of image
        (height, width) = size[:2]
        # Extract relevant database properties
        num_stars = self._db_props['verification_stars_per_fov']
        p_size = self._db_props['pattern_size']
        p_bins = self._db_props['pattern_bins']
        p_max_err = self._db_props['pattern_max_error']
        presorted = self._db_props['presort_patterns']
        upper_tri_index = np.triu_indices(p_size, 1)

        image_centroids = image_centroids[:num_stars, :]
        self._logger.debug('Trimmed centroid input shape to: ' + str(image_centroids.shape))
        t0_solve = precision_timestamp()

        # If distortion is not None, we need to do some prep work
        if isinstance(distortion, Number):
            # If known distortion, undistort centroids, then proceed as normal
            image_centroids = _undistort_centroids(image_centroids, (height, width), k=distortion)
            self._logger.debug('Undistorted centroids with k=%d' % distortion)
        elif isinstance(distortion, (list, tuple)):
            # If given range, need to predistort for future calculations
            # Make each step at most 0.1 (10%) distortion
            distortion_range = np.linspace(min(distortion), max(distortion),
                int(np.ceil(round(max(distortion) - min(distortion), 6)*10) + 1))
            self._logger.debug('Searching distortion range: ' + str(np.round(distortion_range, 6)))
            image_centroids_preundist = np.zeros((len(distortion_range),) + image_centroids.shape)
            for (i, k) in enumerate(distortion_range):
                image_centroids_preundist[i, :] = _undistort_centroids(
                    image_centroids, (height, width), k=k)

        # Try all combinations of p_size of pattern_checking_stars brightest
        for image_pattern_indices in itertools.combinations(
                range(min(len(image_centroids), pattern_checking_stars)), p_size):
            image_pattern_centroids = image_centroids[image_pattern_indices, :]
            # Check if timeout has elapsed, then we must give up
            if solve_timeout is not None:
                elapsed_time = precision_timestamp() - t0_solve
                if elapsed_time > solve_timeout:
                    self._logger.debug('Timeout reached after: ' + str(elapsed_time) + 's.')
                    break
            # Set largest distance to None, this is cached to avoid recalculating in future FOV estimation.
            pattern_largest_distance = None

            # Now find the possible range of edge ratio patterns these four image centroids
            # could correspond to.
            pattlen = int(math.factorial(p_size) / 2 / math.factorial(p_size - 2) - 1)
            image_pattern_edge_ratio_min = np.ones(pattlen)
            image_pattern_edge_ratio_max = np.zeros(pattlen)

            # No or already known distortion, use directly
            if distortion is None or isinstance(distortion, Number):
                # Compute star vectors using an estimate for the field-of-view in the x dimension
                image_pattern_vectors = _compute_vectors(image_pattern_centroids, (height, width), fov_initial)
                # Calculate what the edge ratios are and add p_max_err tolerance
                edge_angles_sorted = np.sort(2 * np.arcsin(.5 * pdist(image_pattern_vectors)))
                image_pattern_largest_edge = edge_angles_sorted[-1]
                image_pattern = edge_angles_sorted[:-1] / image_pattern_largest_edge
                image_pattern_edge_ratio_min = image_pattern - p_max_err
                image_pattern_edge_ratio_max = image_pattern + p_max_err
            else:
                # Calculate edge ratios for all predistortions, take max/min
                image_pattern_edge_ratio_preundist = np.zeros((len(distortion_range), pattlen))
                for i in range(len(distortion_range)):
                    image_pattern_vectors = _compute_vectors(
                        image_centroids_preundist[i, image_pattern_indices], (height, width), fov_initial)
                    edge_angles_sorted = np.sort(2 * np.arcsin(.5 * pdist(image_pattern_vectors)))
                    image_pattern_largest_edge = edge_angles_sorted[-1]
                    image_pattern_edge_ratio_preundist[i, :] = edge_angles_sorted[:-1] / image_pattern_largest_edge
                image_pattern_edge_ratio_min = np.min(image_pattern_edge_ratio_preundist, axis=0)
                image_pattern_edge_ratio_max = np.max(image_pattern_edge_ratio_preundist, axis=0)

            # Possible range of hash codes we need to look up
            hash_code_space_min = np.maximum(0, image_pattern_edge_ratio_min*p_bins).astype(int)
            hash_code_space_max = np.minimum(p_bins, image_pattern_edge_ratio_max*p_bins).astype(int)
            # Make an array of all combinations
            hash_code_range = list(range(low, high + 1) for (low, high) in zip(hash_code_space_min, hash_code_space_max))
            hash_code_list = np.array(list(code for code in itertools.product(*hash_code_range)))
            # Make sure we have unique ascending codes
            hash_code_list = np.sort(hash_code_list, axis=1)
            hash_code_list = np.unique(hash_code_list, axis=0)

            # Calculate hash index for each
            hash_indices = _key_to_index(hash_code_list, p_bins, self.pattern_catalog.shape[0])
            # iterate over hash code space
            i = 1
            for hash_index in hash_indices:
                hash_match_inds = _get_table_index_from_hash(hash_index, self.pattern_catalog)
                if len(hash_match_inds) == 0:
                    continue

                if self.pattern_largest_edge is not None \
                        and fov_estimate is not None \
                        and fov_max_error is not None:
                    # Can immediately compare FOV to patterns to remove mismatches
                    largest_edge = self.pattern_largest_edge[hash_match_inds]
                    fov2 = largest_edge / image_pattern_largest_edge * fov_initial / 1000
                    keep = abs(fov2 - fov_estimate) < fov_max_error
                    hash_match_inds = hash_match_inds[keep]
                    if len(hash_match_inds) == 0:
                        continue
                catalog_matches = self.pattern_catalog[hash_match_inds, :]

                # Get star vectors for all matching hashes
                all_catalog_pattern_vectors = self.star_table[catalog_matches, 2:5]
                # Calculate pattern by angles between vectors
                # this is a bit manual, I could not see a faster way
                arr1 = np.take(all_catalog_pattern_vectors, upper_tri_index[0], axis=1)
                arr2 = np.take(all_catalog_pattern_vectors, upper_tri_index[1], axis=1)
                catalog_pattern_edges = np.sort(norm(arr1 - arr2, axis=-1))
                # implement more accurate angle calculation
                catalog_pattern_edges = 2 * np.arcsin(.5 * catalog_pattern_edges)

                all_catalog_largest_edges = catalog_pattern_edges[:, -1]
                all_catalog_edge_ratios = catalog_pattern_edges[:, :-1] / all_catalog_largest_edges[:, None]

                # Compare catalogue edge ratios to the min/max range from the image pattern
                valid_patterns = np.argwhere(np.all(np.logical_and(
                    image_pattern_edge_ratio_min < all_catalog_edge_ratios,
                    image_pattern_edge_ratio_max > all_catalog_edge_ratios), axis=1)).flatten()

                # Go through each matching pattern and calculate further
                for index in valid_patterns:
                    # Estimate coarse distortion from the pattern
                    if distortion is None or isinstance(distortion, Number):
                        # Distortion is known, set variables and estimate FOV
                        image_centroids_undist = image_centroids
                    else:
                        # Calculate the (coarse) distortion by comparing pattern to the min/max distorted patterns
                        edge_ratio_errors_preundist = all_catalog_edge_ratios[index] - image_pattern_edge_ratio_preundist
                        # Now find the two indices in preundist that are closest to the real distortion
                        if len(distortion_range) > 2:
                            # If there are more than 2 preundistortions, select the two closest ones for interpolation
                            rmserr = np.sum(edge_ratio_errors_preundist**2, axis=1)
                            closest = np.argmin(rmserr)
                            if closest == 0:
                                # Use first two
                                low_ind = 0
                                high_ind = 1
                            elif closest == (len(distortion_range) - 1):
                                # Use last two
                                low_ind = len(distortion_range) - 2
                                high_ind = len(distortion_range) - 1
                            else:
                                if rmserr[closest + 1] > rmserr[closest - 1]:
                                    # Use closest and the one after
                                    low_ind = closest
                                    high_ind = closest + 1
                                else:
                                    # Use closest and the one before
                                    low_ind = closest - 1
                                    high_ind = closest
                        else:
                            # If just two preundistortions, set the variables
                            low_ind = 0
                            high_ind = 1
                        # How far do we need to go from low to high to reach zero
                        x = np.mean(edge_ratio_errors_preundist[low_ind, :]
                            /(edge_ratio_errors_preundist[low_ind, :] - edge_ratio_errors_preundist[high_ind, :]))
                        # Distortion k estimate
                        dist_est = distortion_range[low_ind] + x*(distortion_range[high_ind] - distortion_range[low_ind])
                        # Undistort centroid pattern with estimate
                        image_centroids_undist = _undistort_centroids(image_centroids, (height, width), k=dist_est)

                    # Estimate coarse FOV from the pattern
                    catalog_largest_edge = all_catalog_largest_edges[index]
                    if fov_estimate is not None and (distortion is None or isinstance(distortion, Number)):
                        # Can quickly correct FOV by scaling given estimate
                        fov = catalog_largest_edge / image_pattern_largest_edge * fov_initial
                    else:
                        # Use camera projection to calculate actual fov
                        if distortion is None or isinstance(distortion, Number):
                            # The FOV estimate will be the same for each attempt with this pattern
                            # so we can cache the value by checking if we have already set it
                            if pattern_largest_distance is None:
                                pattern_largest_distance = np.max(pdist(image_centroids_undist[image_pattern_indices, :]))
                        else:
                            # If distortion is allowed to vary, we need to calculate this every time
                            pattern_largest_distance = np.max(pdist(image_centroids_undist[image_pattern_indices, :]))
                        f = pattern_largest_distance / 2 / np.tan(catalog_largest_edge/2)
                        fov = 2*np.arctan(width/2/f)

                    # If the FOV is incorrect we can skip this immediately
                    if fov_estimate is not None and fov_max_error is not None \
                            and abs(fov - fov_estimate) > fov_max_error:
                        continue

                    # Recalculate vectors and uniquely sort them by distance from centroid
                    image_pattern_vectors = _compute_vectors(
                        image_centroids_undist[image_pattern_indices, :], (height, width), fov)
                    # find the centroid, or average position, of the star pattern
                    pattern_centroid = np.mean(image_pattern_vectors, axis=0)
                    # calculate each star's radius, or Euclidean distance from the centroid
                    pattern_radii = cdist(image_pattern_vectors, pattern_centroid[None, :]).flatten()
                    # use the radii to uniquely order the pattern's star vectors so they can be
                    # matched with the catalog vectors
                    image_pattern_vectors = np.array(image_pattern_vectors)[np.argsort(pattern_radii)]

                    # Now get pattern vectors from catalogue, and sort if necessary
                    catalog_pattern_vectors = all_catalog_pattern_vectors[index, :]
                    if not presorted:
                        # find the centroid, or average position, of the star pattern
                        catalog_centroid = np.mean(catalog_pattern_vectors, axis=0)
                        # calculate each star's radius, or Euclidean distance from the centroid
                        catalog_radii = cdist(catalog_pattern_vectors, catalog_centroid[None, :]).flatten()
                        # use the radii to uniquely order the catalog vectors
                        catalog_pattern_vectors = catalog_pattern_vectors[np.argsort(catalog_radii)]

                    # Use the pattern match to find an estimate for the image's rotation matrix
                    rotation_matrix = _find_rotation_matrix(image_pattern_vectors,
                                                            catalog_pattern_vectors)

                    # Find all star vectors inside the (diagonal) field of view for matching
                    image_center_vector = rotation_matrix[0, :]
                    fov_diagonal_rad = fov * np.sqrt(width**2 + height**2) / width
                    nearby_star_inds = self._get_nearby_stars(image_center_vector, fov_diagonal_rad/2)
                    nearby_star_vectors = self.star_table[nearby_star_inds, 2:5]

                    # Derotate nearby stars and get their (undistorted) centroids using coarse fov
                    nearby_star_vectors_derot = np.dot(rotation_matrix, nearby_star_vectors.T).T
                    (nearby_star_centroids, kept) = _compute_centroids(nearby_star_vectors_derot, (height, width), fov)
                    nearby_star_vectors = nearby_star_vectors[kept, :]
                    nearby_star_inds = nearby_star_inds[kept]
                    # Only keep as many as the centroids, they should ideally both be the num_stars brightest
                    nearby_star_centroids = nearby_star_centroids[:len(image_centroids)]
                    nearby_star_vectors = nearby_star_vectors[:len(image_centroids)]
                    nearby_star_inds = nearby_star_inds[:len(image_centroids)]

                    # Match these centroids to the image
                    matched_stars = _find_centroid_matches(image_centroids_undist, nearby_star_centroids, width*match_radius)
                    num_extracted_stars = len(image_centroids)
                    num_nearby_catalog_stars = len(nearby_star_centroids)
                    num_star_matches = len(matched_stars)
                    self._logger.debug("Number of nearby stars: %d, total matched: %d" \
                        % (num_nearby_catalog_stars, num_star_matches))
                    
                    # Probability that a single star is a mismatch (fraction of area that are stars)
                    prob_single_star_mismatch = num_nearby_catalog_stars * match_radius**2
                    # Probability that this rotation matrix's set of matches happen randomly
                    # we subtract two degrees of fredom
                    prob_mismatch = scipy.stats.binom.cdf(num_extracted_stars - (num_star_matches - 2),
                                                          num_extracted_stars,
                                                          1 - prob_single_star_mismatch)
                    self._logger.debug("Mismatch probability = %.2e, at FOV = %.5fdeg" \
                        % (prob_mismatch, np.rad2deg(fov)))

                    if prob_mismatch < match_threshold:
                        # diplay mismatch probability in scientific notation
                        self._logger.debug("MATCH ACCEPTED")
                        self._logger.debug("Prob: %.4g, corr: %.4g"
                            % (prob_mismatch, prob_mismatch*num_patterns))

                        # Get the vectors for all matches in the image using coarse fov
                        matched_image_centroids = image_centroids[matched_stars[:, 0], :]
                        matched_image_vectors = _compute_vectors(matched_image_centroids,
                            (height, width), fov)
                        matched_catalog_vectors = nearby_star_vectors[matched_stars[:, 1], :]
                        # Recompute rotation matrix for more accuracy
                        rotation_matrix = _find_rotation_matrix(matched_image_vectors, matched_catalog_vectors)
                        # extract right ascension, declination, and roll from rotation matrix
                        ra = np.rad2deg(np.arctan2(rotation_matrix[0, 1],
                                                   rotation_matrix[0, 0])) % 360
                        dec = np.rad2deg(np.arctan2(rotation_matrix[0, 2],
                                                    norm(rotation_matrix[1:3, 2])))
                        roll = np.rad2deg(np.arctan2(rotation_matrix[1, 2],
                                                     rotation_matrix[2, 2])) % 360

                        if distortion is None:
                            # Compare mutual angles in catalogue to those with current
                            # FOV estimate in order to scale accurately for fine FOV
                            angles_camera = 2 * np.arcsin(0.5 * pdist(matched_image_vectors))
                            angles_catalogue = 2 * np.arcsin(0.5 * pdist(matched_catalog_vectors))
                            fov *= np.mean(angles_catalogue / angles_camera)
                            k = None
                            matched_image_centroids_undist = matched_image_centroids
                        else:
                            # Accurately calculate the FOV and distortion by looking at the angle from boresight
                            # on all matched catalogue vectors and all matched image centroids
                            matched_catalog_vectors_derot = np.dot(rotation_matrix, matched_catalog_vectors.T).T
                            tangent_matched_catalog_vectors = norm(matched_catalog_vectors_derot[:, 1:], axis=1) \
                                                                  /matched_catalog_vectors_derot[:, 0]
                            # Get the (distorted) pixel distance from image centre for all matches
                            # (scaled relative to width/2)
                            radius_matched_image_centroids = norm(matched_image_centroids
                                                                 - [height/2, width/2], axis=1)/width*2
                            # Solve system of equations in RMS sense for focal length f and distortion k
                            # where f is focal length in units of image width/2
                            # and k is distortion at width/2 (negative is barrel)
                            # undistorted = distorted*(1 - k*(distorted*2/width)^2)
                            A = np.hstack((tangent_matched_catalog_vectors[:, None],
                                           radius_matched_image_centroids[:, None]**3))
                            b = radius_matched_image_centroids[:, None]
                            (f, k) = lstsq(A, b, rcond=None)[0].flatten()
                            # Correct focal length to be at horizontal FOV
                            f = f/(1 - k)
                            self._logger.debug('Calculated focal length to %.2f and distortion to %.3f' % (f, k))
                            # Calculate (horizontal) true field of view
                            fov = 2*np.arctan(1/f)
                            # Undistort centroids for final calculations
                            matched_image_centroids_undist = _undistort_centroids(
                                matched_image_centroids, (height, width), k)

                        # Get vectors
                        final_match_vectors = _compute_vectors(
                            matched_image_centroids_undist, (height, width), fov)
                        # Rotate to the sky
                        final_match_vectors = np.dot(rotation_matrix.T, final_match_vectors.T).T

                        # Calculate residual angles with more accurate formula
                        distance = norm(final_match_vectors - matched_catalog_vectors, axis=1)
                        angle = 2 * np.arcsin(.5 * distance)
                        residual = np.rad2deg(np.sqrt(np.mean(angle**2))) * 3600

                        # Solved in this time
                        t_solve = (precision_timestamp() - t0_solve)*1000
                        solution_dict = {'RA': ra, 'Dec': dec,
                                         'Roll': roll,
                                         'FOV': np.rad2deg(fov), 'distortion': k,
                                         'RMSE': residual,
                                         'Matches': num_star_matches,
                                         'Prob': prob_mismatch*num_patterns,
                                         'epoch_equinox': self._db_props['epoch_equinox'],
                                         'epoch_proper_motion': self._db_props['epoch_proper_motion'],
                                         'T_solve': t_solve}

                        # If we were given target pixel(s), calculate their ra/dec
                        if target_pixel is not None:
                            self._logger.debug('Calculate RA/Dec for targets: '
                                + str(target_pixel))
                            # Calculate the vector in the sky of the target pixel(s)
                            target_pixel = _undistort_centroids(target_pixel, (height, width), k)
                            target_vectors = _compute_vectors(
                                target_pixel, (height, width), fov)
                            rotated_target_vectors = np.dot(rotation_matrix.T, target_vectors.T).T
                            # Calculate and add RA/Dec to solution
                            target_ra = np.rad2deg(np.arctan2(rotated_target_vectors[:, 1],
                                                              rotated_target_vectors[:, 0])) % 360
                            target_dec = 90 - np.rad2deg(
                                np.arccos(rotated_target_vectors[:,2]))

                            if target_ra.shape[0] > 1:
                                solution_dict['RA_target'] = target_ra.tolist()
                                solution_dict['Dec_target'] = target_dec.tolist()
                            else:
                                solution_dict['RA_target'] = target_ra[0]
                                solution_dict['Dec_target'] = target_dec[0]

                        # If requested to return data about matches, append to dict
                        if return_matches:
                            match_data = self._get_matched_star_data(
                                image_centroids[matched_stars[:, 0]], nearby_star_inds[matched_stars[:, 1]])
                            solution_dict.update(match_data)

                        # If requested to create a visualisation, do so and append
                        if return_visual:
                            self._logger.debug('Generating visualisation')
                            img = Image.new('RGB', (width, height))
                            img_draw = ImageDraw.Draw(img)
                            # Make list of matched and not from catalogue
                            matched = matched_stars[:, 1]
                            not_matched = np.array([True]*len(nearby_star_centroids))
                            not_matched[matched] = False
                            not_matched = np.flatnonzero(not_matched)

                            def draw_circle(centre, radius, **kwargs):
                                bbox = [centre[1] - radius,
                                        centre[0] - radius,
                                        centre[1] + radius,
                                        centre[0] + radius]
                                img_draw.ellipse(bbox, **kwargs)

                            for cent in image_centroids:
                                # Centroids with no/given distortion
                                draw_circle(cent, 2, fill='white')
                            for cent in image_centroids_undist:
                                # Image centroids with coarse distortion for matching
                                draw_circle(cent, 1, fill='darkorange')
                            for cent in image_centroids_undist[image_pattern_indices, :]:
                                # Make the pattern ones larger
                                draw_circle(cent, 3, outline='darkorange')
                            for cent in matched_image_centroids_undist:
                                # Centroid position with solution distortion
                                draw_circle(cent, 1, fill='green')
                            for match in matched:
                                # Green circle for succeessful match
                                draw_circle(nearby_star_centroids[match],
                                    width*match_radius, outline='green')
                            for match in not_matched:
                                # Red circle for failed match
                                draw_circle(nearby_star_centroids[match],
                                    width*match_radius, outline='red')

                            solution_dict['visual'] = img

                        self._logger.debug(solution_dict)
                        return solution_dict

        # Failed to solve, get time and return None
        t_solve = (precision_timestamp() - t0_solve) * 1000
        self._logger.debug('FAIL: Did not find a match to the stars! It took '
                           + str(round(t_solve)) + ' ms.')
        return {'RA': None, 'Dec': None, 'Roll': None, 'FOV': None, 'distortion': None,
                'RMSE': None, 'Matches': None, 'Prob': None, 'epoch_equinox': None,
                'epoch_proper_motion': None, 'T_solve': t_solve}

    def _get_nearby_stars(self, vector, radius):
        """Get star indices within radius radians of the vector."""
        # Stars must be within this cartesian cube
        max_dist = 2*np.sin(radius/2)
        range_x = vector[0] + np.array([-max_dist, max_dist])
        range_y = vector[1] + np.array([-max_dist, max_dist])
        range_z = vector[2] + np.array([-max_dist, max_dist])
        # Per axis, find where data is within the range, then combine
        possible_x = (self.star_table[:, 2] > range_x[0]) & (self.star_table[:, 2] < range_x[1])
        possible_y = (self.star_table[:, 3] > range_y[0]) & (self.star_table[:, 3] < range_y[1])
        possible_z = (self.star_table[:, 4] > range_z[0]) & (self.star_table[:, 4] < range_z[1])
        possible = np.nonzero(possible_x & possible_y & possible_z)[0]
        # Find those within the given radius
        nearby = np.dot(np.asarray(vector), self.star_table[possible, 2:5].T) > np.cos(radius)
        return possible[nearby]

    def _get_matched_star_data(self, centroid_data, star_indices):
        """Get dictionary of matched star data to return.

        centroid_data: ndarray of centroid data Nx2, each row (y, x)
        star_indices: ndarray of matching star indices len N

        return dict with keys:
            - matched_centroids: Nx2 (y, x) in pixel coordinates, sorted by brightness
            - matched_stars: Nx3 (ra (deg), dec (deg), magnitude)
            - matched_catID: (N,) or (N, 3) with catalogue ID
        """
        output = {}
        output['matched_centroids'] = centroid_data.tolist()
        stars = self.star_table[star_indices, :][:, [0, 1, 5]]
        stars[:,:2] = np.rad2deg(stars[:,:2])
        output['matched_stars'] = stars.tolist()
        if self.star_catalog_IDs is None:
            output['matched_catID'] = None
        elif len(self.star_catalog_IDs.shape) > 1:
            # Have 2D array, pick rows
            output['matched_catID'] = self.star_catalog_IDs[star_indices, :].tolist()
        else:
            # Have 1D array, pick indices
            output['matched_catID'] = self.star_catalog_IDs[star_indices].tolist()
        return output
    

# Wrapper: 
class PlateSolverWrapper:
    def __init__(self, database_npz_path: str):
        self.engine = PlateSolver(load_database=database_npz_path)

    def solve(
        self,
        detected_list_xy,
        image_size_hw,
        fov_estimate_deg=None,
        fov_max_error_deg=5.0,
    ):
        centroids = np.asarray(detected_list_xy)

        return self.engine.solve_from_centroids(
            centroids,
            image_size_hw,
            fov_estimate=fov_estimate_deg,
            fov_max_error=fov_max_error_deg,
            distortion=None,
            return_matches=True,
            return_visual=False,
        )