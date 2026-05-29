import numpy as np
from dataclasses import dataclass


@dataclass
class CameraModel:
    image_width: int
    image_height: int
    fov_x_deg: float = None
    fov_y_deg: float = None
    scale_arcsec_per_pixel: float = None

    def focal_lengths_pixels(self):
        if self.fov_x_deg is not None and self.fov_y_deg is not None:
            fx = (self.image_width / 2) / np.tan(np.deg2rad(self.fov_x_deg / 2))
            fy = (self.image_height / 2) / np.tan(np.deg2rad(self.fov_y_deg / 2))
            return fx, fy
        if self.scale_arcsec_per_pixel is not None:
            scale_rad_per_pixel = np.deg2rad(self.scale_arcsec_per_pixel / 3600)
            f = 1 / np.tan(scale_rad_per_pixel)
            return f, f

        raise ValueError("Provide either FOV or angular scale.")


def normalize(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def radec_to_inertial_vector(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    return normalize([x, y, z])


def centroid_to_camera_vector(x_pixel, y_pixel, camera: CameraModel):
    fx, fy = camera.focal_lengths_pixels()

    cx = camera.image_width / 2
    cy = camera.image_height / 2

    x_cam = (x_pixel - cx) / fx
    y_cam = -(y_pixel - cy) / fy
    z_cam = 1.0

    return normalize([x_cam, y_cam, z_cam])


def generate_vectors(centroid_list, identified_list, camera: CameraModel):
    identified_dict = {star["id"]: star for star in identified_list}

    vector_pairs = []

    for centroid in centroid_list:
        star_id = centroid["id"]

        if star_id not in identified_dict:
            continue

        x = centroid["x"]
        y = centroid["y"]

        ra = identified_dict[star_id]["ra_deg"]
        dec = identified_dict[star_id]["dec_deg"]

        camera_vec = centroid_to_camera_vector(x, y, camera)
        inertial_vec = radec_to_inertial_vector(ra, dec)

        vector_pairs.append({
            "id": star_id,
            "camera_vector": camera_vec,
            "inertial_vector": inertial_vec
        })

    return vector_pairs