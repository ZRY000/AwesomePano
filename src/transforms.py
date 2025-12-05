import cv2
import numpy as np

# rotate in yaw
def rotate_pano_yaw(pano, yaw_target_rad):
    """
    Shift/rotate an equirectangular panorama so that `yaw_target_rad`
    becomes the center column of the output image.

    pano: H,W,3
    yaw_target_rad: target yaw in radians
    """
    H, W = pano.shape[:2]

    # compute horizontal pixel shift
    shift_px = int(-yaw_target_rad / (2 * np.pi) * W)

    # circular horizontal rotation
    rotated = np.roll(pano, shift_px, axis=1)

    return rotated

# panorama to perspective
def pano_to_perspective(pano, fov, out_w, out_h, yaw=0, pitch=0):
    H, W = pano.shape[:2]

    # Convert FOV to radians
    fov = np.deg2rad(fov)
    fx = out_w / (2 * np.tan(fov / 2))
    fy = fx

    # pixel grid
    x = np.linspace(-out_w/2, out_w/2, out_w)
    y = np.linspace(-out_h/2, out_h/2, out_h)
    xx, yy = np.meshgrid(x, y)

    # ray directions
    z = fx
    dirs = np.stack([xx, yy, np.full_like(xx, z)], axis=-1)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    # rotation: yaw (Y axis) then pitch (X axis)
    Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                   [ 0,           1, 0          ],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1,           0,            0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch),  np.cos(pitch)]])
    R = Ry @ Rx
    dirs = dirs @ R.T

    # spherical coordinates
    theta = np.arctan2(dirs[..., 0], dirs[..., 2])
    phi   = np.arcsin(dirs[..., 1])

    # map to panorama coordinates
    u = ((theta / (2 * np.pi)) + 0.5) * W
    v = ((phi   / np.pi)       + 0.5) * H

    # sample
    return cv2.remap(
        pano,
        u.astype(np.float32),
        v.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP
    )

def perspective_to_pano(persp, fov, pano_w, pano_h, yaw=0, pitch=0):
    out_h, out_w = persp.shape[:2]

    # --- Projection parameters ---
    fov = np.deg2rad(fov)
    fx = out_w / (2 * np.tan(fov / 2))
    fy = fx

    # Pixel grid
    x = np.linspace(-out_w/2, out_w/2, out_w)
    y = np.linspace(-out_h/2, out_h/2, out_h)
    xx, yy = np.meshgrid(x, y)

    # Rays
    z = fx
    dirs = np.stack([xx, yy, np.full_like(xx, z)], axis=-1)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    # Rotation
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])
    R = Ry @ Rx
    dirs = dirs @ R.T

    # Spherical mapping
    theta = np.arctan2(dirs[..., 0], dirs[..., 2])
    phi   = np.arcsin(dirs[..., 1])

    u = ((theta / (2 * np.pi)) + 0.5) * pano_w
    v = ((phi   / np.pi)       + 0.5) * pano_h

    # --- Allocate pano patch ---
    pano_patch = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
    mask = np.zeros((pano_h, pano_w), dtype=np.uint8)

    # --- Forward splatting (write pixels) ---
    for i in range(out_h):
        for j in range(out_w):
            ui = int(u[i, j]) % pano_w
            vi = int(v[i, j])
            if 0 <= vi < pano_h:
                pano_patch[vi, ui] = persp[i, j]
                mask[vi, ui] = 1

    # Optional: count for debugging
    valid_pixels = mask.sum()
    print(f"Valid pixels in pano_patch & mask: {valid_pixels}")

    return pano_patch, mask

# -------------------------
# Helpers: rotation matrices
# -------------------------
def rotation_matrix(yaw, pitch):
    # yaw, pitch in radians
    Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                   [ 0,           1, 0          ],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1,           0,            0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch),  np.cos(pitch)]])
    return Ry @ Rx

# -------------------------
# 1) Inverse mapping (perfect, hole-free)
#    For every panorama pixel, compute if it lies inside the camera frustum,
#    then sample the perspective image using a backward mapping (cv2.remap).
# -------------------------
def inverse_project_to_pano(persp, fov_deg, pano_w, pano_h, yaw=0.0, pitch=0.0):
    """
    Return: pano_patch (H x W x 3), mask (H x W) where mask==1 where filled.
    Perfect mapping: for each pano pixel compute corresponding perspective coords.
    """
    out_h, out_w = persp.shape[:2]
    fov = np.deg2rad(fov_deg)
    fx = out_w / (2 * np.tan(fov / 2))
    fy = fx

    # build world directions for each pano pixel
    u = (np.arange(pano_w) + 0.5) / pano_w  # [0,1)
    v = (np.arange(pano_h) + 0.5) / pano_h
    uu, vv = np.meshgrid(u, v)

    theta = (uu - 0.5) * 2.0 * np.pi   # -pi..pi
    phi   = (vv - 0.5) * np.pi         # -pi/2..pi/2

    cos_phi = np.cos(phi)
    world_x = np.sin(theta) * cos_phi
    world_y = np.sin(phi)
    world_z = np.cos(theta) * cos_phi

    world_dirs = np.stack([world_x, world_y, world_z], axis=-1)  # (H, W, 3)

    # rotate world directions back into camera coords: camera_dir = R^T @ world_dir
    R = rotation_matrix(yaw, pitch)
    R_T = R.T
    # apply R_T to all world_dirs -> camera_dirs
    camera_dirs = world_dirs @ R_T.T  # shape (H, W, 3)  (using broadcasting)

    # points that are in front of camera have z > 0 (camera looks along +z in camera coords)
    cz = camera_dirs[..., 2]
    valid = cz > 1e-6

    # prepare image coords for valid pixels
    x_cam = camera_dirs[..., 0]
    y_cam = camera_dirs[..., 1]

    x_img = (x_cam / cz) * fx
    y_img = (y_cam / cz) * fy

    u_img = x_img + (out_w / 2.0)
    v_img = y_img + (out_h / 2.0)

    # initialize outputs
    pano_patch = np.zeros((pano_h, pano_w, 3), dtype=persp.dtype)
    mask = np.zeros((pano_h, pano_w), dtype=np.uint8)

    # For remap we need float32 maps of size pano_h x pano_w
    map_x = u_img.astype(np.float32)
    map_y = v_img.astype(np.float32)

    # For pixels that are outside perspective image bounds, set map to -1 so remap will return 0
    # but we'll use the 'valid' mask to ignore them.
    # Use cv2.remap to sample persp at these coordinates
    sampled = cv2.remap(
        persp,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,0,0)
    )

    # mask where sampling was from in-front and inside image bounds
    inside = (map_x >= 0) & (map_x < out_w) & (map_y >= 0) & (map_y < out_h)
    filled = valid & inside

    pano_patch[filled] = sampled[filled]
    mask[filled] = 1

    return pano_patch, mask
