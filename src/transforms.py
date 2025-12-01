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
