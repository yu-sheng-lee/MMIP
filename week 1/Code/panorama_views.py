"""Extract perspective BGR views from a full 360 x 180 equirectangular image.

Angles are degrees: yaw=0 looks at the panorama center, positive yaw looks
right; positive pitch looks up. No output files are written.
"""

from pathlib import Path

import cv2
import numpy as np


def read_panorama(path):
    """Read a panorama as uint8 BGR; support Unicode Windows paths."""
    data = np.fromfile(str(Path(path)), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode panorama: {path}")
    return image


def extract_view(panorama, yaw=0, pitch=0, hfov=70, size=(960, 640), roll=0):
    """Return a perspective view of a full equirectangular BGR panorama.

    size is (width, height). hfov is the horizontal field of view (0 < hfov
    < 180). Vertical FOV follows from aspect ratio and square pixels.
    roll tilts the camera about its viewing axis: positive is clockwise from
    the photographer's viewpoint, making the scene rotate counterclockwise.
    The source must cover the entire sphere, with north at its top and the
    horizon at its vertical center. A 2:1 shape alone cannot prove projection.
    """
    if (not isinstance(panorama, np.ndarray) or panorama.dtype != np.uint8
            or panorama.ndim != 3 or panorama.shape[2] != 3
            or min(panorama.shape[:2]) < 2):
        raise ValueError("panorama must be a non-empty uint8 BGR image")
    source_h, source_w = panorama.shape[:2]
    if abs(source_w / source_h - 2) > 0.02:
        raise ValueError("Expected a full 360 x 180 equirectangular panorama with a 2:1 aspect ratio")
    if (not all(np.isfinite(v) for v in (yaw, pitch, hfov, roll))
            or not -90 <= pitch <= 90 or not 0 < hfov < 180):
        raise ValueError("Angles must be finite; pitch in [-90,90], hfov in (0,180)")
    if (len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                              or v <= 0 for v in size)):
        raise ValueError("size must be (positive integer width, positive integer height)")
    width, height = size
    # OpenCV remap's internal coordinate limits apply to source and output.
    if max(source_w + 2, source_h, width, height) >= 32767:
        raise ValueError("OpenCV remap requires source/output dimensions below 32767")

    # Rays through output pixel centers, in a square-pixel pinhole camera.
    focal = width / (2 * np.tan(np.deg2rad(hfov) / 2))
    x, y = np.meshgrid((np.arange(width) + 0.5 - width / 2) / focal,
                       (height / 2 - np.arange(height) - 0.5) / focal)
    yaw_r, pitch_r = np.deg2rad([yaw % 360, pitch])
    # Rotate rays around the local viewing axis before pitch and yaw.
    roll_r = np.deg2rad(roll % 360)
    x, y = (x * np.cos(roll_r) + y * np.sin(roll_r),
            -x * np.sin(roll_r) + y * np.cos(roll_r))
    up = y * np.cos(pitch_r) + np.sin(pitch_r)
    forward = np.cos(pitch_r) - y * np.sin(pitch_r)
    world_x = x * np.cos(yaw_r) + forward * np.sin(yaw_r)
    world_z = forward * np.cos(yaw_r) - x * np.sin(yaw_r)
    longitude = np.arctan2(world_x, world_z)
    latitude = np.arctan2(up, np.hypot(world_x, world_z))

    # Equirectangular pixel centers: wrap longitude, clamp latitude at poles.
    map_x = ((longitude / (2 * np.pi) + 0.5) * source_w - 0.5) % source_w
    map_y = np.clip((0.5 - latitude / np.pi) * source_h - 0.5, 0, source_h - 1)
    # Add seam neighbors horizontally only; never wrap north to south.
    padded = np.concatenate((panorama[:, -1:], panorama, panorama[:, :1]), axis=1)
    return cv2.remap(padded, (map_x + 1).astype(np.float32),
                     map_y.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def extract_views(panorama, yaws, pitch=0, hfov=70, size=(960, 640), roll=0):
    """Return BGR views; roll may be one angle or one angle per yaw."""
    yaws = list(yaws)
    rolls = [roll] * len(yaws) if np.isscalar(roll) else list(roll)
    if len(rolls) != len(yaws):
        raise ValueError("roll must be a single angle or one angle per yaw")
    return [extract_view(panorama, yaw, pitch, hfov, size, tilt)
            for yaw, tilt in zip(yaws, rolls)]


def show_views(views, titles=None):
    """Display extracted BGR images without saving any files."""
    import matplotlib.pyplot as plt

    views = list(views)
    if not views:
        raise ValueError("views cannot be empty")
    if titles is not None and len(titles) != len(views):
        raise ValueError("titles must have one entry per view")
    columns = min(3, len(views))
    rows = (len(views) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows),
                             squeeze=False, layout='constrained')
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i < len(views):
            ax.imshow(cv2.cvtColor(views[i], cv2.COLOR_BGR2RGB))
            ax.set_title(titles[i] if titles is not None else f'View {i + 1}')
    plt.show()
    plt.close(fig)
