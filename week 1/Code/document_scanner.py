"""Quiz 3: detect a flat document and correct its perspective.

Usage: scanned = scan_document(bgr_image, show_steps=True)
"""

import cv2
import numpy as np


def order_corners(points):
    """Return distinct corners in TL, TR, BR, BL order."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    return np.roll(points, -np.argmin(points.sum(axis=1)), axis=0)


def fit_virtual_corners(contour, seed):
    """Extend the straight middle sections of four sides past rounded corners."""
    corners = order_corners(seed)
    # Densify the contour so fit quality does not depend on CHAIN_APPROX_SIMPLE.
    vertices = contour.reshape(-1, 2).astype(np.float32)
    samples = []
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        count = max(1, int(np.linalg.norm(end - start) / 2))
        samples.append(start + np.linspace(0, 1, count, endpoint=False)[:, None]
                       * (end - start))
    points = np.concatenate(samples).astype(np.float32)
    lines = []
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 20:
            return None
        unit = direction / length
        delta = points - start
        along = delta @ unit
        distance = np.abs(delta[:, 0] * unit[1] - delta[:, 1] * unit[0])
        selected = points[(along > 0.2 * length) & (along < 0.8 * length)
                          & (distance < 0.10 * length)]
        if len(selected) < 10 or np.ptp((selected - start) @ unit) < 0.45 * length:
            return None
        vx, vy, x, y = cv2.fitLine(selected, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        vector = np.array([vx, vy])
        origin = np.array([x, y])
        residual = np.abs((selected[:, 0] - x) * vy - (selected[:, 1] - y) * vx)
        if np.quantile(residual, 0.9) > max(1.5, length * 0.008):
            return None
        lines.append((origin, vector))
    intersections = []
    for i in range(4):
        p, u = lines[i - 1]
        q, v = lines[i]
        matrix = np.column_stack((u, -v))
        if abs(np.linalg.det(matrix)) < 0.15:
            return None
        t = np.linalg.solve(matrix, q - p)[0]
        intersections.append(p + t * u)
    result = np.float32(intersections)
    if not cv2.isContourConvex(result.reshape(-1, 1, 2)):
        return None
    if np.max(np.linalg.norm(result - corners, axis=1)) > 0.2 * np.sqrt(cv2.contourArea(corners)):
        return None
    return result


def document_candidates(contour, allow_curved=True):
    """Fit quadrilaterals while rejecting poorly matched rounded objects."""
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0 or cv2.contourArea(contour) / hull_area < 0.85:
        return
    curves = [contour, hull] if allow_curved else [contour]
    tolerances = [0.01, 0.015, 0.025, 0.04, 0.055, 0.07] if allow_curved else [0.015, 0.025, 0.04]
    candidates = []
    for curve in curves:
        perimeter = cv2.arcLength(curve, True)
        for epsilon in tolerances:
            polygon = cv2.approxPolyDP(curve, epsilon * perimeter, True)
            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                candidates.append((polygon, 0.88))
    if allow_curved:
        # Only use a rotated rectangle when it closely fits the hull.
        # It cannot by itself recover strong perspective distortion.
        candidates.append((cv2.boxPoints(cv2.minAreaRect(hull)), 0.93))
        for seed, _ in list(candidates):
            fitted = fit_virtual_corners(contour, seed)
            if fitted is not None:
                candidates.append((fitted, 0.88))
    for polygon, minimum_overlap in candidates:
        corners = order_corners(polygon)
        area = cv2.contourArea(corners)
        intersection, _ = cv2.intersectConvexConvex(
            hull.astype(np.float32), corners.reshape(-1, 1, 2))
        union = hull_area + area - intersection
        overlap = intersection / union if union > 0 else 0
        if overlap >= minimum_overlap:
            yield corners, overlap


def detect_document(image, max_side=1200, min_area_ratio=0.02, allow_curved=True,
                    diagnostics=None):
    """Return (corners in original coordinates or None, edge image).

    This baseline favors large convex quadrilaterals. It can mistake another
    rectangular object for paper, especially with a cluttered background.
    """
    if not 0 < min_area_ratio < 0.98:
        raise ValueError('min_area_ratio must be between 0 and 0.98')
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    small = cv2.resize(image, (max(1, round(width * scale)),
                               max(1, round(height * scale))))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if diagnostics is not None:
        diagnostics['steps'] = [('Original', image), ('Resized', small),
                                ('Gray', gray), ('GaussianBlur (5 x 5)', blurred)]
        diagnostics['selected_threshold'] = None
    best = None
    best_score = -1.0
    best_edges = None
    # Try several thresholds automatically when document contrast varies.
    for low, high in [(50, 150), (25, 75), (75, 200)]:
        canny = cv2.Canny(blurred, low, high)
        edges = cv2.morphologyEx(canny, cv2.MORPH_CLOSE,
                                 np.ones((3, 3), np.uint8))
        if best_edges is None:
            best_edges = edges
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
        candidates_view = small.copy() if diagnostics is not None else None
        if candidates_view is not None:
            cv2.drawContours(candidates_view, contours, -1, (255, 160, 0), 1)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
            for corners, overlap in document_candidates(contour, allow_curved):
                area = cv2.contourArea(corners)
                ratio = area / (small.shape[0] * small.shape[1])
                if not min_area_ratio <= ratio <= 0.98:
                    continue
                if (np.any(corners < 0) or
                        np.any(corners[:, 0] > small.shape[1] - 1) or
                        np.any(corners[:, 1] > small.shape[0] - 1)):
                    continue
                if np.min(np.linalg.norm(corners - np.roll(corners, 1, axis=0),
                                         axis=1)) < 15:
                    continue
                score = ratio * overlap
                if candidates_view is not None:
                    cv2.polylines(candidates_view, [np.rint(corners).astype(np.int32)],
                                  True, (0, 255, 0), 2)
                if score > best_score:
                    best_score = score
                    best = corners
                    best_edges = edges
                    if diagnostics is not None:
                        diagnostics['selected_threshold'] = (low, high)
        if diagnostics is not None:
            diagnostics['steps'].extend([
                (f'Canny ({low}, {high})', canny),
                (f'Closing ({low}, {high})', edges),
                (f'Contours + candidates ({low}, {high})', candidates_view)])
    if best is not None:
        best *= np.array([width / small.shape[1], height / small.shape[0]],
                         dtype=np.float32)
    return best, best_edges


def warp_document(image, points):
    """Rectify a flat document; optionally impose the A4 aspect ratio."""
    tl, tr, br, bl = corners = order_corners(points)
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    width, height = max(2, round(width)), max(2, round(height))
    destination = np.float32([[0, 0], [width - 1, 0],
                              [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def show_pipeline(diagnostics, overlay, scanned):
    """Display the selected threshold; use a labeled fallback on failure."""
    import matplotlib.pyplot as plt

    selected = diagnostics['selected_threshold']
    displayed = selected if selected is not None else (50, 150)
    suffix = f'({displayed[0]}, {displayed[1]})'
    steps = [step for step in diagnostics['steps']
             if not step[0].startswith(('Canny ', 'Closing ', 'Contours + candidates '))
             or step[0].endswith(suffix)]
    steps += [('Detected document', overlay),
                                    ('Perspective corrected', scanned)]
    rows = (len(steps) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 4),
                             layout='constrained')
    for ax, (title, value) in zip(axes.flat, steps):
        if value is None:
            ax.text(0.5, 0.5, 'No document detected', ha='center', va='center',
                    transform=ax.transAxes)
        elif value.ndim == 2:
            ax.imshow(value, cmap='gray', vmin=0, vmax=255)
        else:
            ax.imshow(cv2.cvtColor(value, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    for ax in list(axes.flat)[len(steps):]:
        ax.axis('off')
    status = (f'selected Canny: {selected}' if selected is not None else
              f'No document detected; preview Canny: {displayed}')
    fig.suptitle(f'Document scanning | {status}\n'
                 'Contours: blue/orange; accepted candidates: green', fontsize=14)
    plt.show()
    plt.close(fig)


def scan_document(image, allow_curved=True,
                  min_area_ratio=0.02, show_steps=False):
    """Accept a uint8 BGR image, display steps, and return BGR output or None.

    No files are read or written. The input image is not modified.
    """
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
        raise ValueError("image must be a non-empty uint8 BGR array, e.g. cv2.imread output")
    diagnostics = {} if show_steps else None
    corners, _ = detect_document(image, allow_curved=allow_curved,
                                 min_area_ratio=min_area_ratio,
                                 diagnostics=diagnostics)
    if corners is None:
        if show_steps:
            show_pipeline(diagnostics, image, None)
        print("No document detected.")
        return None
    scanned = warp_document(image, corners)
    if show_steps:
        overlay = image.copy()
        cv2.polylines(overlay, [np.rint(corners).astype(np.int32)], True,
                      (0, 255, 0), max(2, image.shape[1] // 400))
        for index, point in enumerate(corners):
            xy = tuple(np.rint(point).astype(int))
            cv2.circle(overlay, xy, 8, (0, 0, 255), -1)
            cv2.putText(overlay, str(index + 1), xy, cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
        show_pipeline(diagnostics, overlay, scanned)
    return scanned


if __name__ == '__main__':
    print("Import scan_document and call scan_document(image, show_steps=True) "
          "with a cv2 BGR image. No files are saved.")
