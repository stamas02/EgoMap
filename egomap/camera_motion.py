import cv2
import numpy as np
import scipy.optimize as optimize


def get_feature_displacements(prev_img, next_img):
    """
    Given two images it detects 'goodFeaturesToTrack' in the first image which are then tracked to the second image.
    If the same feature can be found in the second image a displacement vector is calculated.
    Parameters
    ----------
    img1: ndarray:
        first image
    img2: nd array:
        second image

    Returns
    -------
    an nd array of displacement vectors
    """

    p1 = cv2.goodFeaturesToTrack(prev_img,
                                 mask=None,
                                 maxCorners=100,
                                 qualityLevel=0.3,
                                 minDistance=7,
                                 blockSize=7)

    if p1 is None:
        return np.array([[0, 0]])

    p2, st, err = cv2.calcOpticalFlowPyrLK(prev_img,
                                           next_img,
                                           p1,
                                           None,
                                           winSize=(15, 15),
                                           maxLevel=2,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    good_p_1 = p1[st == 1].reshape(-1, 2)
    good_p_2 = p2[st == 1].reshape(-1, 2)
    return good_p_2 - good_p_1


def get_grid_centres(width, height, grid_width, grid_height):
    """
    Returns the center coordinate of each grid cell on a rectangular canvas.

    Parameters
    ----------
    width: int,
        width of the canvas the grid is to be defined on.
    height: int
        height of the canvas the grid is to be defined on.
    grid_width:
        Grid width. E.g. whn 7 the canvas will be decided horizontally into 7 equal areas. The center of these is then
        returned.
    grid_height
        Grid width. E.g. whn 7 the canvas will be decided vertically into 7 equal areas. The center of these is then
        returned.
    Returns
    -------

    """
    cell_w = width // grid_width
    cell_h = height // grid_height
    x_centres = np.array(range(0, grid_width)) * cell_w + (cell_w // 2)
    x_centres = np.repeat([x_centres], grid_height, 0)
    y_centres = np.array(range(0, grid_height)) * cell_h + (cell_h // 2)
    y_centres = np.reshape(np.repeat(y_centres, grid_width, -1), x_centres.shape)
    coords = np.stack((x_centres, y_centres), axis=2)
    return coords


def get_grid(image, grid_width, grid_height):
    """
    Given an image, this function returns the grid blocks.

    Parameters
    ----------
    image : numpy array
        RGB or grayscale image.

    grid_height : int
        Height of the grid used to estimate optical flow. For example, 7 means that the image will be split into
        7 equal parts vertically.

    grid_width : int
        Width of the grid used to estimate optical flow. For example, 7 means that the image will be split into
        7 equal parts horizontally.

    Returns
    -------
    list of numpy arrays
        List of blocks as numpy arrays.

    Raises
    ------
    ValueError
        If the image cannot be divided into the given grid accurately.
    """
    h, w = image.shape
    n_rows = grid_height
    n_cols = grid_width
    assert h % n_rows == 0, "{} rows is not evenly divisble by {}".format(h, n_rows)
    assert w % n_cols == 0, "{} cols is not evenly divisble by {}".format(w, n_cols)

    blocks = []
    for y in range(0, h, h // n_rows):
        for x in range(0, w, w // n_cols):
            blocks.append(image[y:y + h // n_rows, x:x + h // n_cols])

    return np.array(blocks)


def get_optical_flow(prev_img, next_img, grid_width=10, grid_height=5):
    """
    Estimates the optical flow between two images. The images are split into an grid_width*grid_height grid.
    for each cell in the grid features are detected and tracked and displacement vectors are calculated. The mean
    of the displacement vector represents the optical flow inside the grid cell.

    Parameters
    ----------
    prev_img: ndarray:
        previous image
    next_img: ndarray:
        nect image
    grid_width: int,
        width of the grid used to estimate optical flow. E.g. 7 means that the image will be split into
        7 equal parts vertically.
    grid_height
        height of the grid used to estimate optical flow. E.g. 7 means that the image will be split into
        7 equal parts horizontally.
    Returns
    -------
    grid_width*grid_height*2 nd array representing optical flow
    """
    prev_img = np.array(prev_img)
    next_img = np.array(next_img)
    if len(prev_img.shape) != 3 and len(prev_img.shape) != 2:
        raise ValueError("Image must be grayscale or RGB got {0}".format(prev_img.shape))
    if len(next_img.shape) != 3 and len(next_img.shape) != 2:
        raise ValueError("Image must be grayscale or RGB got {0}".format(next_img.shape))

    if len(prev_img.shape) == 3:
        prev_img = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
    if len(next_img.shape) == 3:
        next_img = cv2.cvtColor(next_img, cv2.COLOR_RGB2GRAY)

    prev_blocks = get_grid(prev_img, grid_width, grid_height)
    next_blocks = get_grid(next_img, grid_width, grid_height)
    optical_flow = np.zeros((grid_width * grid_height, 2))
    for i, blocks in enumerate(zip(prev_blocks, next_blocks)):
        displacements = get_feature_displacements(blocks[0], blocks[1])
        optical_flow[i] = np.mean(displacements, 0)

    optical_flow = optical_flow.reshape((grid_height, grid_width, 2))
    optical_flow = np.nan_to_num(np.array(optical_flow))
    h, w = next_img.shape
    grid_centres = get_grid_centres(w, h, grid_width, grid_height)
    optical_flow += grid_centres
    return np.array([grid_centres, optical_flow])


def _z_translation_fn(data, parameter, focal_length):
    x = data[:, :, 0]
    y = data[:, :, 1]
    f = focal_length
    new_data = np.array(data)
    new_data[:, :, 0] = f * (np.arctan(x / f)) * (1 + (x ** 2 / f ** 2)) * parameter
    new_data[:, :, 1] = f * (np.arctan(y / f)) * (1 + (y ** 2 / f ** 2)) * parameter

    return new_data.flatten()


def estimate_z_transition(optical_flow):
    """
    Estimates amount of camera transition on the z axis based on optical flow.

    Returns
    -------
        - The estimated amount of transition on the z axis of the camera.
        - The perfect optical flow based on the estimation
    """

    focal_length = 150
    func = lambda data, parameter: _z_translation_fn(data, parameter, focal_length)

    cx = (np.max(optical_flow[0, :, :, 0]) + np.min(optical_flow[0, :, :, 0])) // 2
    cy = (np.max(optical_flow[0, :, :, 1]) + np.min(optical_flow[0, :, :, 1])) // 2
    optical_flow = np.array(optical_flow)
    optical_flow[:, :, :, 0] -= cx
    optical_flow[:, :, :, 1] -= cy
    optical_flow[:, :, :, 0] /= cx*2
    optical_flow[:, :, :, 1] /= cy*2

    param, pcov = optimize.curve_fit(func, optical_flow[0], optical_flow[1].flatten())
    estimated_optical_flow = np.reshape(_z_translation_fn(optical_flow[0], param, focal_length), optical_flow[0].shape)

    estimated_optical_flow = np.array([optical_flow[0], estimated_optical_flow])
    estimated_optical_flow[:, :, :, 0] *= cx*2
    estimated_optical_flow[:, :, :, 1] *= cy*2
    estimated_optical_flow[:, :, :, 0] += cx
    estimated_optical_flow[:, :, :, 1] += cy

    return param, estimated_optical_flow