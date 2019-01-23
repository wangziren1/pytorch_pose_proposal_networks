import numpy as np
import random
import cv2

from src.utils import colors
import src.config as cfg


def mpi2ours(meta):
    # Add person instance which is the mid-point between head top and upper neck
    # Remove pelvis and thorax, and add center which is the mid-point between them
    MPI_to_ours_1 = np.array([9, 9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5, 6])
    MPI_to_ours_2 = np.array([8, 9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5, 7])
    joint = (meta['joint_self'][MPI_to_ours_1][:, :2] +
             meta['joint_self'][MPI_to_ours_2][:, :2]) / 2
    # Note that we use the visible of head top and pelvis as the visible of person
    # instance and center, respectively.
    visible = meta['joint_self'][MPI_to_ours_1][:, -1]
    visible = visible[:, np.newaxis]
    meta['joint_self'] = np.concatenate((joint, visible), axis=-1)
    if meta['numOtherPeople'] != 0.0:
        joint = (meta['joint_others'][:, MPI_to_ours_1, :2] +
                 meta['joint_others'][:, MPI_to_ours_2, :2]) / 2
        visible = meta['joint_others'][:, MPI_to_ours_1, -1]
        visible = visible[:, :, np.newaxis]
        meta['joint_others'] = np.concatenate((joint, visible), axis=-1)
    return meta


def visualize(img, meta):
    img_copy = img.copy()
    radius = int(5 * img.shape[0] / 700)
    # draw self
    cv2.circle(img_copy, (int(meta['objpos'][0]), int(meta['objpos'][1])),
               2 * radius, (255, 255, 0), -1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(16):
        if meta['joint_self'][i, -1] == 2.0:
            continue
        x, y = int(meta['joint_self'][i, 0]), int(meta['joint_self'][i, 1])
        cv2.circle(img_copy, (x, y), radius, colors[0], -1, cv2.LINE_AA)
        # cv2.putText(img_copy, str(i), (x, y), font, .5, color[0], 1, cv2.LINE_AA)

    # draw others
    for i in range(int(meta['numOtherPeople'])):
        c = colors[i]
        for j in range(16):
            if meta['joint_others'][i, j, -1] == 2.0:
                continue
            x, y = int(meta['joint_others'][i, j, 0]), int(meta['joint_others'][i, j, 1])
            cv2.circle(img_copy, (x, y), radius, c, -1, cv2.LINE_AA)
            # cv2.putText(img_copy, str(j), (x, y), font, .5, c, 1, cv2.LINE_AA)

    # draw grid
    interval_x, interval_y = 32, 32
    grid_color = np.array([105, 105, 105], dtype=np.uint8)
    img_copy[:, ::interval_y, :] = grid_color
    img_copy[::interval_x, :, :] = grid_color

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 800, 800)
    cv2.imshow('image', img_copy[:, :, ::-1])
    cv2.waitKey(0)


def scale(img_src, meta, is_train):
    if is_train:
        if random.random() > cfg.SCALE_PROB:
            scale_multiplier = 1
        else:
            scale_multiplier = (cfg.SCALE_MAX - cfg.SCALE_MIN) * random.random() + cfg.SCALE_MIN
    else:
        scale_multiplier = 1

    scale_abs = cfg.TARGET_DIST / meta['scale_provided']
    scale = scale_abs * scale_multiplier
    img_dst = cv2.resize(img_src, None, fx=scale, fy=scale)

    meta['objpos'] *= scale
    meta['joint_self'][:, :2] *= scale

    # scale objpos and joint
    if meta['numOtherPeople'] != 0.0:
        meta['objpos_other'] *= scale
        meta['joint_others'][:, :, :2] *= scale
    return img_dst


def rotate_point(array, mat_rotate):
    if array.ndim == 1:
        array_extend = np.ones(3)
        array_extend[:2] = array[:]
        array_transform = mat_rotate.dot(array_extend[:, np.newaxis])
        return array_transform[:, 0]
    elif array.ndim == 2:
        array_extend = np.ones((array.shape[0], 3))
        array_extend[:, :2] = array[:, :2]
        array_transform = mat_rotate.dot(array_extend.transpose())
        return array_transform.transpose()
    elif array.ndim == 3:
        array_extend = np.ones((array.shape[0], array.shape[1], 3))
        array_extend[:, :, :2] = array[:, :, :2]
        array_transform = mat_rotate.dot(array_extend.transpose((0, 2, 1)))
        return array_transform.transpose(1, 2, 0)


def rotate(img_src, meta):
    degree = (random.random() - 0.5) * 2 * cfg.MAX_ROTATE_DEGREE

    h, w = img_src.shape[:2]
    center = (w / 2, h / 2)
    mat_rotate = cv2.getRotationMatrix2D(center, degree, 1)

    cos = np.abs(mat_rotate[0, 0])
    sin = np.abs(mat_rotate[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    mat_rotate[0, 2] += (new_w / 2) - center[0]
    mat_rotate[1, 2] += (new_h / 2) - center[1]

    img_dst = cv2.warpAffine(img_src, mat_rotate, (new_w, new_h))

    # rotate objpos and joint
    meta['objpos'] = rotate_point(meta['objpos'], mat_rotate)
    meta['joint_self'][:, :2] = rotate_point(meta['joint_self'], mat_rotate)

    if meta['numOtherPeople'] != 0.0:
        meta['objpos_other'] = rotate_point(meta['objpos_other'], mat_rotate)
        meta['joint_others'][:, :, :2] = rotate_point(meta['joint_others'], mat_rotate)

    return img_dst


def croppad(img_src, meta, is_train):
    crop_size = np.array(cfg.IMG_SIZE)
    img_dst = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)

    if is_train:
        offset = (np.random.rand(2) - 0.5) * 2 * cfg.CENTER_PERTURB_MAX
    else:
        offset = 0
    center = meta['objpos'] + offset

    # the location of top-left corner of destination image on the source image 
    tl = center - crop_size / 2
    x, y = int(tl[0]), int(tl[1])

    gridx, gridy = np.meshgrid(np.arange(crop_size), np.arange(crop_size))
    gridx += x
    gridy += y

    logit_x = np.logical_and((gridx >= 0), (gridx < img_src.shape[1]))
    logit_y = np.logical_and((gridy >= 0), (gridy < img_src.shape[0]))
    logit = np.logical_and(logit_x, logit_y)

    img_dst[logit] = img_src[gridy[logit], gridx[logit], :]

    meta['objpos'] -= tl
    meta['joint_self'][:, :2] -= tl
    if meta['numOtherPeople'] != 0.0:
        meta['objpos_other'] -= tl
        meta['joint_others'][:, :, :2] -= tl
    return img_dst


def label_crop_joint(meta):
    is_crop_x = np.logical_or(meta['joint_self'][:, 0] < 0,
                              meta['joint_self'][:, 0] >= cfg.IMG_SIZE)
    is_crop_y = np.logical_or(meta['joint_self'][:, 1] < 0,
                              meta['joint_self'][:, 1] >= cfg.IMG_SIZE)
    is_crop = np.logical_or(is_crop_x, is_crop_y)
    if is_crop.any():
        meta['joint_self'][is_crop, -1] = 2.0
    if meta['numOtherPeople'] != 0.0:
        is_crop_x = np.logical_or(meta['joint_others'][:, :, 0] < 0,
                                  meta['joint_others'][:, :, 0] >= cfg.IMG_SIZE)
        is_crop_y = np.logical_or(meta['joint_others'][:, :, 1] < 0,
                                  meta['joint_others'][:, :, 1] >= cfg.IMG_SIZE)
        is_crop = np.logical_or(is_crop_x, is_crop_y)
        if is_crop.any():
            meta['joint_others'][is_crop, -1] = 2.0


def flip(img_src, meta):
    if random.random() <= cfg.FLIP_PROB:
        img_dst = img_src[:, ::-1, :]
        w = img_src.shape[1] - 1
        meta['objpos'][0] = w - meta['objpos'][0]
        meta['joint_self'][:, 0] = w - meta['joint_self'][:, 0]
        # 0 - person instance
        # 1 - head top, 2 - upper neck,
        # 3 - r shoulder, 4 - r elbow, 5 - r wrist, 6 - l shoulder, 7 - l elbow, 8 - l wrist,
        # 9 - r hip, 10 - r knee, 11 - r ankle, 12 - l hip, 13 - l knee, 14 - l ankle
        # 15 - center
        right = np.array([3, 4, 5, 9, 10, 11])
        left = np.array([6, 7, 8, 12, 13, 14])
        # swap left right
        meta['joint_self'][right], meta['joint_self'][left] = \
            meta['joint_self'][left], meta['joint_self'][right]

        if meta['numOtherPeople'] != 0.0:
            meta['objpos_other'][:, 0] = w - meta['objpos_other'][:, 0]
            meta['joint_others'][:, :, 0] = w - meta['joint_others'][:, :, 0]
            # swap left right
            meta['joint_others'][:, right], meta['joint_others'][:, left] = \
                meta['joint_others'][:, left], meta['joint_others'][:, right]

        return img_dst
    else:
        return img_src
