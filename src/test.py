import cv2
import torch.utils.data

from src.utils import *
import src.config as cfg
from src.dataset.mpi import MPI
from src.model.model import which_model

valid_dset = MPI(cfg.IMG_DIR, cfg.ANNOTATION_PATH, is_train=False)
valid_loader = torch.utils.data.DataLoader(valid_dset, batch_size=1,
                                           shuffle=False, num_workers=cfg.NUM_WORKS)

checkpoint = torch.load(cfg.CHECKPOINT_PATH)
net = which_model(is_shallow=cfg.IS_SHALLOW, net_state_dict=checkpoint['net_state_dict'])
net.to(device)


def test():
    net.eval()
    for l, (input, target, mask) in enumerate(valid_loader):
        input = input.to(device)
        target, mask = target.to(device), mask.to(device)
        with torch.no_grad():
            output = net(input)

        # Transform relative coordinates to absolute coordinates
        output[:, :, :, 2:96:6] = (output[:, :, :, 2:96:6] + x_offset) * cfg.CELL_SIZE
        output[:, :, :, 3:96:6] = (output[:, :, :, 3:96:6] + y_offset) * cfg.CELL_SIZE
        output[:, :, :, 4:96:6] = output[:, :, :, 4:96:6].pow(2) * cfg.IMG_SIZE
        output[:, :, :, 5:96:6] = output[:, :, :, 5:96:6].pow(2) * cfg.IMG_SIZE

        input = input.cpu().numpy().transpose(0, 2, 3, 1)
        input = input * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        input = np.clip(input, 0, 1)
        output = output.cpu().numpy()
        output_limb = np.reshape(output[0, :, :, 96:], (12, 12, 9, 9, 15))
        target_limb = target.cpu().numpy()[0, :, :, 96:].reshape((12, 12, 9, 9, 15))
        mask_limb = mask.cpu().numpy()[0, :, :, 96:].reshape((12, 12, 9, 9, 15))

        # Extract candidate joints from output tensor and none maximum suppression
        all_joints_before = [[] for _ in range(16)]
        all_joints = [[] for _ in range(16)]
        for m in range(16):
            is_exist = output[:, :, :, 6*m] >= cfg.thres1
            joints = output[:, :, :, 6*m:6*m+6][is_exist, :]
            if joints.size > 0:
                # The seventh element is the person id which is used to group joints.
                # -1 denotes that it does not belong to any person.
                all_joints_before[m].append(np.concatenate((joints, np.full((joints.shape[0], 1), -1)), axis=1))
                joints = nms(np.concatenate((joints, np.full((joints.shape[0], 1), -1)), axis=1))
                all_joints[m].append(joints)
        draw_box(input, all_joints_before)
        draw_box(input, all_joints)

        # Get connection between joints that constitute limbs
        connection = [[] for _ in range(15)]
        for limb_id in range(15):
            if not all_joints[limbs_start[limb_id]] or not all_joints[limbs_end[limb_id]]:
                continue
            l_start = all_joints[limbs_start[limb_id]][0]
            start_x = (l_start[:, 2] // cfg.CELL_SIZE).astype(int)
            start_y = (l_start[:, 3] // cfg.CELL_SIZE).astype(int)
            l_end = all_joints[limbs_end[limb_id]][0]
            end_x = (l_end[:, 2] // cfg.CELL_SIZE).astype(int)
            end_y = (l_end[:, 3] // cfg.CELL_SIZE).astype(int)

            edges = np.zeros((len(l_end), len(l_start)))
            for i in range(len(l_end)):
                for j in range(len(l_start)):
                    s_y, s_x = start_y[j], start_x[j]
                    if 12 > s_y >= 0 and 12 > s_x >= 0:
                        e_y, e_x = 4 + end_y[i] - s_y, 4 + end_x[i] - s_x
                        if 9 > e_y >= 0 and 9 > e_x >= 0:
                            limb_score = output_limb[s_y, s_x, e_y, e_x, limb_id]
                            if limb_score > 0.1:
                                edges[i, j] = l_end[i, 0] * limb_score * l_start[j, 0]

            n = min(len(l_end), len(l_start))
            for _ in range(n):
                max_score = np.max(edges)
                index_end, index_start = np.nonzero(edges == max_score)
                if max_score != 0:
                    connection[limb_id].append((index_start[0], index_end[0]))
                edges[index_end[0], :] = 0
                edges[:, index_start[0]] = 0

        # Group joints into theirs person
        num_person = len(connection[0])
        persons = []
        for p_id in range(num_person):
            new_person = np.full((16, 2), -1.0)
            person_instance_id = connection[0][p_id][1]
            new_person[0] = all_joints[0][0][person_instance_id][2:4]
            all_joints[0][0][person_instance_id][-1] = p_id
            persons.append(new_person)

        # In order to decide which person the right/left hip joint belong to,
        # we must first know which person the center joint belongs to.
        order = [0, 1, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        for limb_type in order:
            start_id, end_id = limbs_start[limb_type], limbs_end[limb_type]
            for i, j in connection[limb_type]:
                l_start = all_joints[start_id][0]
                l_end = all_joints[end_id][0]
                p = int(l_end[j, -1])
                if p == -1:
                    continue
                persons[p][start_id, :] = l_start[i, 2:4]
                l_start[i, -1] = l_end[j, -1]
        draw_limb(input, persons)


def nms(joints):
    iou = joints[:, 0]
    order = np.argsort(iou)[::-1]
    joints = joints[order, :]
    for j in range(len(joints)-1):
        if joints[j][1] == 0:
            continue
        tl_x = np.maximum(joints[j, 2] - 0.5 * joints[j, 4], joints[j+1:, 2] - 0.5 * joints[j+1:, 4])
        tl_y = np.maximum(joints[j, 3] - 0.5 * joints[j, 5], joints[j+1:, 3] - 0.5 * joints[j+1:, 5])
        br_x = np.minimum(joints[j, 2] + 0.5 * joints[j, 4], joints[j+1:, 2] + 0.5 * joints[j+1:, 4])
        br_y = np.minimum(joints[j, 3] + 0.5 * joints[j, 5], joints[j+1:, 3] + 0.5 * joints[j+1:, 5])
        delta_x, delta_y = br_x - tl_x, br_y - tl_y
        condition = np.logical_or(delta_x < 0, delta_y < 0)
        intersection = np.where(condition, 0, delta_x * delta_y)
        union = joints[j, 4] * joints[j, 5] + joints[j+1:, 4] * joints[j+1:, 5] - intersection
        joints[j+1:, 1][intersection / union >= cfg.thres2] = 0
    joints = joints[np.nonzero(joints[:, 1] > 0)]
    return joints


def draw_box(img, joints):
    overlay = img[0].copy()
    for i in range(1, 16):
        if joints[i]:
            for j in range(len(joints[i][0])):
                box = joints[i][0][j]
                tl_x, tl_y, br_x, br_y = int(box[2] - 0.5 * box[4]), int(box[3] - 0.5 * box[5]), \
                                         int(box[2] + 0.5 * box[4]), int(box[3] + 0.5 * box[5])
                cv2.rectangle(overlay, (tl_x, tl_y), (br_x, br_y), colors[i-1], -1)

    img_transparent = cv2.addWeighted(overlay, alpha, img[0], 1 - alpha, 0)[:, :, ::-1]
    img_transparent[:, ::cfg.CELL_SIZE, :] = np.array([1., 1, 1])
    img_transparent[::cfg.CELL_SIZE, :, :] = np.array([1., 1, 1])
    cv2.namedWindow('box', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('box', 600, 600)
    cv2.imshow('box', img_transparent)
    key = cv2.waitKey(0)
    if key == ord('s'):
        cv2.imwrite('box.png', img_transparent * 255)


def draw_limb(img, persons):
    overlay = img[0].copy()
    for p in persons:
        for j in range(1, 16):
            if p[j][0] == -1 or p[j][1] == -1:
                continue
            cv2.circle(overlay, (int(p[j][0]), int(p[j][1])), 3, colors[j-1],
                       -1, cv2.LINE_AA)

    for p in persons:
        for j in range(14):
            j1, j2 = p[limbs1[j]], p[limbs2[j]]
            if (j1 == -1).any() or (j2 == -1).any():
                continue
            cv2.line(overlay, (int(j1[0]), int(j1[1])), (int(j2[0]), int(j2[1])),
                     colors[j], 2, cv2.LINE_AA)
    img_dst = cv2.addWeighted(overlay, alpha, img[0], 1-alpha, 0)[:, :, ::-1]
    cv2.namedWindow('persons', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('persons', 600, 600)
    cv2.imshow('persons', img_dst)
    key = cv2.waitKey(0)
    if key == ord('s'):
        cv2.imwrite('persons.png', img_dst * 255)


if __name__ == '__main__':
    test()
