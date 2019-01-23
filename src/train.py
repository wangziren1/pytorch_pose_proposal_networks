import os
import cv2
from tensorboardX import SummaryWriter

import torch.optim as optim
import torch.utils.data

from src.utils import *
import src.config as cfg
from src.dataset.mpi import MPI
from src.model.model import which_model

train_dset = MPI(cfg.IMG_DIR, cfg.ANNOTATION_PATH, is_train=True)
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=cfg.BATCH_SIZE,
                                           shuffle=True, num_workers=cfg.NUM_WORKS)

writer = SummaryWriter(log_dir=cfg.SUMMARY_PATH, purge_step=0)

# Initialize network
if os.path.exists(cfg.CHECKPOINT_PATH):
    checkpoint = torch.load(cfg.CHECKPOINT_PATH)
    step = checkpoint['step']
    start = checkpoint['epoch'] + 1
    net = which_model(cfg.IS_SHALLOW, net_state_dict=checkpoint['net_state_dict'])
else:
    net = which_model(cfg.IS_SHALLOW)
    start = 0
    step = -1
net.to(device)
optimizer = optim.SGD(net.parameters(), lr=cfg.LR, momentum=cfg.MOMENTUM,
                      weight_decay=cfg.WEIGHT_DECAY)
if os.path.exists(cfg.CHECKPOINT_PATH):
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def main():
    net.train()
    loss_recorder = AverageMeter(num=6)
    for epoch in range(start, cfg.MAX_EPOCH):
        loss_recorder.reset()
        for i, (input, target, mask) in enumerate(train_loader, 1):
            input = input.to(device)
            target, mask = target.to(device), mask.to(device)
            output = net(input)

            output_detach = output.detach()

            iou = get_iou(output_detach, target, mask)

            mask_iou = torch.zeros_like(mask)
            mask_iou[:, :, :, 1:96:6] = mask[:, :, :, 1:96:6]
            mask_iou = mask_iou.type(torch.uint8)
            target[mask_iou] = iou

            # Get each outpout and target
            output_resp = output[:, :, :, 0:96:6]
            output_iou = output[:, :, :, 1:96:6]
            output_coor_x, output_coor_y = output[:, :, :, 2:96:6], output[:, :, :, 3:96:6]
            output_w, output_h = output[:, :, :, 4:96:6], output[:, :, :, 5:96:6]
            output_limb = output[:, :, :, 96:]
            # output_limb = output[:, :, :, 96:].detach().cpu().numpy().reshape((12, 12, 9, 9, 15))

            target_resp = target[:, :, :, 0:96:6]
            target_iou = target[:, :, :, 1:96:6]
            target_coor_x = target[:, :, :, 2:96:6] / cfg.CELL_SIZE - x_offset
            target_coor_y = target[:, :, :, 3:96:6] / cfg.CELL_SIZE - y_offset
            target_w = torch.sqrt(target[:, :, :, 4:96:6] / cfg.IMG_SIZE)
            target_h = torch.sqrt(target[:, :, :, 5:96:6] / cfg.IMG_SIZE)
            target_limb = target[:, :, :, 96:]
            # target_limb = target[:, :, :, 96:].cpu().numpy().reshape((12, 12, 9, 9, 15))

            mask_resp = mask[:, :, :, 0:96:6]
            mask_joint = mask[:, :, :, 1:96:6]
            mask_limb = mask[:, :, :, 96:]
            # mask_limb = mask[:, :, :, 96:].cpu().numpy().reshape((12, 12, 9, 9, 15))

            # draw_box(input, output_detach)

            for n in range(output.shape[0]):
                print('resp:')
                print(output_resp[n, :, :, :][target_resp[n, :, :, :].type(torch.uint8)])
                print('iou:')
                print(output_iou[n][mask_joint[n].type(torch.uint8)])
                print(target_iou[n][mask_joint[n].type(torch.uint8)])
                print('x:')
                print(output_coor_x[n][mask_joint[n].type(torch.uint8)])
                print(target_coor_x[n][mask_joint[n].type(torch.uint8)])
                print('y:')
                print(output_coor_y[n][mask_joint[n].type(torch.uint8)])
                print(target_coor_y[n][mask_joint[n].type(torch.uint8)])
                print('w:')
                print(output_w[n][mask_joint[n].type(torch.uint8)])
                print(target_w[n][mask_joint[n].type(torch.uint8)])
                print('h:')
                print(output_h[n][mask_joint[n].type(torch.uint8)])
                print(target_h[n][mask_joint[n].type(torch.uint8)])
                print('limb:')
                print(output_limb[n][target_limb[n].type(torch.uint8)])
                print()

            # Calculate each loss
            loss_resp = cfg.scale_resp * square_error(output_resp, target_resp, mask_resp)
            loss_iou = cfg.scale_iou * square_error(output_iou, target_iou, mask_joint)
            loss_coor = cfg.scale_coor * (square_error(output_coor_x, target_coor_x, mask_joint) +
                                          square_error(output_coor_y, target_coor_y, mask_joint))
            loss_size = cfg.scale_size * (square_error(output_w, target_w, mask_joint) +
                                          square_error(output_h, target_h, mask_joint))
            loss_limb = cfg.scale_limb * square_error(output_limb, target_limb, mask_limb)
            loss = loss_resp + loss_iou + loss_coor + loss_size + loss_limb
            loss_recorder.update(loss_resp.item(), loss_iou.item(), loss_coor.item(),
                                 loss_size.item(), loss_limb.item(), loss.item(),
                                 n=output.shape[0])

            # Modify learning rate
            global step
            step += 1
            lr = cfg.LR * (1 - step / (cfg.MAX_EPOCH * len(train_loader)))
            optimizer.param_groups[0]['lr'] = lr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('epoch:{}, step:{}, lr:{:.6f} resp_loss:{:.2f}, iou_loss:{:.3f}, coor_loss:{:.2f}, '
                      'size_loss:{:.3f}, limb_loss:{:.2f}, loss:{:.2f}'.format(epoch, i, lr,
                                                                               loss_recorder.avg[0],
                                                                               loss_recorder.avg[1],
                                                                               loss_recorder.avg[2],
                                                                               loss_recorder.avg[3],
                                                                               loss_recorder.avg[4],
                                                                               loss_recorder.avg[5]))
        print()
        writer.add_scalar('loss_resp', loss_recorder.avg[0], epoch)
        writer.add_scalar('loss_iou', loss_recorder.avg[1], epoch)
        writer.add_scalar('loss_coor', loss_recorder.avg[2], epoch)
        writer.add_scalar('loss_size', loss_recorder.avg[3], epoch)
        writer.add_scalar('loss_limb', loss_recorder.avg[4], epoch)
        writer.add_scalar('loss', loss_recorder.avg[5], epoch)

        torch.save({'epoch': epoch,
                    'step': step,
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   cfg.CHECKPOINT_PATH)


def get_iou(output, target, mask):
    output, target, mask = output[:, :, :, :96].clone(), target[:, :, :, :96].clone(), \
                           mask[:, :, :, :96].clone()

    output[:, :, :, 2::6] += x_offset
    output[:, :, :, 3::6] += y_offset

    mask[:, :, :, 0::6] = 0
    mask[:, :, :, 1::6] = 0
    mask = mask.type(torch.uint8)

    ox_p = output[mask][0::4] * cfg.CELL_SIZE
    oy_p = output[mask][1::4] * cfg.CELL_SIZE
    w_p = output[mask][2::4].pow(2) * cfg.IMG_SIZE
    h_p = output[mask][3::4].pow(2) * cfg.IMG_SIZE

    ox_gt = target[mask][0::4]
    oy_gt = target[mask][1::4]
    w_gt = target[mask][2::4]
    h_gt = target[mask][3::4]

    tl_x = torch.max(ox_p - 0.5 * w_p, ox_gt - 0.5 * w_gt)
    tl_y = torch.max(oy_p - 0.5 * h_p, oy_gt - 0.5 * h_gt)

    br_x = torch.min(ox_p + 0.5 * w_p, ox_gt + 0.5 * w_gt)
    br_y = torch.min(oy_p + 0.5 * h_p, oy_gt + 0.5 * h_gt)

    delta_x, delta_y = br_x - tl_x, br_y - tl_y
    condition = (delta_x < 0) | (delta_y < 0)
    intersection = torch.where(condition, torch.zeros_like(delta_x), delta_x * delta_y)
    union = torch.max(w_p * h_p + w_gt * h_gt - intersection,
                      torch.full_like(delta_x, 1e-10))

    iou = intersection / union
    iou = torch.clamp(iou, 0, 1)

    return iou


def square_error(output, target, mask):
    return 1 / output.shape[0] * torch.sum(mask * (output - target).pow(2))


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, num):
        self.avg = np.zeros(num)
        self.sum = np.zeros(num)
        self.count = 0

    def reset(self):
        self.avg[:] = 0
        self.sum[:] = 0
        self.count = 0

    def update(self, *val, n=1):
        val_array = np.array(val)
        self.sum += val_array * n
        self.count += n
        self.avg = self.sum / self.count


def draw_box(img, output):
    img = img.cpu().numpy().transpose(0, 2, 3, 1)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    output = output.cpu().numpy()
    output[:, :, :, 2:96:6] = (output[:, :, :, 2:96:6] + x_offset.cpu().numpy()) * cfg.CELL_SIZE
    output[:, :, :, 3:96:6] = (output[:, :, :, 3:96:6] + y_offset.cpu().numpy()) * cfg.CELL_SIZE
    output[:, :, :, 4:96:6] = output[:, :, :, 4:96:6] ** 2
    output[:, :, :, 5:96:6] = output[:, :, :, 5:96:6] ** 2
    output[:, :, :, 4:96:6] *= cfg.IMG_SIZE
    output[:, :, :, 5:96:6] *= cfg.IMG_SIZE

    n = img.shape[0]
    alpha = 0.8
    for i in range(n):
        overlay = img[i].copy()
        for j in range(1, 16):
            exist_output = output[i, :, :, 6 * j] > cfg.thres1
            box_output = output[i][exist_output][:, 6 * j + 2:6 * j + 6]
            box_output[:, 0], box_output[:, 1], box_output[:, 2], box_output[:, 3] = \
                box_output[:, 0] - 0.5 * box_output[:, 2], box_output[:, 1] - 0.5 * box_output[:, 3], \
                box_output[:, 0] + 0.5 * box_output[:, 2], box_output[:, 1] + 0.5 * box_output[:, 3]

            for b in box_output:
                cv2.rectangle(overlay, (b[0], b[1]), (b[2], b[3]), colors[j - 1], -1)

        img_transparent = cv2.addWeighted(overlay, alpha, img[i], 1 - alpha, 0)[:, :, ::-1]
        img_transparent[:, ::cfg.CELL_SIZE, :] = np.array([1., 1, 1])
        img_transparent[::cfg.CELL_SIZE, :, :] = np.array([1., 1, 1])
        cv2.namedWindow('joint box', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('joint box', 800, 800)
        cv2.imshow('joint box', img_transparent)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
