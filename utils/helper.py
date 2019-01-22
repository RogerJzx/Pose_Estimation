import os
import numpy as np
import  time
import torch
import cv2
from torch.autograd import Variable
from scipy import spatial


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)



def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def get_all_files(directory):
    files = []

    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            files.append(os.path.join(directory, f))
        else:
            files.extend(get_all_files(os.path.join(directory, f)))
    return files


def calcAngularDistance(gt_rot, pr_rot):
    rotDiff = np.dot(gt_rot, np.transpose(pr_rot))
    trace = np.trace(rotDiff)
    return np.rad2deg(np.arccos((trace - 1.0) / 2.0))


def get_camera_intrinsic():
    K = np.zeros((3, 3), dtype='float64')
    K[0, 0], K[0, 2] = 572.4114, 325.2611
    K[1, 1], K[1, 2] = 573.5704, 242.0489
    K[2, 2] = 1.
    return K


def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :] / camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :] / camera_projection[2, :]
    return projections_2d


def compute_transformation(points_3D, transformation):
    return transformation.dot(points_3D)


def calc_pts_diameter(pts):
    diameter = -1
    for pt_id in range(pts.shape[0]):
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter


def adi(pts_est, pts_gt):
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)
    e = nn_dists.mean()
    return e


def get_camera_intrinsic():
    K = np.zeros((3, 3), dtype='float64')
    K[0, 0], K[0, 2] = 572.4114, 325.2611
    K[1, 1], K[1, 2] = 573.5704, 242.0489
    K[2, 2] = 1.
    return K


def get_3D_corners(vertices):
    min_x = np.min(vertices[0, :])
    max_x = np.max(vertices[0, :])
    min_y = np.min(vertices[1, :])
    max_y = np.max(vertices[1, :])
    min_z = np.min(vertices[2, :])
    max_z = np.max(vertices[2, :])

    corners = np.array([[min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])

    corners = np.concatenate((np.transpose(corners), np.ones((1, 8))), axis=0)
    return corners

def pnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32')

    assert points_2D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    _, R_exp, t = cv2.solvePnP(points_3D,
                               # points_2D,
                               np.ascontiguousarray(points_2D[:, :2]).reshape((-1, 1, 2)),
                               cameraMatrix,
                               distCoeffs)
    # , None, None, False, cv2.SOLVEPNP_UPNP)

    # R_exp, t, _ = cv2.solvePnPRansac(points_3D,
    #                            points_2D,
    #                            cameraMatrix,
    #                            distCoeffs,
    #                            reprojectionError=12.0)
    #

    R, _ = cv2.Rodrigues(R_exp)
    # Rt = np.c_[R, t]
    return R, t
def get_region_boxes(output, conf_thresh, num_classes, only_objectness=1, validation=False):
    # Parameters
    anchor_dim = 1
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (19 + num_classes) * anchor_dim)
    h = output.size(2)
    w = output.size(3)

    # Activation
    t0 = time.time()
    all_boxes = []
    max_conf = -100000
    output = output.view(batch * anchor_dim, 19 + num_classes, h * w).transpose(0, 1).contiguous().view(
        19 + num_classes, batch * anchor_dim * h * w)
    grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch * anchor_dim, 1, 1).view(
        batch * anchor_dim * h * w).cuda()
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * anchor_dim, 1, 1).view(
        batch * anchor_dim * h * w).cuda()
    xs0 = torch.sigmoid(output[0]) + grid_x
    ys0 = torch.sigmoid(output[1]) + grid_y
    xs1 = output[2] + grid_x
    ys1 = output[3] + grid_y
    xs2 = output[4] + grid_x
    ys2 = output[5] + grid_y
    xs3 = output[6] + grid_x
    ys3 = output[7] + grid_y
    xs4 = output[8] + grid_x
    ys4 = output[9] + grid_y
    xs5 = output[10] + grid_x
    ys5 = output[11] + grid_y
    xs6 = output[12] + grid_x
    ys6 = output[13] + grid_y
    xs7 = output[14] + grid_x
    ys7 = output[15] + grid_y
    xs8 = output[16] + grid_x
    ys8 = output[17] + grid_y
    det_confs = torch.sigmoid(output[18])
    cls_confs = torch.nn.Softmax()(Variable(output[19:19 + num_classes].transpose(0, 1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()

    # GPU to CPU
    sz_hw = h * w
    sz_hwa = sz_hw * anchor_dim
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs0 = convert2cpu(xs0)
    ys0 = convert2cpu(ys0)
    xs1 = convert2cpu(xs1)
    ys1 = convert2cpu(ys1)
    xs2 = convert2cpu(xs2)
    ys2 = convert2cpu(ys2)
    xs3 = convert2cpu(xs3)
    ys3 = convert2cpu(ys3)
    xs4 = convert2cpu(xs4)
    ys4 = convert2cpu(ys4)
    xs5 = convert2cpu(xs5)
    ys5 = convert2cpu(ys5)
    xs6 = convert2cpu(xs6)
    ys6 = convert2cpu(ys6)
    xs7 = convert2cpu(xs7)
    ys7 = convert2cpu(ys7)
    xs8 = convert2cpu(xs8)
    ys8 = convert2cpu(ys8)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()

    # Boxes filter
    for b in range(batch):
        boxes = []
        max_conf = -1
        for cy in range(h):
            for cx in range(w):
                for i in range(anchor_dim):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > max_conf:
                        max_conf = conf
                        max_ind = ind

                    if conf > conf_thresh:
                        bcx0 = xs0[ind]
                        bcy0 = ys0[ind]
                        bcx1 = xs1[ind]
                        bcy1 = ys1[ind]
                        bcx2 = xs2[ind]
                        bcy2 = ys2[ind]
                        bcx3 = xs3[ind]
                        bcy3 = ys3[ind]
                        bcx4 = xs4[ind]
                        bcy4 = ys4[ind]
                        bcx5 = xs5[ind]
                        bcy5 = ys5[ind]
                        bcx6 = xs6[ind]
                        bcy6 = ys6[ind]
                        bcx7 = xs7[ind]
                        bcy7 = ys7[ind]
                        bcx8 = xs8[ind]
                        bcy8 = ys8[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx0 / w, bcy0 / h, bcx1 / w, bcy1 / h, bcx2 / w, bcy2 / h, bcx3 / w, bcy3 / h, bcx4 / w,
                               bcy4 / h, bcx5 / w, bcy5 / h, bcx6 / w, bcy6 / h, bcx7 / w, bcy7 / h, bcx8 / w, bcy8 / h,
                               det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
            if len(boxes) == 0:
                bcx0 = xs0[max_ind]
                bcy0 = ys0[max_ind]
                bcx1 = xs1[max_ind]
                bcy1 = ys1[max_ind]
                bcx2 = xs2[max_ind]
                bcy2 = ys2[max_ind]
                bcx3 = xs3[max_ind]
                bcy3 = ys3[max_ind]
                bcx4 = xs4[max_ind]
                bcy4 = ys4[max_ind]
                bcx5 = xs5[max_ind]
                bcy5 = ys5[max_ind]
                bcx6 = xs6[max_ind]
                bcy6 = ys6[max_ind]
                bcx7 = xs7[max_ind]
                bcy7 = ys7[max_ind]
                bcx8 = xs8[max_ind]
                bcy8 = ys8[max_ind]
                cls_max_conf = cls_max_confs[max_ind]
                cls_max_id = cls_max_ids[max_ind]
                det_conf = det_confs[max_ind]
                box = [bcx0 / w, bcy0 / h, bcx1 / w, bcy1 / h, bcx2 / w, bcy2 / h, bcx3 / w, bcy3 / h, bcx4 / w,
                       bcy4 / h, bcx5 / w, bcy5 / h, bcx6 / w, bcy6 / h, bcx7 / w, bcy7 / h, bcx8 / w, bcy8 / h,
                       det_conf, cls_max_conf, cls_max_id]
                boxes.append(box)
                all_boxes.append(boxes)
            else:
                all_boxes.append(boxes)

        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1 - t0))
        print('        gpu to cpu : %f' % (t2 - t1))
        print('      boxes filter : %f' % (t3 - t2))
        print('---------------------------------')
    return all_boxes
