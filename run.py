from data.dataset import listDataset
import torch
import torch as t
import torch.nn as nn
from torch.nn import Module
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from config import DefaultConfig
import models
from tensorboardX import SummaryWriter
import os
import sys
import time
from models.debug import conv_vis
from utils.helper import *
from utils.box_utils import *
import numpy as np
from torch.autograd import Variable


opt = DefaultConfig()

def adjust_learning_rate(optimizer, batch):
    lr = opt.lr
    steps = opt.steps
    scales = opt.scales

    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / opt.batch_size
    return lr

def train(**kwargs):
    previous_acc = -10.

    global internal_calibration, corners3D, vertices
    processed_batches = 0
    opt.parse(kwargs)

    mesh = MeshPly('/'.join(opt.train_data_root.split('/')[:-3])+'/LINEMOD/ape/ape.ply')
    vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()

    corners3D = get_3D_corners(vertices)
    internal_calibration = get_camera_intrinsic()


    writer = SummaryWriter(opt.debug_file)
    bg_files = get_all_files(opt.bg_images)

    #############################
    #  model define
    model = getattr(models, opt.model)()

    model.train()
    if opt.use_gpu:
        model.cuda()
    if opt.load_model_path is not None:
        model.load(opt.load_model_path)
    if opt.pretrain is not None:
        model.load_weights(opt.pretrain)

    #############################
    #  dataset preparation
    transform_train = T.Compose([T.ToTensor()])

    transform_val = T.Compose([T.ToTensor()])
    val_width, val_height = 416,416
    train_dataset = listDataset(opt.train_data_root, shape=(416, 416),
                                                               shuffle=True,
                                                               transform=transform_train,
                                                               train=True,
                                                               bg_file_names=bg_files)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(listDataset(opt.test_data_root, shape=(val_width, val_height),
                                                             shuffle=False, bg_file_names=bg_files,
                                                             transform=transform_val,
                                                             train=False),
                                                 batch_size=1, shuffle=False)
    #############################
    #  loss define
    criterion = model.loss

    #############################
    #  optimizer define   !!! SGD will be better
    lr = opt.lr
    optimizer = t.optim.SGD(params=model.parameters(), lr=opt.lr / opt.batch_size, momentum=opt.momentum,
                            weight_decay=opt.weight_decay)

    step = 0

    lr = adjust_learning_rate(optimizer, processed_batches)

    count = 0
    for epoch in range(opt.max_epoch):

        model.train()
        start_time = time.time()

        for ii, (data, label) in enumerate(train_dataloader):

            input, target = data, label
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            adjust_learning_rate(optimizer, processed_batches)
            processed_batches = processed_batches + 1


            # debug  grad  explosion
            # for name, param in model.feature_layers.named_parameters():
            #     print(name, param.grad[:10])
            #     break


            batch_time = time.time() - start_time
            if count == 0:
                print("One Batch Time is {}".format(batch_time))
                print("One epoch Time is {}".format(batch_time * (1. * len(train_dataset)/opt.batch_size)))
                count += 1
            writer.add_scalar('loss', loss.item(), processed_batches)
        print("{}th epoch:  loss is {}, learning rate is {}.".format(epoch, loss.item(), lr))

        if epoch%50==0:
            if epoch!=0:
                print('testttt ')
                acc_dict, err_dict = eval(model, val_dataloader)
                writer.add_scalars('metrics_acc', acc_dict, epoch)
                writer.add_scalars('mettrix_err', err_dict, epoch)

                if acc_dict['px5_acc_raw'] > previous_acc:
                    previous_acc = acc_dict['px5_acc_raw']
                    writer.add_text('Text', 'best model logged at step:' + str(lr) + '_' + str(epoch), epoch)
                    model.save(opt.save_name)

            if epoch%500==0:
                continue#conv_vis(writer, model.state_dict(), epoch)

        #model.save()



def eval(model, test_loader):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

        # Set the module in evaluation mode (turn off dropout, batch normalization etc.)

    model.eval()

    # Parameters
    num_classes = model.num_classes
    anchors = model.anchors
    num_anchors = model.num_anchors
    testtime = True
    testing_error_trans = 0.0
    testing_error_angle = 0.0
    testing_error_pixel = 0.0
    testing_samples = 0.0
    errs_2d = []
    errs_3d = []
    errs_trans = []
    errs_angle = []
    errs_corner2D = []

    print("   Testing...")
    print("   Number of test samples: %d" % len(test_loader.dataset))
    notpredicted = 0
    # Iterate through test examples
    for batch_idx, (data, target) in enumerate(test_loader):
        t1 = time.time()
        # Pass the data to GPU
        if 1:
            data = data.cuda()
            target = target.cuda()
        # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
        data = Variable(data, volatile=True)
        t2 = time.time()
        # Formward pass
        output = model(data).data
        t3 = time.time()
        # Using confidence threshold, eliminate low-confidence predictions
        all_boxes = get_region_boxes(output, .1, num_classes, anchors, num_anchors)
        t4 = time.time()
        # Iterate through all batch elements
        for i in range(output.size(0)):
            # For each image, get all the predictions
            boxes = all_boxes[i]
            # For each image, get all the targets (for multiple object pose estimation, there might be more than 1 target per image)
            truths = target[i].view(-1, 21)
            # Get how many object are present in the scene
            num_gts = truths_length(truths)

            # Iterate through each ground-truth object
            for k in range(num_gts):
                box_gt = [truths[k][1], truths[k][2], truths[k][3], truths[k][4], truths[k][5], truths[k][6],
                          truths[k][7], truths[k][8], truths[k][9], truths[k][10], truths[k][11], truths[k][12],
                          truths[k][13], truths[k][14], truths[k][15], truths[k][16], truths[k][17], truths[k][18], 1.0,
                          1.0, truths[k][0]]
                best_conf_est = -1

                # If the prediction has the highest confidence, choose it as our prediction
                for j in range(len(boxes)):
                    if boxes[j][18] > best_conf_est:
                        best_conf_est = boxes[j][18]
                        box_pr = boxes[j]
                        match = corner_confidence9(torch.FloatTensor(box_gt[:18]), torch.FloatTensor(boxes[j][:18]))
                im_width=412
                im_height=412
                # Denormalize the corner predictions
                corners2D_gt = np.array(np.reshape(box_gt[:18], [9, 2]), dtype='float32')
                corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
                corners2D_gt[:, 0] = corners2D_gt[:, 0] * im_width
                corners2D_gt[:, 1] = corners2D_gt[:, 1] * im_height
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height

                # Compute corner prediction error
                corner_norm = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
                corner_dist = np.mean(corner_norm)
                errs_corner2D.append(corner_dist)

                # print(corner_norm)
                # Compute [R|t] by pnp
                R_gt, t_gt = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)),
                                          dtype='float32'), corners2D_gt,
                                 np.array(internal_calibration, dtype='float32'))
                R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)),
                                          dtype='float32'), corners2D_pr,
                                 np.array(internal_calibration, dtype='float32'))

                # Compute errors

                # Compute translation error
                trans_dist = np.sqrt(np.sum(np.square(t_gt - t_pr)))
                errs_trans.append(trans_dist)

                # Compute angle error
                angle_dist = calcAngularDistance(R_gt, R_pr)
                errs_angle.append(angle_dist)

                # Compute pixel error
                Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
                Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
                proj_2d_gt = compute_projection(vertices, Rt_gt, internal_calibration)
                proj_2d_pred = compute_projection(vertices, Rt_pr, internal_calibration)
                norm = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
                pixel_dist = np.mean(norm)
                errs_2d.append(pixel_dist)

                # Compute 3D distances
                transform_3d_gt = compute_transformation(vertices, Rt_gt)
                transform_3d_pred = compute_transformation(vertices, Rt_pr)
                norm3d = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
                vertex_dist = np.mean(norm3d)
                errs_3d.append(vertex_dist)

                # Sum errors
                testing_error_trans += trans_dist
                testing_error_angle += angle_dist
                testing_error_pixel += pixel_dist
                testing_samples += 1

        t5 = time.time()

    # Compute 2D projection, 6D pose and 5cm5degree scores
    px_threshold = 5
    vx_threshold=0.0103
    eps=1e-5
    acc = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d) + eps)
    acc3d = len(np.where(np.array(errs_3d) <= vx_threshold)[0]) * 100. / (len(errs_3d) + eps)
    acc5cm5deg = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (
            len(errs_trans) + eps)
    corner_acc = len(np.where(np.array(errs_corner2D) <= px_threshold)[0]) * 100. / (len(errs_corner2D) + eps)
    mean_err_2d = np.mean(errs_2d)
    mean_corner_err_2d = np.mean(errs_corner2D)
    nts = float(testing_samples)

    if testtime:
        print('-----------------------------------')
        print('  tensor to cuda : %f' % (t2 - t1))
        print('         predict : %f' % (t3 - t2))
        print('get_region_boxes : %f' % (t4 - t3))
        print('            eval : %f' % (t5 - t4))
        print('           total : %f' % (t5 - t1))
        print('-----------------------------------')

    # Print test statistics
    print("   Mean corner error is %f" % (mean_corner_err_2d))
    print('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
    print('   Acc using {} vx 3D Transformation = {:.2f}%'.format(vx_threshold, acc3d))
    print('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
    print('   Translation error: %f, angle error: %f' % (
    testing_error_trans / (nts + eps), testing_error_angle / (nts + eps)))

    # Register losses and errors for saving later on
#    testing_iters.append(niter)
    testing_errors_trans.append(testing_error_trans / (nts + eps))
    testing_errors_angle.append(testing_error_angle / (nts + eps))
    testing_errors_pixel.append(testing_error_pixel / (nts + eps))
    testing_accuracies.append(acc)
    # Register losses and errors for saving later on

    h =  ({'px5_acc_after':acc, 'add_acc': acc3d, 'px5_acc_raw':corner_acc, 'acc5cm5deg': acc5cm5deg},
            {'mean_err_2d': mean_err_2d, 'mean_err_2d_raw': mean_corner_err_2d})
    return h


def test(**kwargs):
    previous_acc = -10.

    global internal_calibration, corners3D, vertices
    processed_batches = 0
    opt.parse(kwargs)

    mesh = MeshPly('/'.join(opt.train_data_root.split('/')[:-3])+'/LINEMOD/ape/ape.ply')
    vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()

    corners3D = get_3D_corners(vertices)
    internal_calibration = get_camera_intrinsic()

    bg_files = get_all_files(opt.bg_images)

    #############################
    #  model define
    model = getattr(models, opt.model)()

    model.train()
    if opt.use_gpu:
        model.cuda()
    if opt.load_model_path is not None:
        model.load(opt.load_model_path)
    if opt.pretrain is not None:
        model.load_weights(opt.pretrain)

    #############################
    #  dataset preparation
    transform_train = T.Compose([T.ToTensor()])

    transform_val = T.Compose([T.ToTensor()])
    val_width, val_height = 416,416
    train_dataset = listDataset(opt.train_data_root, shape=(416, 416),
                                                               shuffle=True,
                                                               transform=transform_train,
                                                               train=True,
                                                               bg_file_names=bg_files)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batch_size, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(listDataset(opt.test_data_root, shape=(val_width, val_height),
                                                             shuffle=False, bg_file_names=bg_files,
                                                             transform=transform_val,
                                                             train=False),
                                                 batch_size=1, shuffle=False)
    eval(model, val_dataloader)

def help():
    """
     python file.py help
    """

    print("""
    usage : python {0} <function> [--args=value,]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

class MeshPly:
    def __init__(self, filename, color=[0., 0., 0.]):

        f = open(filename, 'r')
        self.vertices = []
        self.colors = []
        self.indices = []
        self.normals = []

        vertex_mode = False
        face_mode = False

        nb_vertices = 0
        nb_faces = 0

        idx = 0

        with f as open_file_object:
            for line in open_file_object:
                elements = line.split()
                if vertex_mode:
                    self.vertices.append([float(i) for i in elements[:3]])
                    self.normals.append([float(i) for i in elements[3:6]])

                    if elements[6:9]:
                        self.colors.append([float(i) / 255. for i in elements[6:9]])
                    else:
                        self.colors.append([float(i) / 255. for i in color])

                    idx += 1
                    if idx == nb_vertices:
                        vertex_mode = False
                        face_mode = True
                        idx = 0
                elif face_mode:
                    self.indices.append([float(i) for i in elements[1:4]])
                    idx += 1
                    if idx == nb_faces:
                        face_mode = False
                elif elements[0] == 'element':
                    if elements[1] == 'vertex':
                        nb_vertices = int(elements[2])
                    elif elements[1] == 'face':
                        nb_faces = int(elements[2])
                elif elements[0] == 'end_header':
                    vertex_mode = True

if __name__ == '__main__':
    training_iters = []
    training_losses = []
    testing_iters = []
    testing_losses = []
    testing_errors_trans = []
    testing_errors_angle = []
    testing_errors_pixel = []
    testing_accuracies = []
    corners3D = None
    internal_calibration = None
    vertices = None
    import fire
    fire.Fire()
    # train(pretrain='/workspace/liuzhen/remote_workspace/pose/model.weights',
    #       train_data_root='/workspace/liuzhen/remote_workspace/pose/LINEMOD/self/train1.txt',
    #       test_data_root = '/workspace/liuzhen/remote_workspace/pose/LINEMOD/self/test1.txt',
    #       max_epoch=3000,
    #       batch_size = 16,
    #       load_model_path=None)#'/workspace/liuzhen/remote_workspace/pose/lrsmall.pth',)