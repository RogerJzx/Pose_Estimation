import sys, torch
import torch.nn as nn
from models.BasicModule import BasicModule
from utils.helper import *
import torch.nn.functional as F
from utils.box_utils import corner_confidence9
from collections import OrderedDict

## yolo v2 config
configs = [
    ## conv : filters  ks  stride padding
    ####################416 * 416 * 32 ##########################

    {'conv': [32, 3, 1, 1]},
    ####################208 * 208 * 64 ##########################

    ## pool : ks stride
    {'pool': [2, 2]},
    {'conv': [64, 3, 1, 1]},
    ####################104 * 104 * 128 ##########################

    {'pool': [2, 2]},

    {'conv': [128, 3, 1, 1]},
    {'conv': [64, 1, 1, 0]},
    {'conv': [128, 3, 1, 1]},
    {'pool': [2, 2]},
    ####################52 * 52 * 256 ##########################

    {'conv': [256, 3, 1, 1]},
    {'conv': [128, 1, 1, 0]},
    {'conv': [256, 3, 1, 1]},
    {'pool': [2, 2]},

    ####### route
    # {'conv': [256, 3, 1, 1]},
    # {'conv': [256, 3, 1, 1]},  # 14   or  -1
    # {'route': [-9]},
    #
    # {'conv': [16, 1, 1, 0]},
    # {'reorg': [4]},
    # {'route': [-1, -4]},

    ####################26 * 26 * 512 ##########################
    {'conv': [512, 3, 1, 1]},
    {'conv': [256, 1, 1, 0]},
    {'conv': [512, 3, 1, 1]},
    {'conv': [256, 1, 1, 0]},
    {'conv': [512, 3, 1, 1]},

    ############################################################
    {'pool': [2, 2]},

    {'conv': [1024, 3, 1, 1]},
    {'conv': [512, 1, 1, 0]},
    {'conv': [1024, 3, 1, 1]},
    {'conv': [512, 1, 1, 0]},
    {'conv': [1024, 3, 1, 1]},
    ####### route
    {'conv': [1024, 3, 1, 1]},
    {'conv': [1024, 3, 1, 1]},  # 24   or  -1
    {'route': [-9]},

    {'conv': [64, 1, 1, 0]},
    {'reorg': [2]},
    {'route': [-1, -4]},

    {'conv': [1024, 3, 1, 1]},
    {'region': []}
    # {'conv': [128, 1, 1, 0]},
]


# route and shoutcut
class EmptyModule(nn.Module):
    def __init__(self, skip_backup):
        super(EmptyModule, self).__init__()
        self.skip_backup = skip_backup
    def forward(self, input):
        return input

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, int(H / hs), hs,int( W / ws), ws).transpose(3, 4).contiguous()
        x = x.view(B, C,int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, int(H / hs),int(W / ws)).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C,int( H / hs), int(W / ws))
        return x

def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]));
    start = start + num_w
    return start


def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]));
    start = start + num_w
    return start


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    temp = torch.Tensor(buf[start:start+num_w].shape[0])
    temp.copy_(torch.from_numpy(buf[start:start+num_w]))
    p=(conv_model.weight.data.shape)
    resize = temp.view(*p)
    conv_model.weight.data.copy_(resize)#torch.from_numpy(buf[start:start + num_w]));
    start = start + num_w
    return start


class DarkNet(BasicModule):
    def __init__(self):
        super(DarkNet, self).__init__()
        layers = nn.ModuleList()
        in_channels = 3
        prev_channel = 3
        channels = []
        for i, l in enumerate(configs):

            if ('conv') in l:
                layers.append(nn.Sequential(OrderedDict([
                    ('conv'+str(i), nn.Conv2d(in_channels, l['conv'][0], kernel_size=l['conv'][1], stride=l['conv'][2],
                              padding=l['conv'][3],bias=False)),
                    ('batch'+str(i), nn.BatchNorm2d(l['conv'][0], eps=1e-4)),
                    ('leak'+str(i), nn.LeakyReLU(.1, inplace=True))
                ])))
                in_channels = l['conv'][0]
                channels.append(in_channels)
                prev_channel = in_channels
            elif ('pool') in l:
                layers.append(nn.MaxPool2d(kernel_size=l['pool'][0], stride=l['pool'][1]))
                channels.append(prev_channel)

            elif ('route') in l:
                layers.append(EmptyModule(l['route']))

                if len(l['route']) == 1:
                    in_channels = channels[i + l['route'][0]]

                    channels.append(in_channels)
                    prev_channel = in_channels
                elif len(l['route']) == 2:
                    in_channels = channels[i + l['route'][0]] + channels[i + l['route'][1]]

                    channels.append(in_channels)
                    prev_channel = in_channels
                else:
                    raise Exception('has not define yet')
            elif ('reorg') in l:

                prev_channel = prev_channel * l['reorg'][0] * l['reorg'][0]

                layers.append(Reorg(l['reorg'][0]))
                channels.append(prev_channel)
            elif ('region') in l:
                loss = RegionLoss()

                loss.anchors = [1.4820, 2.2412, 2.0501, 3.1265, 2.3946, 4.6891, 3.1018, 3.9910, 3.4879, 5.8851]
                loss.num_classes = 1
                loss.num_anchors = 1
                loss.anchor_step = len(loss.anchors) / loss.num_anchors
                loss.object_scale = 5.
                loss.noobject_scale = .1
                loss.class_scale = 1
                loss.coord_scale = 1
                channels.append(prev_channel)
                self.loss = loss

        self.layers_ = layers
        self.iter = 0
        self.last_conv = nn.Conv2d(channels[-1], 20, 1, 1)

        self.anchors = self.loss.anchors
        self.num_anchors = self.loss.num_anchors
        self.anchor_step = self.loss.anchor_step
        self.num_classes = self.loss.num_classes


    def forward(self, x):
        output = dict()

        for i, m in enumerate(self.layers_):
            x = m(x)

            if isinstance(m, EmptyModule):

                # backup will append output
                if len(m.skip_backup) == 1:
                    output[i] = output[i + m.skip_backup[0]]
                    x = output[i]

                elif len(m.skip_backup) == 2:

                    x = torch.cat((output[i + m.skip_backup[0]], output[i + m.skip_backup[1]]), 1)
                else:
                    raise Exception('has not define yet')
            else:
                output[i] = x

        return self.last_conv(x)

    def load_weights(self, weightfile):
        import numpy as np
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        start = 0

        for i, l in enumerate(configs):
            if ('conv') in l:
                model = self.layers_[i]
                start = load_conv_bn(buf, start, model[0], model[1])
            else:
                pass

        def load_conv(buf, start, conv_model):
            num_w = conv_model.weight.numel()
            num_b = conv_model.bias.numel()
            conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]));
            start = start + num_b
            temp = torch.Tensor(buf[start:start + num_w].shape[0])
            temp.copy_(torch.from_numpy(buf[start:start + num_w]))
            p = (conv_model.weight.data.shape)
            resize = temp.view(*p)
            conv_model.weight.data.copy_(resize)  #

            #conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]));
            start = start + num_w
            return start
        start = load_conv(buf, start, self.last_conv)
        print(start)

def build_targets(predict_corners, target, num_anchors, nH, nW, noobject_scale, object_scale):

    nB = target.size(0)
    nA = num_anchors
    conf_mask = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask = torch.zeros(nB, nA, nH, nW)
    tar = torch.zeros(nB, nA, 18, nH, nW)
    tconf = torch.zeros(nB, nA, nH, nW)
    tcls = torch.zeros(nB, nA, nH, nW)

    nGT, nCorrect = 0, 0
    nAnchors = nA * nH * nW
    nPixels = nH * nW

    for b in range(nB):
        for t in range(50):
            if target[b][t * 21 + 1] == 0:
                break
            nGT = nGT + 1
            gx0 = target[b][t * 21 + 1] * nW
            gy0 = target[b][t * 21 + 2] * nH
            gi0 = int(gx0)
            gj0 = int(gy0)

            best_n = 0  # 1 anchor box
            gt_box = target[b][t * 21 + 1: t * 21 + 19].to(device=torch.device('cpu')).float()

            pred_box = predict_corners[b * nAnchors + best_n * nPixels + gj0 * nW + gi0]

            conf = corner_confidence9(gt_box, pred_box)
            coord_mask[b][best_n][gj0][gi0] = 1
            cls_mask[b][best_n][gj0][gi0] = 1
            conf_mask[b][best_n][gj0][gi0] = object_scale

            for i in range(9):
                tar[b][best_n][2*i][gj0][gi0] = target[b][t * 21 + 1 + 2*i] * nW - gi0
                tar[b][best_n][2*i+1][gj0][gi0] = target[b][t * 21 + 1 + 2*i+1] * nH - gj0

            tconf[b][best_n][gj0][gi0] = conf
            tcls[b][best_n][gj0][gi0] = target[b][t * 21]

            if conf > 0.5:
                nCorrect = nCorrect + 1
    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tconf, tcls, tar



class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1, device=torch.device('cuda')):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) / num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0
        self.device = device

    def forward(self, output, target):
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        output = output.view(nB, nA, (19 + nC), nH, nW)

        grid_x = torch.linspace(0, nW -1 , nW).repeat(nH, 1).repeat(nB, nA, 9, 1, 1).to(self.device)
        grid_y = torch.linspace(0, nH -1 , nH).repeat(nW, 1).t().repeat(nB, nA, 9, 1, 1).to(self.device)

        # centriod point uses sigmoid activation function
        output[:, :, 0] = torch.sigmoid(output[:, :, 0])
        output[:, :, 1] = torch.sigmoid(output[:, :, 1])

        index_xs = torch.arange(0, 18, 2).to(self.device).view(1,1,9,1,1).expand(nB, nA, 9, nH, nW)
        index_ys = torch.arange(1, 19, 2).to(self.device).view(1,1,9,1,1).expand(nB, nA, 9, nH, nW)

        out_xs = (output.data.gather(2, index_xs) + grid_x) / nW
        out_ys = (output.data.gather(2, index_ys) + grid_y) / nH
        # deep copy
        predict_grids = output.data.clone()
        predict_grids.scatter_(2, index_xs, out_xs)
        predict_grids.scatter_(2, index_ys, out_ys)


        predict_corners = predict_grids.index_select(2, torch.arange(0,18).to(self.device)).cpu()
        predict_corners = predict_corners.transpose(0,1).transpose(0,2).contiguous()
        predict_corners = predict_corners.view(18,-1).transpose(0,1).view(-1,18)

        nGt, nCorrect, coord_mask, conf_mask, cls_mask, tconf, tcls, tar = \
            build_targets(predict_corners, target.data, nA, nH, nW, self.noobject_scale,
                          self.object_scale)

        tar = tar.to(self.device)
        coord_mask = coord_mask[:, :, None].to(self.device)

        xys_pred = output.index_select(2, torch.arange(0, 18).to(self.device))
        loss_xy = self.coord_scale  * nn.MSELoss(size_average=False)(xys_pred * coord_mask, tar * coord_mask) / 2.0

        conf = F.sigmoid(output.index_select(2, (torch.Tensor([18]).long().to(self.device))).view(nB, nA, nH, nW))
        conf_mask = conf_mask.to(self.device).sqrt()
        loss_conf = nn.MSELoss(size_average=False)(conf * conf_mask, tconf.to(self.device) * conf_mask) / 2.0
        loss = loss_xy + loss_conf

        nProposals = int((conf > 0.25).sum().data)
        print('nGT %d, recall %d, proposals %d, loss: xy %f, conf %f,cls  total %f' % (
                     nGt, nCorrect, nProposals, loss_xy.data, loss_conf.data, loss.data))

        return loss