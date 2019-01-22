import torch


def corner_confidence9(gt_corners, pr_corners, th=80, sharpness=2, im_width=640, im_height=480):
    ''' gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (18,) type: list
        pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (18,), type: list
        th        : distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a list of shape (9,) with 9 confidence values 

        article :: Real-Time Seamless Single Shot 6D Object Pose Prediction
    '''
    dist =(gt_corners) - pr_corners
    dist = dist.view(9, 2)
    dist[:, 0] = dist[:, 0] * im_width
    dist[:, 1] = dist[:, 1] * im_height
    eps = 1e-5
    dist  = torch.sqrt(torch.sum((dist)**2, dim=1))
    mask  = (dist < th).type(torch.FloatTensor)
    conf  = torch.exp(sharpness * (1.0 - dist/th)) - 1
    conf0 = torch.exp(torch.FloatTensor([sharpness])) - 1 + eps
    conf  = conf / conf0.repeat(9, 1)
    # conf = 1.0 - dist/th
    conf  = mask * conf
    return torch.mean(conf)