# coding: utf-8
import torchvision.utils as vutils

def conv_vis(writer, params, n_iter):
    # 将一个卷积层的卷积核绘制在一起，每一行是一个feature map的卷积核
    for k, v in params.items():
        if 'conv' in k and 'weight' in k and v.shape[-1] != 1:
            c_int = v.size()[1]     # 输入层通道数
            c_out = v.size()[0]     # 输出层通道数

 
            k_w, k_h = v.size()[-1], v.size()[-2]
            kernel_all = v.view(-1, 1, k_w, k_h)
            kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=c_int)  # 1*输入通道数, w, h
            writer.add_image(k + '_all', kernel_grid, global_step=n_iter)
