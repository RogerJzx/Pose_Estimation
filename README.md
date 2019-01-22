## Pose Estimation
It is a simpler version of https://github.com/Microsoft/singleshotpose/tree/master/multi_obj_pose_estimation
## Required packages:
- fire
- torch, torchvision, torchnet
- inspect


```bash
    usage : python main.py <function> [--args=value,]
    <function> := train | test | help
    example:
            python main.py train  --lr=0.01 ...
            python main.py test --dataset='path/to/dataset/' ...
            python main.py help
    avaiable args:
class DefaultConfig(object):
    env = 'default' # visdom 环境
    model = 'AlexNet' # 使用的模型

    train_data_root = './data/train/' # 训练集存放路径
    test_data_root = './data/test1' # 测试集存放路径
    load_model_path = 'checkpoints/model.pth' # 加载预训练的模型

    batch_size = 128 # batch size
    use_gpu = True # user GPU or not
    num_workers = 4 # how many workers for loading data
    print_freq = 20 # print info every N batch

    debug_file = '/tmp/debug'
    result_file = 'result.csv' # 结果文件

    max_epoch = 10
    lr = 0.1 # initial learning rate
    lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4 # 损失函数
```