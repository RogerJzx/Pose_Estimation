class DefaultConfig(object):

    num_classes = 2
    env = 'default'
    model = 'DarkNet'
    pretrain = None
    train_data_root = '/workspace/liuzhen/remote_workspace/pose/LINEMOD/self/train1.txt'
    test_data_root = '/workspace/liuzhen/remote_workspace/pose/LINEMOD/self/test1.txt'
    load_model_path = None

    batch_size = 16  # batch size
    use_gpu = True  # use GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = 'Result/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
    bg_images = '/workspace/Data/VOCdevkit/VOC2012/JPEGImages'
    steps = [-1,  500, 8000, 12000, 21000, 24000, 25000, 290000]
    scales = [0.1, 1,    0.1,  10,   .01,     100,  0.1,  .001]
    max_epoch = 10
    lr = 0.001  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4
    momentum = .9
    save_name = 'checkpoints/best.pth'
    def parse(self, kwargs):
        import warnings
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)
        print('user config: ')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
