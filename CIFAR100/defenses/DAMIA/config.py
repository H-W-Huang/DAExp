CFG = {
    'data_path': './data/Original_images/',
    'kwargs': {'num_workers': 2},
    'batch_size': 64,
    'epoch': 150,
    'lr': 1e-4,
    'momentum': .9,
    'log_interval': 10,
    'save_interval': 10,
    'l2_decay': 0,
    'lambda': 10,
    'backbone': 'alexnet',
    'n_class': 10
}
