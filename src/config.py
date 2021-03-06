ckpt_dir = '../ckpts'
data_dir = '/WorkSpace/data'
tb_dir = '../runs'
download = True

num_trains = {
    'mnist': 60000,
    'cifar': 50000,
    'fmnist': 60000,
}

num_tests = {
    'mnist': 10000,
    'cifar': 10000,
    'fmnist': 10000,
}

input_sizes = {
    'mnist': 28*28,
    'cifar': 3*32*32,
    'fmnist': 28*28,
}

output_sizes = {
    'mnist': 10,
    'cifar': 10,
    'fmnist': 10,
}

num_channels = {
    'mnist': 1,
    'cifar': 3,
    'fmnist': 1,
}
