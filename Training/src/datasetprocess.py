import numpy as np


def cifar10_partition(dataset, num_users, iid=True):
    """
    对cifar10数据集的下标进行划分，有iid(大小均匀)/ noniid(大小不均匀)两种方式
    Sample non-I.I.D client data from CIFAR10 dataset
    :param iid:
    :param dataset:
    :param num_users:
    :return:
    """
    alpha = 0.5
    #  data是图片， targets是标签
    X_train, y_train = dataset.data, np.array(dataset.targets)
    n_train = int(X_train.shape[0])
    net_dataidx_map = {}
    if iid:
        idxs = np.random.permutation(n_train)
        idx_batch = np.array_split(idxs, num_users)
        net_dataidx_map = {i: idx_batch[i] for i in range(num_users)}
    else:
        min_size = 0
        # cifar10的种类数为10
        K = 10
        N = n_train
        idx_batch = []
        # 保证至少有一个batch 256
        while min_size < 256:
            idx_batch = [[] for _ in range(num_users)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_users))
                # Balance
                proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                print('k', k)
                print('idx_batch', [len(idx_j) for idx_j in idx_batch])
                print('min_size', min_size)
        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    return net_dataidx_map
