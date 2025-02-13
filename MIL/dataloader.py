import monai.data as data
import monai.transforms as transforms
import torch
import random
import numpy as np
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    # print(worker_seed)
    #set_determinism(worker_seed, use_deterministic_algorithms=True)
    worker_info = torch.utils.data.get_worker_info()
    data.utils.set_rnd(worker_info.dataset, seed=worker_seed)  # type: ignore[union-attr]
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def create_dataloader(df, my_transform,shuffle, args,g):
    pw = args.num_workers != 0  # For persistent workers to be enabled, we need at least one worker.
    images = [
        {'tumor_id' : tumor_id, 'image': path, 'segmentation': seg_path, 'label': lbl, 'm_path': seg_path, 'i_path': img_path} for
        # m_path is for debug purposes
        tumor_id, path, seg_path, img_path, lbl in zip(df['tumor_id'],df['image_path'], df['annot_path'], df['image_path'], df['label'])]
    my_set = data.Dataset(
        data=images,
        transform=my_transform,
    )
    # train_sampler = data.DistributedSampler(train_set, shuffle=True)
    data_loader = data.DataLoader(
        my_set,
        collate_fn=torch.utils.data._utils.collate.default_collate,
        # sampler=train_sampler,
        shuffle=shuffle,
        batch_size=1,
        num_workers=args.num_workers,
        persistent_workers=pw,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    return data_loader