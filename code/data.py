import jax
import numpy as np
import gzip
import hashlib
import h5py
import os
import random
import shutil
import torch
import urllib.request
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlretrieve
from torch.utils.data import DataLoader, random_split


def get_audio_dataset(cache_dir, cache_subdir, dataset_name):
    # The remote directory with the data files
    base_url = 'https://zenkelab.org/datasets'

    # Retrieve MD5 hashes from remote
    response = urllib.request.urlopen(f'{base_url}/md5sums.txt')
    data = response.read()
    lines = data.decode('utf-8').split("\n")
    file_hashes = {line.split()[1]: line.split()[0]
                   for line in lines if len(line.split()) == 2}

    # Download the Spiking Heidelberg Digits (SHD) dataset
    if dataset_name == 'shd':
        files = ['shd_train.h5.gz', 'shd_test.h5.gz']
    if dataset_name == 'ssc':
        files = ['ssc_train.h5.gz', 'ssc_test.h5.gz']
    if dataset_name == 'all':
        files = ['shd_train.h5.gz', 'shd_test.h5.gz',
                 'ssc_train.h5.gz', 'ssc_test.h5.gz']

    for fn in files:
        origin = f'{base_url}/{fn}'
        hdf5_file_path = get_and_gunzip(origin, fn, md5hash=file_hashes[fn],
                                        cache_dir=cache_dir,
                                        cache_subdir=cache_subdir)
        # print(f"File {fn} decompressed to:")
        print(f' - available at: {hdf5_file_path}')

def get_and_gunzip(origin, filename, md5hash=None, cache_dir=None,
                   cache_subdir=None):
    gz_file_path = get_file(filename, origin, md5_hash=md5hash,
                            cache_dir=cache_dir, cache_subdir=cache_subdir)
    hdf5_file_path = gz_file_path[:-3]
    if not os.path.isfile(hdf5_file_path) or \
            os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path):
        print(f'Decompressing {gz_file_path}')
        with gzip.open(gz_file_path, 'r') as f_in, \
                open(hdf5_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return hdf5_file_path

def get_numpy_datasets(dataset_name, n_inp,
                       cache_dir, download=True, timestep=0.014,
                       truncation=False):
    cache_subdir = f'../datasets/audiospikes_{n_inp}'
    if download:
        get_audio_dataset(cache_dir, cache_subdir, dataset_name)

    train_ds = []; test_ds = []
    if dataset_name in ['shd', 'all']:
        train_shd_file = h5py.File(os.path.join(cache_dir, cache_subdir,
                                                'shd_train.h5'
                                                ), 'r')
        test_shd_file = h5py.File(os.path.join(cache_dir, cache_subdir,
                                               'shd_test.h5'
                                               ), 'r')
        shd_train_ds = DatasetNumpy(train_shd_file['spikes'],
                                    train_shd_file['labels'],
                                    name='shd', target_dim=n_inp, nb_rep=1,
                                    timestep=timestep,
                                    truncation=truncation)
        shd_test_ds = DatasetNumpy(test_shd_file['spikes'],
                                   test_shd_file['labels'],
                                   name='shd', target_dim=n_inp, nb_rep=1,
                                   timestep=timestep, truncation=truncation)
        train_ds.append(shd_train_ds)
        test_ds.append(shd_test_ds)

    # if dataset_name in ['ssc', 'all']:
    #     train_ssc_file = h5py.File(os.path.join(cache_dir, cache_subdir,
    #                                             'ssc_train.h5'
    #                                             ), 'r')
    #     test_ssc_file  = h5py.File(os.path.join(cache_dir, cache_subdir,
    #                                             'ssc_test.h5'
    #                                             ), 'r')
    #     ssc_train_ds = DatasetNumpy(train_ssc_file['spikes'],
    #                                 train_ssc_file['labels'],
    #                                 name='ssc', target_dim=n_inp)
    #     ssc_test_ds  = DatasetNumpy(test_ssc_file['spikes'],
    #                                 test_ssc_file['labels'],
    #                                 name='ssc', target_dim=n_inp)
    #     train_ds.append(ssc_train_ds)
    #     test_ds.append(ssc_test_ds)

    return train_ds[0], test_ds[0]

class DatasetNumpy(torch.utils.data.Dataset):
    """
    Numpy based generator
    """
    def __init__(self, spikes, labels, name, target_dim, nb_rep, timestep,
                 truncation=False):
        self.nb_steps = int(1.4/timestep)   # number of time steps in the input
        self.nb_units = 700   # number of input units (channels)
        self.max_time = 1.4   # maximum recording time of a digit (in s)
        self.spikes = spikes  # recover the 'spikes' dictionary from h5 file
        self.labels = labels  # recover the 'labels' array from h5 file
        self.name = name      # name of the dataset or name of speaker

        self.firing_times = self.spikes['times']
        self.units_fired = self.spikes['units']
        self.num_samples = self.firing_times.shape[0]
        self.time_bins = np.linspace(0, self.max_time, num=self.nb_steps)

        # initialize the input (3D) and output (1D) arrays
        self.input = np.zeros((self.num_samples, self.nb_steps,
                              self.nb_units), dtype=np.uint8)
        self.output = np.array(self.labels, dtype=np.uint8)

        self.load_spikes()
        self.reduce_inp_dimensions(target_dim=target_dim, axis=2,
                                   nb_rep=nb_rep)

        if truncation:
            self.input = self.input[:, :150, :]

        self.num_samples = self.input.shape[0]

    def __len__(self):
        return self.num_samples

    def load_spikes(self):
        """
        For each sample, we create a 2D array of size (nb_steps, nb_units).
        We downsample the firing times and the units fired to the time bins
        :return:
        """
        for idx in range(self.num_samples):
            times = np.digitize(self.firing_times[idx], self.time_bins)
            units = self.units_fired[idx]
            self.input[idx, times, units] = 1

    def reduce_inp_dimensions(self, target_dim, axis, nb_rep):
        sample_ind = int(np.ceil(self.nb_units / target_dim))
        assert nb_rep <= sample_ind, \
            (f'The maximum factor of data augmentation is {sample_ind}, '
             f'you provided {nb_rep}')
        index = [np.arange(i, 700, sample_ind) for i in range(sample_ind)]
        reshaped = [np.take(self.input, index[i], axis)
                    for i in range(nb_rep)] # this samples the data a
        reshaped = [np.pad(reshaped[i],
                            [(0, 0), (0, 0),
                             (0, int(target_dim-reshaped[i].shape[2]))],
                            mode='constant')
                    for i in range(nb_rep)]
        reshaped = np.concatenate(reshaped, axis=0)

        self.input = reshaped
        self.output = np.tile(self.output, nb_rep)
        self.num_samples = reshaped.shape[0]

    def __getitem__(self, idx):
        inputs, outputs = self.__data_generation(idx)
        return inputs, outputs

    def __data_generation(self, idx):
        if self.name == 'shd':
            output = self.output[idx]
        # if self.name == 'ssc':
        #     output = self.output[idx] + 20
        return self.input[idx], output


def get_file(fname,
             origin,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.data-cache')
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.data-cache')
    datadir = os.path.join(datadir_base, cache_subdir)

    # Create directories if they don't exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(datadir, exist_ok=True)

    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
    # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated because the ' + hash_algorithm +
                      ' file hash does not match the original value of '
                      + file_hash + ' so we will re-download the data.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno,
                                                 e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)

    return fpath


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    if (algorithm == 'sha256') or \
            (algorithm == 'auto' and len(file_hash) == 64):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def custom_collate_fn(batch):
    transposed_data = list(zip(*batch))

    labels = np.array(transposed_data[1])
    spikes = np.array(transposed_data[0])

    return spikes, labels


def get_data_loaders(sim_params):
    cache_dir = os.getcwd()
    print(f'datasets:')
    train_ds, test_ds = get_numpy_datasets(
        'shd', sim_params.n_in, cache_dir=cache_dir, download=True,
        timestep=sim_params.timestep, truncation=sim_params.truncation)

    # Set random seeds for reproducibility
    torch.manual_seed(sim_params.seed)
    np.random.seed(sim_params.seed)
    random.seed(sim_params.seed)

    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds_split, val_ds_split = random_split(train_ds,
                                                [train_size, val_size])

    train_loader = DataLoader(train_ds_split, sim_params.batch_size,
                              shuffle=True, collate_fn=custom_collate_fn,
                              drop_last=True)
    val_loader = DataLoader(val_ds_split, sim_params.batch_size,
                            shuffle=True, collate_fn=custom_collate_fn,
                            drop_last=True)
    test_loader = DataLoader(test_ds, sim_params.batch_size,
                             shuffle=None, collate_fn=custom_collate_fn,
                             drop_last=True)
    return train_loader, val_loader, test_loader