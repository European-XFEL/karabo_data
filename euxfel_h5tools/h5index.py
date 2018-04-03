import csv
import h5py
import sys

def hdf5_datasets(grp):
    """Print CSV data of all datasets in an HDF5 file.

    path, shape, dtype
    """
    writer = csv.writer(sys.stdout)
    writer.writerow(['path', 'shape', 'dtype'])

    def visitor(path, item):
        if isinstance(item, h5py.Dataset):
            writer.writerow([path, item.shape, item.dtype.str])
    grp.visititems(visitor)

def main():
    file = h5py.File(sys.argv[1])
    hdf5_datasets(file)

if __name__ == '__main__':
    main()
