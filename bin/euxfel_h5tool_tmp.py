#!/usr/bin/python
"""
European XFEL HDF5 files stat
"""

import sys

import h5py
import numpy as np

import euxfel_h5tools

import matplotlib.pyplot as plt




if __name__ == "__main__":
    euxfel_h5tools.stat(sys.argv[1:])

    filename = sys.argv[1]
    # need to split the following in functions that can be activated
    # with switches. Only works with one file at the moment
    print("Overview of structure")
    f = h5py.File(filename, 'r')
    euxfel_h5tools.rec_print_h5_level(f, maxlen=3)

    # identify detector (should be provided by user as argument, maybe, or found automatically.
    # Hard code for SPB_DET_AGIPD1M-1"
    instrument = "INSTRUMENT/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf"
    headerpath = instrument + '/header'
    datapath = instrument + '/image'

    trainId = np.array(f[headerpath + '/trainId'])
    pulseCount = np.array(f[headerpath + '/pulseCount'])

    # assume same number of pulses in each train - need to check!
    print ("We have {} trains with {} pulses each ".format(trainId.shape, pulseCount[0]))
    assert trainId.shape == pulseCount.shape

    # iterate over trains and pulses and  give detailed breakdown of pulses:
    index = 0
    for t_id, p_count in zip(trainId, pulseCount):
        print("Train ID = {}, {} pulses".format(t_id, p_count))

    # iterate over image data and do something (to be added)
    train_ids = f[datapath + '/trainId']
    pulse_ids = f[datapath + '/pulseId']

    for i in range(len(f[datapath + '/data'])):
        data = np.array(f[datapath + '/data'][i])
        print("train = {}, pulse = {}, data shape = {}, data = ...".format(
             train_ids[i], pulse_ids[i], data.shape))
        # image data available in numpy array 'data'

        # for example plot first pulse in every train
        if pulse_ids[i] == 8:
            plt.imshow(data[0,:,:])
            plt.title("trainid={}, pulseid={}".format(train_ids[i], pulse_ids[i]))
            plt.show(block=False)
            plt.pause(1e-3)
