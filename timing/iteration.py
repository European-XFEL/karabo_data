from time import monotonic
from karabo_data import RunDirectory

print("Opening raw run...")
start = monotonic()
run = RunDirectory('/gpfs/exfel/exp/SPB/201830/p900022/raw/r0034')
delta = monotonic() - start
print(len(run.files), "files")
print(delta, "seconds")

print()
print("Iterating over 600 trains, only image.data ...")
start = monotonic()
train_iter = run.trains(devices=[('*/DET/*', 'image.data')], require_all=True)
for i, (tid, data) in zip(range(1000), train_iter):
    assert data['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf']['image.data'].shape[0] == 64
arr = data['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf']['image.data']
print(arr.nbytes * 16 / 1024, "KiB per train")  # * 16 detector modules
delta = monotonic() - start
print(delta, "seconds")

print()
print("Opening proc run...")
start = monotonic()
run = RunDirectory('/gpfs/exfel/exp/SPB/201830/p900022/proc/r0034')
delta = monotonic() - start
print(len(run.files), "files")
print(delta, "seconds")

print()
print("Iterating over 600 trains, only image.data ...")
start = monotonic()
train_iter = run.trains(devices=[('*/DET/*', 'image.data')], require_all=True)
for i, (tid, data) in zip(range(1000), train_iter):
    assert data['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf']['image.data'].shape[0] == 64
arr = data['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf']['image.data']
print(arr.nbytes * 16 / 1024, "KiB per train")
delta = monotonic() - start
print(delta, "seconds")
