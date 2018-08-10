from time import monotonic
from karabo_data import RunDirectory

print("Opening raw run...")
start = monotonic()
run = RunDirectory('/gpfs/exfel/exp/SA1/201830/p900025/raw/r0150/')
delta = monotonic() - start
print(len(run.files), "files")
print(delta, "seconds")

print()
print("Retrieving data frame for XGM ixPos & iyPos...")
start = monotonic()
df = run.get_dataframe(fields=[("*_XGM/*", "*.i[xy]Pos"), ("*_XGM/*", "*.photonFlux")])
delta = monotonic() - start
print(delta, "seconds")
print(df.head())

