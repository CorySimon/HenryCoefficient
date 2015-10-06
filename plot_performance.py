import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
plt.style.use('bmh')
matplotlib.rc('axes', facecolor='w') # Or any suitable colour...
matplotlib.rc('lines',linewidth=3)
matplotlib.rc('font',size=16)

def get_seconds(time_string):
    """
    Convert e.g. 1m5.928s to seconds
    """
    minutes = float(time_string.split("m")[0])
    seconds = float(time_string.split("m")[1].split("s")[0])
    return minutes * 60.0 + seconds

ninsertions = 100000.0 * 256.0

# GPU
df = pd.read_csv("GPU_performance.csv")
df['sec'] = df['time'].map(get_seconds)

fig = plt.figure()
plt.xlabel("Number of blocks")
plt.ylabel("Insertions per run time (1000/sec)")
plt.plot(df["NUMBLOCKS"], ninsertions / df["sec"] / 1000.0, marker='s', color='g', markersize=10, clip_on=False)
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.tight_layout()
plt.savefig('GPU_performance.png', format='png', dpi=300)
plt.show()

# OpenMP
df = pd.read_csv("OpenMP_performance.csv")
df['sec'] = df['time'].map(get_seconds)

fig = plt.figure()
plt.xlabel("OMP_NUM_THREADS")
plt.ylabel("Insertions per run time (1000/sec)")
plt.plot(df["OMP_NUM_THREADS"], ninsertions / df["sec"] / 1000.0, marker='s', color='b', markersize=10, clip_on=False)
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.tight_layout()
plt.savefig('OMP_performance.png', format='png', dpi=300)
plt.show()
