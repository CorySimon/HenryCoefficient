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

# GPU
df = pd.read_csv("GPU_performance.csv")
df['sec'] = df['time'].map(get_seconds)

fig = plt.figure()
plt.xlabel("Number of blocks")
plt.ylabel("Time (sec)")
plt.plot(df["NUMBLOCKS"], df["sec"], marker='s')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.show()

# OpenMP
df = pd.read_csv("OpenMP_performance.csv")
df['sec'] = df['time'].map(get_seconds)
print(df)

fig = plt.figure()
plt.xlabel("OMP_NUM_THREADS")
plt.ylabel("Time (sec)")
plt.plot(df["OMP_NUM_THREADS"], df["sec"], marker='s')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.show()
