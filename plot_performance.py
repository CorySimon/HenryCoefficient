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
#df_CUDA = pd.read_csv("GPU_performance.csv")
#df_CUDA['sec'] = df_CUDA['time'].map(get_seconds)
#df_CUDA['ninsertions'] = df_CUDA['GPUkernelcalls'] * 256 * 64

# OpenMP
df_OpenMP = pd.read_csv("OpenMP_performance.csv")
df_OpenMP['sec'] = df_OpenMP['time'].map(get_seconds)
df_OpenMP['ninsertions'] = df_OpenMP['EquivalentGPUkernelcalls'] * 256 * 64

# Plot
fig = plt.figure()

plt.xlabel("Monte Carlo insertions (thousands)")
plt.ylabel("Insertions per run time (10000/sec)")

plt.plot(df_OpenMP["ninsertions"] / 1000.0, df_OpenMP["ninsertions"] / df_OpenMP["sec"] / 10000.0, marker='s', 
        color='b', markersize=10, clip_on=False, label='OpenMP (72 OpenMP threads)')
#plt.plot(df_CUDA["ninsertions"] / 1000.0, df_CUDA["ninsertions"] / df_CUDA["sec"] / 10000.0, marker='o', 
#        color='g', markersize=10, clip_on=False, label='CUDA (64 blocks, 256 threads)')

plt.xlim([0, 9000])
#plt.ylim(ymin=0)
plt.ylim([0, 1500])

plt.legend(loc='center')

plt.tight_layout()
plt.savefig('Performance.png', format='png', dpi=300)

plt.show()
