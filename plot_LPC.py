import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from lpctorch import LPCCoefficients
from librosa.core import lpc

# Load audio file
sr = 16000  # 16 kHz
path = 'speech_samples/ByASingular.wav'
data, _sr = torchaudio.load(path, normalization=lambda x: x.abs().max())
data = torchaudio.transforms.Resample(_sr, sr)(data)
duration = data.size(1) / sr

# Get audio sample worth of 512 ms
worth_duration = .512 * 5  # 512 ms ( 256 ms before and 256 ms after )
worth_size = int(np.floor(worth_duration * sr))
X = data[:, :worth_size]
X_duration = X.size(1) / sr
X = torch.cat([X for i in range(4)])

# ====================== ME ====================================================
# Divide in 64 2x overlapping frames
frame_duration = .016  # 16 ms
frame_overlap = .5
K = 26
lpc_prep = LPCCoefficients(
    sr,
    frame_duration,
    frame_overlap,
    order=(K - 1)
).eval().cuda()
alphas = lpc_prep(X.cuda()).detach().cpu().numpy()

# Print details
print(f'[Init]   [Audio]  src: {path}, sr: {sr}, duration: {duration}')
print(f'[Init]   [Sample] size: {X.shape}, duration: {X_duration}')
print(f'[Me]     [Alphas] size: {alphas.shape}')


# ====================== NOT ME ================================================
def librosa_lpc(X, order):
    try:
        return lpc(X, order)
    except:
        res = np.zeros((order + 1,))
        res[0] = 1.
        return res


frames = lpc_prep.frames(X.cuda())
frames = frames[0].detach().cpu().numpy()
_alphas = np.array([librosa_lpc(frames[i], K - 1) for i in range(frames.shape[0])])
print(f'[Not Me] [Alphas] size: {_alphas.shape}')

print(f'Error [Me] vs [Not Me]: {(alphas[0] - _alphas).sum(axis=-1).mean()}')

# Draw frames
fig = plt.figure()
ax = fig.add_subplot(211)
ax.imshow(np.transpose(alphas[0]))
ax = fig.add_subplot(212)
ax.imshow(np.transpose(_alphas))
fig.canvas.draw()
plt.show()

# draw individual plots
SMALL_SIZE = 24
MEDIUM_SIZE = 26
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.figure(figsize=(16, 5))
y_vector = np.linspace(0, _alphas.shape[1]-1, _alphas.shape[1])
time_vector = np.linspace(0, worth_duration, len(_alphas))
time_mesh,y_mesh = np.meshgrid(time_vector,y_vector)
plt.pcolormesh(time_mesh,y_mesh,np.transpose(_alphas), cmap='Greys')
plt.colorbar()
plt.xlabel('Time (sec)')
plt.tight_layout()
plt.show()

