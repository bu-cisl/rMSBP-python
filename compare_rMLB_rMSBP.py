import os
import json

import numpy as np
import matplotlib.pyplot as plt
from pycuda import gpuarray

import ssnp
import ssnp.calc
import ssnp.funcs

base_dir = "2dres_mbs"

sim_name = 'h10_scale04'
measurements = ssnp.read(os.path.join(base_dir, f"sim/{sim_name}.tiff"), gpu=False) * 2.61
LAYER_THICKNESS = 0.3
oversample = 3

with open(os.path.join(base_dir, "exp_bfdf.json")) as f:
    metadata = json.load(f)
    N0 = metadata['n_medium']

slice0 = ssnp.read(os.path.join(base_dir, "USAF-1951_68.5nm_1024p.tiff"), gpu=False)
slice0 *= -0.07
slice0 = slice0 + np.zeros([oversample, 1, 1])
slices = [np.rot90(slice0, i, (1, 2)) for i in range(4)]
slices = [gpuarray.to_gpu(np.ascontiguousarray(si)) for si in slices]

beam = ssnp.BeamArray(np.zeros_like(slice0[0], dtype=np.complex128))
beam.config.lambda0 = metadata['lambda']
beam.config.xyz = (*metadata['xy_size'], 1)
beam.config.n0 = N0

pupil = beam.multiplier.binary_pupil(metadata['NA'] / N0, gpu=True)

bpm_angles = []
mlb_angles = []

mlb_field = beam.array_pool.get()
mlb_tmp_arr = beam.array_pool.get()

illumination = metadata['illumination'][:]
LEDs = len(illumination)
assert LEDs == len(measurements)
N0 = metadata['n_medium']
u_in = ssnp.read("plane", np.complex128, shape=(LEDs, *slice0[0].shape), gpu=False)

gap = 10.
for i in range(109):
    c_ab = [c / N0 for c in illumination[i]]
    u_in = beam.multiplier.tilt(c_ab, trunc=True, gpu=False)
    beam.forward = u_in
    for si in slices:
        beam.bpm(LAYER_THICKNESS / len(si), si)
        beam.bpm(gap)
    beam *= -1
    for si in reversed(slices):
        beam.bpm(gap)
        beam.bpm(LAYER_THICKNESS / len(si), si)
    beam.bpm(-4 * (gap + LAYER_THICKNESS))
    beam.a_mul(pupil)
    bpm_angles.append(np.abs(beam.forward.get()) ** 2)

    mlb_field.set(u_in)
    for si in slices:
        for sii in si:
            ssnp.calc.mlb_step(mlb_field, mlb_tmp_arr, LAYER_THICKNESS / len(si), sii, beam.config)
        ssnp.calc.mlb_step(mlb_field, None, gap, None, beam.config)
    mlb_field *= -1
    for si in reversed(slices):
        ssnp.calc.mlb_step(mlb_field, None, gap, None, beam.config)
        for sii in si:
            ssnp.calc.mlb_step(mlb_field, mlb_tmp_arr, LAYER_THICKNESS / len(si), sii, beam.config)
    ssnp.calc.mlb_step(mlb_field, None, -4 * (gap + LAYER_THICKNESS), None, beam.config)
    with beam._fft_funcs.fourier(mlb_field) as mlb_freq:
        mlb_freq *= pupil
    mlb_angles.append(np.abs(mlb_field.get()) ** 2)

plt.imshow(bpm_angles[2])
plt.clim([0.5, 1.5])
plt.colorbar()
plt.show()

plt.imshow(mlb_angles[2])
plt.clim([0.5, 1.5])
plt.colorbar()
plt.show()

plt.imshow(measurements[2])
plt.clim([0.5, 1.5])
plt.colorbar()
plt.show()

ssnp.write(os.path.join(base_dir, f"sim/{sim_name}_ovrsmp{oversample}_bpm.tiff"), np.stack(bpm_angles) * 0.5)
ssnp.write(os.path.join(base_dir, f"sim/{sim_name}_ovrsmp{oversample}_mlb.tiff"), np.stack(mlb_angles) * 0.5)

diff_stat = []
for gti, bi, mi in zip(measurements, bpm_angles, mlb_angles):
    diff_stat.append([np.mean(np.abs(arr) ** 2) for arr in [gti - bi, gti - mi]])

with open(os.path.join(base_dir, f'sim/{oversample}step_diff_bpm_mlb.json'), 'w') as f:
    json.dump(diff_stat, f)

