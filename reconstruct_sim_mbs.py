import logging
import os
import json
import numpy as np
import init_helper

import ssnp
from pycuda import gpuarray
from tools import rev_ng_acc
from ssnp.utils.fista import get_q
from tv2d_temp import prox_tv_Michael

logging.info('version 1115')

base_dir = "2dres_mbs"

sim_name = 'h06_scale04'
measurements = np.sqrt(ssnp.read(os.path.join(base_dir, f"sim/{sim_name}.tiff"), gpu=False) / 0.4)

measurements = gpuarray.to_gpu(measurements)

with open(os.path.join(base_dir, "exp_bfdf.json")) as f:
    metadata = json.load(f)

parameters = {
    # data parameters
    "slices":     8,
    "z_step":     1,
    "xy_patch":   (1024, 1024),
    "overlap":    0,

    # reconstruction parameters
    "iterations": 50,
    "gamma":      5,
    "tau":        8e-7,
    "model_type": "bpm"
}

illumination = metadata['illumination'][:]
LEDs = len(illumination)
assert LEDs == len(measurements)
N0 = metadata['n_medium']

positive = False
iterations = parameters['iterations']
xy_patch = parameters['xy_patch']
z_step = parameters['z_step']
SLICES = parameters['slices']
GAMMA = parameters['gamma']
TAU = parameters['tau']
u_in = ssnp.read("plane", np.complex128, shape=(LEDs, *xy_patch), gpu=False)
n = np.zeros((SLICES, *xy_patch), dtype=np.double)
n = gpuarray.to_gpu(n)
ng = n.copy()
x_1 = n.copy()

beam = ssnp.BeamArray(u_in[0])
beam.config.lambda0 = metadata['lambda']
beam.config.xyz = (*metadata['xy_size'], z_step)
beam.config.n0 = N0
pupil = beam.multiplier.binary_pupil(metadata['NA'] / N0, gpu=True)
for i in range(LEDs):
    c_ab = [c / N0 for c in illumination[i]]
    u_in[i] = beam.multiplier.tilt(c_ab, trunc=True, gpu=False)

illumination = np.array(illumination)

ng_acc = rev_ng_acc(ng)

for step in range(iterations):
    logging.info(f"{step=} started")

    loss_tot = 0
    for ill_i in range(109):
        ng.fill(0)
        beam.forward = u_in[ill_i]
        with beam.track():
            for i in range(4):
                beam.bpm(0.5, n[i])
                beam.bpm(10)
            # beam.conj()
            beam *= -1
            for i in range(4, 8):
                beam.bpm(10)
                beam.bpm(0.5, n[i])
            beam.bpm(-42)
            beam.a_mul(pupil)
            loss = beam.forward_mse_loss(measurements[ill_i]) * LEDs
            logging.debug(f"{ill_i = }, {loss = :.9f}")
            loss_tot += loss
        beam.tape.collect_gradient({'n': ng_acc}, reverse=True)
        # tape_grad = beam.tape.collect_gradient({'n': ng_acc, 'u1_in': None}, reverse=True)
        # u_in_grad_arr = tape_grad['u1_in'][0]
        # beam.recycle_array(u_in_grad_arr)
        # u_in_grad_arr = u_in_grad_arr.get()
        # u_in_grad = np.real(np.sum(u_in_grad_arr * np.conj(u_in[ill_i])))
        #
        # # Nesterov
        # norm_x[ill_i] = normalization[ill_i] - u_in_grad * (0.3 if ill_i < 25 else 5)
        # normalization[ill_i] = norm_x[ill_i] + (norm_x[ill_i] - norm_x1[ill_i]) * (get_q(step) - 1) / get_q(step + 1)
        # norm_x1[ill_i] = norm_x[ill_i]

        assert next(ng_acc) is None
        step_size = parameters['gamma']

        ng *= step_size
        n -= ng
    x = ng  # only memory assignment
    if positive:
        prox_fgp(n, GAMMA * TAU, out=x)
    else:
        prox_tv_Michael(n, GAMMA * TAU, out=x)

    n.set(x)
    # x.set(n)

    n -= x_1
    n *= (get_q(step) - 1) / get_q(step + 1)
    logging.info((get_q(step) - 1) / get_q(step + 1))
    n += x
    x_1.set(x)
    logging.info(f"Step {step} finished, {loss_tot = :.9f}")


tiff_range = (-0.08, 0.08)
ssnp.write(os.path.join(base_dir, f"rec/h10_BFDF_reg_{N0 + tiff_range[0]:.3f}_{N0 + tiff_range[1]:.3f}.tiff"), n,
           pre_operator=lambda x: (x - tiff_range[0]) / (tiff_range[1] - tiff_range[0]))
# np.save(os.path.join(base_dir, f"norm_hist.npy"), norm_history)
# ssnp.write(os.path.join(base_dir, "rec/5l_45l.npy"), n)

