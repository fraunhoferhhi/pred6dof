# '''
# The copyright in this software is being made available under this Software
# Copyright License. This software may be subject to other third party and
# contributor rights, including patent rights, and no such rights are
# granted under this license.
# Copyright (c) 1995 - 2021 Fraunhofer-Gesellschaft zur Förderung der
# angewandten Forschung e.V. (Fraunhofer)
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted for purpose of testing the functionalities of
# this software provided that the following conditions are met:
# *     Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# *     Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# *     Neither the names of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT CLAIMS, INCLUDING
# WITHOUT LIMITATION THE PATENTS OF THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS, ARE GRANTED BY THIS SOFTWARE LICENSE. THE
# COPYRIGHT HOLDERS AND CONTRIBUTORS PROVIDE NO WARRANTY OF PATENT
# NON-INFRINGEMENT WITH RESPECT TO THIS SOFTWARE.
# '''

import pickle
import os
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
pd.options.mode.chained_assignment = None

def preprocess_trace(trace_path, dt, out_dir):
    """
    Resample and interpolate a raw Hololens trace (which contains unevenly sampled data) with a given sampling frequency (e.g. 5ms) and write the output to a csv file.
    Arguments:
        trace_path: Path to the raw Hololens trace.
        dt: Desired time distance between two samples [s]
        out_dir: Output directory containing the interpolated traces.
    Outputs:
        df_intp: Dataframe containing interpolated position (x,y,z) and rotation values (quaternions and Euler angles with equal spacing.
    """

    case = os.path.splitext(os.path.basename(trace_path))[0]
    df = pd.read_csv(trace_path, skipfooter=1, engine='python')
    df['timestamp'] *= 100
    # Start the timestamp from 0
    df['timestamp'] -= df['timestamp'].iloc[0]
    df = df.astype(float)
    qs = df.loc[:, ['timestamp', 'qx', 'qy', 'qz', 'qw']].to_numpy()

    # Resample and interpolate the position samples (x,y,z) onto a uniform grid
    df_t = df.loc[:, 'timestamp':'z']
    df_t['timestamp'] = pd.to_timedelta(df_t['timestamp'], unit='ns')
    df_t_intp = df_t.resample(str(dt*1e3) + 'L', on='timestamp').mean().interpolate('linear')
    t_intp = df_t_intp.to_numpy()

    # Resample and interpolate the quaternion samples
    rots = R.from_quat(qs[:, 1:])
    times = qs[:, 0]
    slerp = Slerp(times, rots)  # Spherical Linear Interpolation of Rotations (SLERP)
    rots_intp = slerp(times)
    t = df_t_intp.index.to_numpy().astype(float)
    rots_intp = slerp(t)
    q_intp = rots_intp.as_quat()

    # Compute Euler angles for the interpolated quaternion samples
    e_intp = rots_intp.as_euler('ZXY', degrees=True)

    # Combine the interpolated array and create a DataFrame
    intp = np.hstack((t[:, np.newaxis], t_intp, q_intp, e_intp))
    df_intp = pd.DataFrame(intp, columns=np.hstack((df.columns, ['roll', 'pitch', 'yaw'])))

    # Save interpolated DataFrame to csv
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df_intp = df_intp.iloc[:12001]   # Make length of all traces the same.
    df_intp.to_csv(os.path.join(out_dir, case + '.csv'), index=False)
    
    return df_intp


def get_csv_files(dataset_path):
    """
    Generator function to recursively output the CSV files in a directory and its sub-directories.
    Arguments:
        dataset_path: Path to the directory containing the CSV files.
    Outputs:
        Paths of the found CSV files.
    """
    numerical_files = []
    numerical_files_sorted = []
    for f in os.listdir(dataset_path):
        if not os.path.isdir(f):
            file_name, extension = f.split('.')
            if extension == "csv" and file_name.isnumeric():
                numerical_files.append(file_name)
            else:
                logging.warn("Invalid file: {}. Ignoring...".format(f))

    numerical_filenames_ints = [int(f) for f in numerical_files]
    numerical_filenames_ints.sort()

    for f in numerical_filenames_ints:
        file = str(f) + ".csv"
        numerical_files_sorted.append(os.path.join(dataset_path, file))

    return numerical_files_sorted