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

import json
import os
import re
import pandas as pd
import matplotlib as mpl
import numpy as np
import toml
import logging
import glob
import shutil
from distutils.dir_util import copy_tree
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.signal import savgol_filter
from .utils import get_csv_files

style_path = os.path.join(os.getcwd(), 'pred6dof/style.json')
style = json.load(open(style_path))

# Fixes type 1 fonts issue
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True

config_path = os.path.join(os.getcwd(), 'config.toml')
cfg = toml.load(config_path)

class Reporter():
    """Computes and plots trace statistics, per-trace results and average results"""
    @staticmethod
    def plot_trace(trace_path, figures_path):
        ts = np.arange(0, 60 + cfg['dt'], cfg['dt'])
        df = pd.read_csv(trace_path)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
        
        # Plot position
        ax1.plot(ts, df.loc[:len(ts)-1, 'x'], label='x')
        ax1.plot(ts, df.loc[:len(ts)-1, 'y'], label='y', linestyle='--')
        ax1.plot(ts, df.loc[:len(ts)-1, 'z'], label='z', linestyle='-.')
        ax1.set_ylabel('meters')
        ax1.set_xlim(0, 60)
        ax1.legend(loc='upper left')
        ax1.yaxis.grid(which='major', linestyle='dotted', linewidth=1)
        ax1.xaxis.set_major_locator(MultipleLocator(10))
        
        # Plot orientation
        ax2.plot(ts, df.loc[:len(ts)-1, 'yaw'], label='yaw')
        ax2.plot(ts, df.loc[:len(ts)-1, 'pitch'], label='pitch', linestyle='--')
        ax2.plot(ts, df.loc[:len(ts)-1, 'roll'], label='roll', linestyle='-.')
        ax2.set_xlabel('seconds')
        ax2.set_ylabel('degrees')
        ax2.set_xlim(0, 60)
        ax2.legend(loc='upper left')
        ax2.yaxis.grid(which='major', linestyle='dotted', linewidth=1)
        ax2.xaxis.set_major_locator(MultipleLocator(10))
        
        trace_id = os.path.splitext(os.path.basename(trace_path))[0]
        dest = os.path.join(figures_path, "Fig5-trace{}.pdf".format(trace_id))
        fig.savefig(dest)
        logging.info("Plotting trace {} and saving to file {}".format(trace_path, dest))
  
    @staticmethod
    def plot_head_velocity(dataset_path, figures_path):
        params = {'figure.dpi' : 300,
          'legend.fontsize': 12,
          'legend.handlelength': 2,
          'axes.titlesize': 12,
          'axes.labelsize': 12,
          'xtick.labelsize' : 12,
          'ytick.labelsize' : 12
          }
        plt.rcParams.update(params) 
        
        # Figure 6: CDF of linear velocity (left) and angular velocity (right) for trace 1.
        avg_vs_pos, std_vs_pos, max_vs_pos  = [], [], []
        avg_vs_ang, std_vs_ang, max_vs_ang = [], [], []
        pos_95s, ang_95s = [], []
        
        for i, trace_path in enumerate(get_csv_files(dataset_path)):
            trace = os.path.splitext(os.path.basename(trace_path))[0]
            trace_name = os.path.splitext(os.path.basename(trace_path))[0]

            df = pd.read_csv(trace_path)
            coords = ["x", "y", "z", "yaw", "pitch", "roll"]
            zs = df[coords].to_numpy()
            # Calculate positional speed directly using Savitzky-Golay filter
            zs_pos = zs[:, :3]
            v_pos = savgol_filter(zs_pos, window_length=29, polyorder=4, deriv=1, delta=0.005, axis=0)

            # For angular speed, deal with the jumps in Euler angles first
            zs_ang = zs[:, 3:]
            zs_ang_diff = np.diff(zs_ang, 1, axis=0)
            zs_ang_diff[np.abs(zs_ang_diff) > 180] = 0
            v_ang = savgol_filter(zs_ang_diff, window_length=29, polyorder=4, axis=0)
            v_ang /= 0.005

            vs_pos_sorted = np.sort(np.abs(v_pos), axis=0)
            vs_ang_sorted = np.sort(np.abs(v_ang), axis=0)

            avg_v_pos = np.mean(np.abs(v_pos), axis=0)
            std_v_pos = np.std(np.abs(v_pos), axis=0)
            v_pos95 = np.percentile(vs_pos_sorted, 95, axis=0)
            max_v_pos = np.max(np.abs(v_pos), axis=0)
            avg_vs_pos.append(avg_v_pos)
            std_vs_pos.append(std_v_pos)
            pos_95s.append(v_pos95)

            avg_v_ang = np.mean(np.abs(v_ang), axis=0)
            std_v_ang = np.std(np.abs(v_ang), axis=0)
            v_ang95 = np.percentile(vs_ang_sorted, 95, axis=0)
            max_v_ang = np.max(np.abs(v_ang), axis=0)
            avg_vs_ang.append(avg_v_ang)
            std_vs_ang.append(std_v_ang)
            ang_95s.append(v_ang95)

            logging.debug("Average linear velocity [m/s] {}".format(avg_v_pos))
            logging.debug("Stdev linear velocity [m/s] {}".format(std_v_pos))
            logging.debug("Max. linear velocity [m/s] {}".format(max_v_pos))
            logging.debug("Average angular velocity[deg/s] {}".format(avg_v_ang))
            logging.debug("Stdev angular velocity[deg/s] {}".format(std_v_ang))
            logging.debug("Max. angular velocity [deg/s] {}".format(max_v_ang))
            logging.debug("Position 95h percentile: {}".format(np.percentile(vs_pos_sorted, 95, axis=0)))
            logging.debug("Angular 95h percentile: {}".format(np.percentile(vs_ang_sorted, 95, axis=0)))

            # Plot CDF just for Trace 1 - Figure 6
            if i==0: 
                # CDF for linear velocity
                p = np.linspace(0, 1, len(v_pos))
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
                ax1.plot(vs_pos_sorted[:, 0], p, label='x')
                ax1.plot(vs_pos_sorted[:, 1], p, label='y', linestyle='--')
                ax1.plot(vs_pos_sorted[:, 2], p, label='z', linestyle='-.')
                ax1.set_xlabel('m/s')
                ax1.set_ylabel('CDF')
                ax1.set_xlim(0, np.max(vs_pos_sorted[-1]))
                ax1.set_ylim(0, 1)
                ax1.spines['right'].set_visible(False)
                ax1.spines['top'].set_visible(False)
                ax1.xaxis.grid(which='major', linestyle='dotted', linewidth=1)
                ax1.yaxis.grid(which='major', linestyle='dotted', linewidth=1)
                ax1.legend(loc='lower right')

                # CDF for angular velocity
                q = np.linspace(0, 1, len(v_ang))
                ax2.plot(vs_ang_sorted[:, 0], q, label='yaw')
                ax2.plot(vs_ang_sorted[:, 1], q, label='pitch', linestyle='--')
                ax2.plot(vs_ang_sorted[:, 2], q, label='roll', linestyle='-.')
                ax2.set_xlabel('deg/s')
                ax2.set_xlim(0, np.max(vs_ang_sorted[-1]))
                ax2.set_ylim(0, 1)
                ax2.spines['right'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                ax2.legend(loc='lower right')
                ax2.xaxis.grid(which='major', linestyle='dotted', linewidth=1)
                ax2.yaxis.grid(which='major', linestyle='dotted', linewidth=1)

                dest = 'Fig6-trace{}_cdf_velocity.pdf'.format(trace_name)
                fig.savefig(os.path.join(figures_path, dest))
                logging.info("Saving velocity CDF plots to {}".format(dest))
        
        # Figure 7: 
        # Mean linear velocity (left) and mean angular velocity (right) for five traces. 
        # Lighter shades show the 95th percentile.
        avg_vs_pos = np.array(avg_vs_pos)
        std_vs_pos = np.array(std_vs_pos)
        avg_vs_ang = np.array(avg_vs_ang)
        std_vs_ang = np.array(std_vs_ang)
        pos_95s, ang_95s = np.array(pos_95s), np.array(ang_95s)

        # Bar plots to show avg/max/stdev velocity across traces
        barWidth = 0.25
        N = 5
        r1 = np.arange(N)
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.bar(r1, avg_vs_pos[:N, 0], color='#1f77b4', width=barWidth, label='x')
        ax1.bar(r1, pos_95s[:N, 0] - avg_vs_pos[:N, 0], bottom=avg_vs_pos[:N, 0], color='#1f77b4', alpha=0.3, width=barWidth)
        ax1.bar(r2, avg_vs_pos[:N, 1], color='#ff7f0e', width=barWidth, label='y')
        ax1.bar(r2, pos_95s[:N, 1] - avg_vs_pos[:N, 1], bottom=avg_vs_pos[:N, 1], color='#ff7f0e', alpha=0.3, width=barWidth)
        ax1.bar(r3, avg_vs_pos[:N, 2], color='#2ca02c', width=barWidth, label='z')
        ax1.bar(r3, pos_95s[:N, 2] - avg_vs_pos[:N, 2], bottom=avg_vs_pos[:N, 2], color='#2ca02c', alpha=0.3, width=barWidth)
        ax1.set_xlabel('Trace')
        ax1.set_ylabel('m/s')
        ax1.set_xticks([r + barWidth for r in range(N)])
        ax1.set_xticklabels(['1', '2', '3', '4', '5'])
        ax1.legend()

        ax2.bar(r1, avg_vs_ang[:N, 0], color='#1f77b4', width=barWidth, label='yaw',)
        ax2.bar(r1, ang_95s[:N, 0] - avg_vs_ang[:N, 0], bottom=avg_vs_ang[:N, 0], color='#1f77b4', alpha=0.3, width=barWidth)
        ax2.bar(r2, avg_vs_ang[:N, 1], color='#ff7f0e', width=barWidth, label='pitch')
        ax2.bar(r2, ang_95s[:N, 1] - avg_vs_ang[:N, 1], bottom=avg_vs_ang[:N, 1], color='#ff7f0e', alpha=0.3, width=barWidth)
        ax2.bar(r3, avg_vs_ang[:N, 2], color='#2ca02c', width=barWidth, label='roll')
        ax2.bar(r3, ang_95s[:N, 2] - avg_vs_ang[:N, 2], bottom=avg_vs_ang[:N, 2], color='#2ca02c', alpha=0.3, width=barWidth)
        ax2.set_xlabel('Trace')
        ax2.set_ylabel('deg/s')
        ax2.set_xticks([r + barWidth for r in range(N)])
        ax2.set_xticklabels(['1', '2', '3', '4', '5'])
        ax2.legend()

        dest = 'Fig7-avg_velocity.pdf'
        fig.savefig(os.path.join(figures_path, dest))
        logging.info("Saving mean velocity plots to {}".format(dest))

    
    @staticmethod
    def plot_res_per_trace(results_path, figures_path, w):
        dists_path = os.path.join(results_path, 'distances')
        # Figure 8
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax = ax.flatten()
        for a in ax:
            plt.setp(a.spines.values(), linewidth=1)
            a.xaxis.grid(which='major', linestyle='dotted', linewidth=1)
            a.yaxis.grid(which='major', linestyle='dotted', linewidth=1)
            a.set_yscale('log')
        
        ax[0].set_ylabel('meters')
        ax[0].set_ylim(1e-5, 0.5)
        ax[1].set_xlabel('Trace')
        ax[1].set_ylabel('degrees')
        flierprops = dict(marker='o', markersize=3, markeredgecolor='#686868', linewidth=0.1)
        
        all_euc_dists = []
        all_ang_dists = []
        euc_files = glob.glob(os.path.join(dists_path, "euc_dists_*_{}ms.npy".format(w)))
        euc_files = sorted(euc_files, key=lambda x:float(re.findall("(\d+)",x)[0]))
        ang_files = glob.glob(os.path.join(dists_path, "ang_dists_*_{}ms.npy".format(w)))
        ang_files = sorted(ang_files, key=lambda x:float(re.findall("(\d+)",x)[0]))
        
        for i, file in enumerate(euc_files):
            euc_dists = np.load(file)
            all_euc_dists.append(euc_dists)
            
        for i, file in enumerate(ang_files):
            ang_dists = np.load(file)
            all_ang_dists.append(ang_dists)
        
        bp = ax[0].boxplot(all_euc_dists, whis=(1,99), flierprops=flierprops)
        bp = ax[1].boxplot(all_ang_dists, whis=(1,99), flierprops=flierprops)
        plt.setp(bp['caps'], color='black')
        plt.tight_layout()
        
        dest = os.path.join(figures_path, 'Fig8-boxplots.pdf')
        fig.savefig(dest, facecolor='white')
        logging.info("Saving per-trace boxplots to {}".format(dest))
    
    @staticmethod
    def compute_mean(results_path):
        for i, pred in enumerate(['Baseline', 'AutoReg', 'Kalman']):
            res_pred_path = os.path.join(results_path, 'res_{}.csv'.format(pred.lower()))
            df = pd.read_csv(res_pred_path)
            mean_df = df.groupby("LAT")[df.columns[2:]].mean()
            mean_csv = os.path.join(results_path, 'res_{}_mean.csv'.format(pred.lower()))
            mean_df.to_csv(mean_csv)
            logging.info("Saving mean results for {} to {}".format(pred, mean_csv))
            
    @staticmethod
    def plot_mean(results_path, figures_path, metric):
        plt.rcParams.update(style)
        fig, ax = plt.subplots(1, 2, figsize=(20,8))
        ax = ax.flatten()
        
        for i, pred in enumerate(zip(['Baseline', 'AutoReg', 'Kalman'], ['s', 'o', 'H'])):
            res_mean_path = os.path.join(results_path, 'res_{}_mean.csv'.format(pred[0].lower()))
            df_mean = pd.read_csv(res_mean_path)
            pws = (df_mean.loc[:, 'LAT'].to_numpy() * 1e3).astype(int)
            
            for j, dist in enumerate(zip(['euc', 'ang'], ['meter', 'degree'])):
                ax[j].plot(pws, df_mean.loc[:, '{}_{}'.format(metric, dist[0])].to_numpy(), label=pred[0], marker=pred[1])
                ax[j].set_ylabel('{} [{}]'.format(metric.upper(), dist[1]), labelpad=10)
                ax[j].set_xlabel('Look-ahead time [ms]', labelpad=10)
                ax[j].set_xlim(pws[0]-1, pws[-1]+1)
                ax[j].xaxis.set_major_locator(MultipleLocator(pws[0]))
                ax[j].xaxis.set_major_formatter(FormatStrFormatter('%d'))
                ax[j].tick_params(labelsize=14)
                ax[j].yaxis.grid(which='major', linestyle='dotted', linewidth=1)
                ax[j].legend(loc='upper left', fontsize=20)
                
        fig.tight_layout()
        dest = os.path.join(figures_path, 'Fig9-avg_{}.pdf'.format(metric))
        fig.savefig(dest)
        logging.info("Plotting mean results for all predictors to {}".format(dest))
        
    @staticmethod
    def make_pdf(figures_path):
        if shutil.which('pdflatex') != None:
            logging.info("Generating pdf using the reproduced results")
            # Copy the generated plots to Latex dir
            copy_tree(figures_path, 'acmmm20_src/figures')
            # Create pdf
            os.chdir('acmmm20_src')
            os.system('pdflatex acmmm20.tex > /dev/null 2>&1')
            os.system('bibtex acmmm20 > /dev/null 2>&1')
            # Repeat to get the references
            os.system('pdflatex acmmm20.tex > /dev/null 2>&1')
            os.system('pdflatex acmmm20.tex > /dev/null 2>&1')
            # Move pdf to the top-level folder and cleanup
            shutil.copy('acmmm20.pdf', '../acmmm20.pdf')
            os.remove('acmmm20.pdf')
            os.chdir('..')
        else:
            logging.error("pdflatex not found on your system, cant't generate PDF!")
        
