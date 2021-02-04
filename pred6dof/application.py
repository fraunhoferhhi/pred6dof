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

__version__ = '0.2'

import argparse
import logging
import os
import sys
import numpy as np
import toml
from .runners import AutoregRunner, KalmanRunner, BaselineRunner
from .reporter import Reporter
from .utils import get_csv_files, preprocess_trace

class Application:
    """Command line interface for the pred6dof application"""

    def __init__(self):
        """Initializes a new application instance"""
        config_path = os.path.join(os.getcwd(), 'config.toml')
        self.cfg = toml.load(config_path)

        self.command = None
        self.verbosity = None
        self.raw_dataset_path = None
        self.raw_dataset_path = None
        self.output_path = None
        self.sampling_time = self.cfg['dt']
        self.algorithm = None
        self.dataset_path = None
        self.results_path = None
        self.figures_path = None
        self.pred_window = None

    def run(self):
        """Runs the application"""
        self.parse_command_line_arguments()

        # Configure logging
        numeric_level = getattr(logging, self.verbosity, None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % self.verbosity)
        logging.basicConfig(level=numeric_level)
        
        # Check desired action
        if self.command == 'run':
            if self.algorithm == 'autoreg':
                self.run_autoreg()
            elif self.algorithm == 'kalman':
                self.run_kalman()
            elif self.algorithm == 'baseline':
                self.run_baseline()
        elif self.command == 'prepare':
            self.prepare()
        elif self.command == 'report':
            self.report()

    def run_autoreg(self):
        """Runs Autoregression on all traces and evaluates the results"""
        runner = AutoregRunner(self.pred_window,
                               self.dataset_path,
                               self.results_path)
        runner.run()

    def run_kalman(self):
        """Runs Kalman filter on all traces and evaluates the results"""
        runner = KalmanRunner(self.pred_window,
                              self.dataset_path,
                              self.results_path)
        runner.run()
        
    def run_baseline(self):
        """Runs baseline (no-prediction) on all traces and evaluates the results"""
        runner = BaselineRunner(self.pred_window,
                                self.dataset_path,
                                self.results_path)
        runner.run()

    def prepare(self):
        """Resample all user traces in the given path to a common sampling time and make
        temporal spacing between samples equal"""
        for trace_path in get_csv_files(self.raw_dataset_path):
            preprocess_trace(trace_path, self.sampling_time, self.output_path)
        logging.info("Interpolated traces written to {}".format(self.output_path))

    def report(self):
        trace_path = os.path.join(self.dataset_path, "1.csv")
        Reporter.plot_trace(trace_path, self.figures_path) # Fig. 5
        Reporter.plot_head_velocity(self.dataset_path, self.figures_path) # Fig. 6-7
        Reporter.compute_mean(self.results_path)
        Reporter.plot_res_per_trace(self.results_path, self.figures_path, "60") # Fig. 8
        Reporter.plot_mean(self.results_path, self.figures_path, "mae") # Fig. 9
        Reporter.make_pdf(self.figures_path) 

    def parse_command_line_arguments(self):
        """Parses the cmdline arguments of the application"""
        argument_parser = argparse.ArgumentParser(
            prog='pred6dof',
            description='A tool for testing different 6DoF head motion prediction algorithms'
        )

        argument_parser.add_argument(
            '-v',
            '--version',
            action='version',
            version='pred6dof command line interface {0}'.format(__version__),
            help='display the version string of the application and exit'
        )

        argument_parser.add_argument(
            '-V',
            '--verbosity',
            dest='verbosity',
            type=str.upper,
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='set the verbosity level of the logging. Defaults to "INFO"'
        )

        sub_parsers = argument_parser.add_subparsers(dest='command')
        Application.add_prepare_command(sub_parsers)
        Application.add_run_command(sub_parsers)
        Application.add_report_command(sub_parsers)

        # Parses the arguments
        args = argument_parser.parse_args()
        self.command = args.command
        self.verbosity = args.verbosity

        if len(sys.argv) < 2:
            argument_parser.print_usage()
            sys.exit(1)
        elif self.command == 'prepare':
            self.raw_dataset_path = args.raw_dataset_path
            self.output_path = args.output_path
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            self.sampling_time = args.sampling_time
        elif self.command == 'run':
            self.algorithm = args.algorithm
            self.dataset_path = args.dataset_path
            self.results_path = args.results_path
            if not os.path.exists(self.results_path):
                os.makedirs(self.results_path)
            self.pred_window = np.asarray(args.pred_window)
        elif self.command == 'report':
            self.dataset_path = args.dataset_path
            self.results_path = args.results_path
            self.figures_path = args.figures_path
            if not os.path.exists(self.figures_path):
                os.makedirs(self.figures_path)
            
    @staticmethod
    def add_prepare_command(sub_parsers):
        """"
        Adds the process command which samples the raw data traces with
        even temporal spacing.

        Parameters
        ----------
            sub_parsers: Action
                The sub-parser to which the command is to be added.

        """
        prepare_command_parser = sub_parsers.add_parser(
            'prepare',
            help='preprocess the raw data traces to obtain temporally evenly-spaced samples',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        prepare_command_parser.add_argument(
            '-r',
            '--raw-dataset-path',
            dest='raw_dataset_path',
            type=str,
            metavar='',
            default='./data/raw',
            help='Path to the raw head motion traces collected from the headset'
        )

        prepare_command_parser.add_argument(
            '-o',
            '--output-path',
            dest='output_path',
            type=str,
            metavar='',
            default='./data/interpolated',
            help='Path to the interpolated (resampled) dataset'
        )

        prepare_command_parser.add_argument(
            '-t',
            '--sampling-time',
            dest='sampling_time',
            type=int,
            metavar='',
            default=0.005,
            help='Sampling time [s]'
        )

    @staticmethod
    def add_run_command(sub_parsers):
        """"
        Adds the run command which runs and evalutes the selected prediction algorithm.

        Parameters
        ----------
            sub_parsers: Action
                The sub-parser to which the command is to be
                added.

        """

        run_command_parser = sub_parsers.add_parser(
            'run',
            help='run the selected prediction algorithm and evaluate its performance',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter
        )
        
        run_command_parser.add_argument(
            '-a',
            '--algorithm',
            dest='algorithm',
            type=str,
            choices=['autoreg', 'kalman', 'baseline'],
            default='kalman',
            help='Selects which prediction algorithm is run on the data traces'
        )

        run_command_parser.add_argument(
            '-d',
            '--dataset-path',
            dest='dataset_path',
            type=str,
            metavar='',
            default='./data/interpolated',
            help='Path to the head motion traces dataset'
        )
        
        run_command_parser.add_argument(
            '-w',
            '--pred-window',
            dest='pred_window',
            metavar='',
            type=int,
            nargs='+',
            default=[40],
            help='Sets the prediction window/look-ahead time [ms]'
        )

        run_command_parser.add_argument(
            '-r',
            '--results-path',
            dest='results_path',
            type=str,
            metavar='',
            default='./results/tabular',
            help='Path where results are stored as CSV'
        )
        
    @staticmethod
    def add_report_command(sub_parsers):
        """"
        Adds the report

        Parameters
        ----------
            sub_parsers: Action
                The sub-parser to which the command is to be
                added.

        """

        report_command_parser = sub_parsers.add_parser(
            'report',
            help='compute the mean results and generate plots',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter
        )
        
        report_command_parser.add_argument(
            '-d',
            '--dataset-path',
            dest='dataset_path',
            type=str,
            metavar='',
            default='./data/interpolated',
            help='Dataset path'
        )
        
        report_command_parser.add_argument(
            '-f',
            '--figures-path',
            dest='figures_path',
            type=str,
            metavar='',
            default='./results/figures',
            help='Path where plots are saved as PDF and PNG'
        )

        report_command_parser.add_argument(
            '-r',
            '--results-path',
            dest='results_path',
            type=str,
            metavar='',
            default='./results/tabular',
            help='Path where results are stored as CSV'
        )