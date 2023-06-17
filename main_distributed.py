import argparse
import logging
import os
import pprint
import sys
import yaml

import submitit

logging.basicConfig(
    filename='application.log',  # Specify the log file name
    filemode='a',  # Append to the log file
    format='%(asctime)s %(levelname)s - %(message)s',  # Specify the log format
    level=logging.INFO
)
logger = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder', type=str,
    help='location to save submitit logs')
parser.add_argument(
    '--batch-launch', action='store_true',
    help='whether fname points to a file to batch-launch several config files')
parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')
parser.add_argument(
    '--partition', type=str,
    help='cluster partition to submit jobs on')
parser.add_argument(
    '--nodes', type=int, default=1,
    help='num. nodes to request for job')
parser.add_argument(
    '--tasks-per-node', type=int, default=1,
    help='num. procs per node')
parser.add_argument(
    '--time', type=int, default=4300,
    help='time in minutes to run job')


class Trainer:
    def __init__(self, fname='configs.yaml', resume_training=False):
        self.fname = fname
        self.resume_training = resume_training

    def __call__(self):
        try:
            fname = self.fname
            resume_training = self.resume_training
            logger.info(f'called-params {fname}')

            # Load script params
            with open(fname, 'r') as y_file:
                params = yaml.safe_load(y_file)
                logger.info('loaded params...')
                pp = pprint.PrettyPrinter(indent=4)
                pp.pprint(params)

            resume_preempt = False if resume_training is None else resume_training
            app_main(args=params, resume_preempt=resume_preempt)
        except Exception as e:
            logger.exception(f'An error occurred: {str(e)}')
            sys.exit(1)

    def checkpoint(self):
        try:
            fb_trainer = Trainer(self.fname, True)
            return submitit.helpers.DelayedSubmission(fb_trainer)
        except Exception as e:
            logger.exception(f'An error occurred: {str(e)}')
            sys.exit(1)


def launch():
    try:
        executor = submitit.AutoExecutor(
            folder=os.path.join(args.folder, 'job_%j'),
            slurm_max_num_timeout=20)
        executor.update_parameters(
            slurm_partition=args.partition,
            slurm_mem_per_gpu='55G',
            timeout_min=args.time,
            nodes=args.nodes,
            tasks_per_node=args.tasks_per_node,
            cpus_per_task=10,
            gpus_per_node=args.tasks_per_node)

        config_fnames = [args.fname]

        jobs, trainers = [], []
        with executor.batch():
            for cf in config_fnames:
                fb_trainer = Trainer(cf)
                job = executor.submit(fb_trainer)
                trainers.append(fb_trainer)
                jobs.append(job)

        for job in jobs:
            print(job.job_id)
    except Exception as e:
        logger.exception(f'An error occurred: {str(e)}')
        sys.exit(1)


if __name__ == '__main__':
    try:
        args = parser.parse_args()
        launch()
    except Exception as e:
        logger.exception(f'An error occurred: {str(e)}')
        sys.exit(1)
