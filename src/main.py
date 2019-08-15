import argparse
from typing import Dict

import env
import jobs
from options import job_options


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', ""):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_options(opts: str) -> Dict[str, str]:
    """Parse an options string from key1:value1,key2:value2 to dict representation."""
    pairs = opts.strip().split(',')
    parsed: Dict[str, str] = {}
    for pair in pairs:
        key, value = (elem.strip() for elem in pair.split(':'))
        parsed[key] = value

    return parsed


def main(args):
    """Parse arguments and start jobs."""
    if args.local_rank in [-1, 0]:
        env.logging.info('Running main script.')
    job, opts = args.job, parse_options(args.opts)
    opts['resume'] = args.resume
    opts['local_rank'] = args.local_rank

    if job == 'setup':
        jobs.setup(job_options.SetupOptions(**opts))
    elif job == 'create_dummy_set':
        jobs.create_dummy_set(job_options.CreateDummySetOptions(**opts))
    elif job == 'prepro_set':
        jobs.prepro_set(job_options.PreproSetOptions(**opts))
    elif job == 'run_model':
        jobs.run_model(job_options.ModelRunOptions(**opts))
    elif job == 'evaluate_model':
        jobs.evaluate_model(job_options.ModelEvaluateOptions(**opts))
    elif job == 'visualise_model':
        jobs.visualise_model(job_options.ModelVisualiseOptions(**opts))
    else:
        raise ValueError(f'Unknown job: {job}.')


if __name__ == '__main__':
    """Main entry point of application."""
    parser = argparse.ArgumentParser()
    parser.add_argument('job',
                        type=str,
                        help='The job to start.',
                        choices=['setup', 'create_dummy_set', 'prepro_set', 'run_model', 'evaluate_model'])
    parser.add_argument('-o', '--opts', required=False,
                        type=str,
                        help='Optional arguments to be passed to the job formatted as key1:value1,key2:value2',
                        default='')
    parser.add_argument('-r', '--resume', type=str2bool, nargs='?', const=False, default=False)
    parser.add_argument('--local_rank', type=int, default=-1)
    arguments = parser.parse_args()
    main(arguments)
