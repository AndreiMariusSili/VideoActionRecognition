import argparse as ap
import typing as tp

import env
import jobs
import options.job_options as jo


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', ""):
        return False
    else:
        raise ap.ArgumentTypeError('Boolean value expected.')


def parse_options(opts: str) -> tp.Dict[str, str]:
    """Parse an options string from key1:value1,key2:value2 to dict representation."""
    pairs = opts.strip().split(',')
    parsed: tp.Dict[str, str] = {}
    for pair in pairs:
        key, value = (elem.strip() for elem in pair.split(':'))
        parsed[key] = value

    return parsed


def main(args):
    """Parse arguments and start jobs."""
    if args.local_rank in [-1, 0]:
        env.LOGGER.info('Running main script.')
    job, opts = args.job, parse_options(args.opts)
    opts['local_rank'] = args.local_rank

    if job == 'setup':
        jobs.setup(jo.SetupOptions(**opts))
    elif job == 'create_dummy_set':
        jobs.create_dummy_set(jo.CreateDummySetOptions(**opts))
    elif job == 'prepro_set':
        jobs.prepro_set(jo.PreproSetOptions(**opts))
    elif job == 'run_experiment':
        jobs.run_experiment(jo.RunExperimentOptions(**opts))
    elif job == 'evaluate_experiment':
        jobs.evaluate_model(jo.EvaluateExperimentOptions(**opts))
    elif job == 'visualise_model':
        jobs.visualise_model(jo.VisualiseOptions(**opts))
    else:
        raise ValueError(f'Unknown job: {job}.')


if __name__ == '__main__':
    """Main entry point of application."""
    parser = ap.ArgumentParser()
    parser.add_argument('job',
                        type=str,
                        help='The job to start.',
                        choices=['setup', 'create_dummy_set', 'prepro_set', 'run_experiment', 'evaluate_experiment'])
    parser.add_argument('-o', '--opts', required=False,
                        type=str,
                        help='Optional arguments to be passed to the job formatted as key1:value1,key2:value2',
                        default='')
    parser.add_argument('-r', '--resume', type=str2bool, default=False)
    parser.add_argument('--local_rank', type=int, default=-1)
    arguments = parser.parse_args()
    main(arguments)
