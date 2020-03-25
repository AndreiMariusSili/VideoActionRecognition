import argparse as ap
import dataclasses as dc
import datetime
import typing as tp

import env
import jobs
import options.job_options as jo


def parse_options(opts: str, opts_dc: dc.dataclass) -> tp.Any:
    """Parse an options string from key1:value1,key2:value2 to dict representation."""
    type_fns = opts_dc.__annotations__
    pairs = opts.strip().split(',')
    parsed: tp.Dict[str, str] = {}
    for pair in pairs:
        key, value = (elem.strip() for elem in pair.split(':'))
        parsed[key] = type_fns[key](value)

    return parsed


def main(args):
    """Parse arguments and start jobs."""
    start = datetime.datetime.now()
    if args.local_rank <= 0:
        env.LOGGER.info(f'START: {start}')

    if args.job == 'setup':
        opts = parse_options(args.opts, jo.SetupOptions)
        jobs.setup(jo.SetupOptions(**opts))
    elif args.job == 'select_subset':
        opts = parse_options(args.opts, jo.SelectSubsetOptions)
        jobs.select_subset(jo.SelectSubsetOptions(**opts))
    elif args.job == 'prepro_set':
        opts = parse_options(args.opts, jo.PreproSetOptions)
        jobs.prepro_set(jo.PreproSetOptions(**opts))
    elif args.job == 'run_experiment':
        opts = parse_options(args.opts, jo.RunExperimentOptions)
        opts = jo.RunExperimentOptions(**opts)
        jobs.run_experiment(args.local_rank, opts)
    elif args.job == 'eval_experiment':
        opts = parse_options(args.opts, jo.EvaluateExperimentOptions)
        opts = jo.EvaluateExperimentOptions(**opts)
        jobs.evaluate_experiment(args.local_rank, opts)
    elif args.job == 'vis_experiment':
        opts = parse_options(args.opts, jo.VisualiseExperimentOptions)
        jobs.visualise_model(jo.VisualiseExperimentOptions(**opts))
    else:
        raise ValueError(f'Unknown job: {args.job}.')

    end = datetime.datetime.now()
    if args.local_rank <= 0:
        env.LOGGER.info(f'END: {end}. Run time: {end - start}')


if __name__ == '__main__':
    """Main entry point of application."""
    parser = ap.ArgumentParser()
    parser.add_argument('job',
                        type=str,
                        help='The job to start.',
                        choices=['setup', 'select_subset', 'prepro_set', 'run_experiment', 'eval_experiment'])
    parser.add_argument('-o', '--opts', required=False,
                        type=str,
                        help='Optional arguments to be passed to the job formatted as key1:value1,key2:value2',
                        default='')
    parser.add_argument('--local_rank', default=-1, type=int, help='The local rank in distributed setting.')
    arguments = parser.parse_args()
    main(arguments)
