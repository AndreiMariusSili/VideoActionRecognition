from typing import Dict
import argparse
import jobs


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
    job, opts = args.job, parse_options(args.opts)

    if job == 'create_dummy_set':
        jobs.create_dummy_set(jobs.options.CreateDummySetOptions(**opts))
    elif job == 'prepro_set':
        jobs.prepro_set(jobs.options.PreproSetOptions(**opts))
    else:
        raise ValueError(f'Unknown job: {job}.')


if __name__ == '__main__':
    """Main entry point of application."""
    parser = argparse.ArgumentParser()
    parser.add_argument('job',
                        type=str,
                        help='The job to start.',
                        choices=['create_dummy_set', 'prepro_set'])
    parser.add_argument('-o', '--opts', required=False,
                        type=str,
                        help='Optional arguments to be passed to the job formatted as key1:value1,key2:value2',
                        default='')
    arguments = parser.parse_args()
    main(arguments)
