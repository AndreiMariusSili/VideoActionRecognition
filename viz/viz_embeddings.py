import argparse

from main import main

parser = argparse.ArgumentParser()
parser.add_argument('job',
                    type=str,
                    help='The job to start.',
                    choices=['visualise_model'])
parser.add_argument('-o', '--opts', required=False,
                    type=str,
                    help='Optional arguments to be passed to the job formatted as key1:value1,key2:value2',
                    default='')
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('-r', '--resume', type=bool, default=False)
arguments = parser.parse_args()
arguments.opts += ',page:embeddings'
main(arguments)
