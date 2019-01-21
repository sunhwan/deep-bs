import argparse
import sys

from . import grid
from . import load, save

def convert(args):
    # guess input/output format from filename
    input_fmt = input.split('.')[-1]
    output_fmt = output.split('.')[-1]
    grid = load(open(args.input), format=input_fmt)
    save(grid, args.output, format=output_fmt)

def subtract(args):
    ref = load(open(args.ref), format="map")
    target = load(open(args.target), format="map")
    try:
        diff = ref - target
    except:
        if args.resample:
            if args.exponent:
                target = (target / -args.factor).exp()
                target = target.resample(center=ref.center, shape=ref.shape)
                target = -args.factor * target.log()
            else:
                target = target.resample(center=ref.center, shape=ref.shape)
            diff = ref - target
        else:
            raise AssertionError('The size of the map must be match. Use --resample keyword.')
    save(diff, open(args.output, 'w'), format="map")

def multiply(args):
    grid = load(open(args.mapfile), format="map")
    if args.exponent:
        grid = (-grid / args.factor).exp()
    else:
        grid = grid * args.factor
    diff = ref - target
    save(diff, open(args.output, 'w'), format="map")

def run(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = 'command'

    psubtract = subparsers.add_parser('convert')
    psubtract.add_argument('input')
    psubtract.add_argument('output')
    psubtract.set_defaults(func=convert)

    psubtract = subparsers.add_parser('subtract')
    psubtract.add_argument('ref')
    psubtract.add_argument('target')
    psubtract.add_argument('--resample', default=False, action='store_true')
    psubtract.add_argument('--exponent', default=False, action='store_true')
    psubtract.add_argument('--factor', default=1, type=float)
    psubtract.add_argument('--output', default="diff.map")
    psubtract.set_defaults(func=subtract)

    pmultiply = subparsers.add_parser('multiply')
    pmultiply.add_argument('mapfile')
    pmultiply.add_argument('--exponent', default=False, action='store_true')
    pmultiply.add_argument('--factor', default=1, type=float)
    pmultiply.add_argument('--output', default="output.map")
    pmultiply.set_defaults(func=multiply)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    run()
