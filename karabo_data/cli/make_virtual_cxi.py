import argparse
import logging
import os
import os.path as osp
import re
import sys

from karabo_data import RunDirectory
from karabo_data.components import AGIPD1M, LPD1M
from karabo_data.exceptions import SourceNameError

log = logging.getLogger(__name__)


def _get_detector(data, min_modules):
    for cls in (AGIPD1M, LPD1M):
        try:
            return cls(data, min_modules=min_modules)
        except SourceNameError:
            continue


def main(argv=None):
    ap = argparse.ArgumentParser('karabo-data-make-virtual-cxi')
    ap.add_argument('run_dir', help="Path to an EuXFEL run directory")
    # Specifying a proposal directory & a run number is the older interface.
    # If the run_number argument is passed, run_dir is used as proposal.
    ap.add_argument('run_number', nargs="?", help=argparse.SUPPRESS)
    ap.add_argument(
        '-o', '--output',
        help="Filename or path for the CXI output file. "
             "By default, it is written in the proposal's scratch directory."
    )
    ap.add_argument(
        '--min-modules', type=int, default=9, metavar='N',
        help="Include trains where at least N modules have data (default 9)"
    )
    args = ap.parse_args(argv)
    out_file = args.output

    logging.basicConfig(level=logging.INFO)

    if args.run_number:
        # proposal directory, run number
        run  = 'r%04d' % int(args.run_number)
        proposal = args.run_dir
        run_dir = osp.join(args.run_dir, 'proc', run)
        if out_file is None:
            out_file = osp.join(proposal, 'scratch', '{}_detectors_virt.cxi'.format(run))

    else:
        # run directory
        run_dir = os.path.abspath(args.run_dir)
        if out_file is None:
            m = re.search(r'/(raw|proc)/(r\d{4})/?$', run_dir)
            if not m:
                sys.exit("ERROR: '-o outfile' option needed when "
                         "input directory doesn't look like .../proc/r0123")
            proposal = run_dir[:m.start()]
            fname = '{}_{}_detectors_virt.cxi'.format(*m.group(2, 1))
            out_file = osp.join(proposal, 'scratch', fname)

    out_dir = osp.dirname(osp.abspath(out_file))

    if not os.access(run_dir, os.R_OK):
        sys.exit("ERROR: Don't have read access to {}".format(run_dir))
    if not os.access(out_dir, os.W_OK):
        sys.exit("ERROR: Don't have write access to {}".format(out_dir))

    log.info("Reading run directory %s", run_dir)
    run = RunDirectory(run_dir)
    det = _get_detector(run, args.min_modules)
    if det is None:
        sys.exit("No AGIPD or LPD sources found in {!r}".format(run_dir))

    det.write_virtual_cxi(out_file)

if __name__ == '__main__':
    main()
