import numpy as np
import gnuplotlib as gp
import yaml
import os

def main(args):
    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    d = np.loadtxt(os.path.join(manifest['outdir'],
                                'train_report.dat'))

    n_epoch = d[:,0]
    n_iter = d[:,1]
    loss = d[:,2]

    x = n_epoch + n_iter/np.max(n_iter)

    y = loss

    gp.plot(x, y,
            _with = 'lines',
            terminal = 'dumb 80,15',
            unset = 'grid')

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type = str,
                        help = "manifest of file to track")

    args = parser.parse_args()

    main(args)
