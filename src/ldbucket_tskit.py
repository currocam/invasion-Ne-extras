#!/usr/bin/env python3
import numba
import numpy as np
import tskit
import pandas as pd


def allele_frequencies(ts: tskit.TreeSequence, sample_sets=None) -> np.ndarray:
    if sample_sets is None:
        sample_sets = [ts.samples()]
    n = np.array([len(x) for x in sample_sets])

    def f(x):
        return x / n

    return ts.sample_count_stat(
        sample_sets,
        f,
        len(sample_sets),
        windows="sites",
        polarised=True,
        mode="site",
        strict=False,
        span_normalise=False,
    )


@numba.njit
def _ldbucket(
    positions: np.ndarray,
    freqs: np.ndarray,
    Y: np.ndarray,
    nbins: int,
    minimum: float,
    maximum: float,
    right_edges_in_bp: np.ndarray,
):
    """
    Just-in-time helper function for `ldbucket`.

    - positions: np.ndarray, positions of the sites in the genome.
    - freqs: np.ndarray, frequencies of the alleles at the sites.
    - Y: np.ndarray, genotype matrix of shape (num_samples, num_sites).
    - nbins: int, number of bins to divide the genome into.
    - minimum: float, minimum distance between sites to consider.
    - maximum: float, maximum distance between sites to consider.
    - right_edges_in_bp: np.ndarray, right edges of the bins in base pairs.
    """
    num_sites = len(positions)
    s = Y.shape[0]
    ld = np.zeros(nbins)
    counter = np.zeros(nbins)

    for i in range(num_sites):
        px = freqs[i]
        x = (Y[:, i] - 2 * px) / np.sqrt(2 * px * (1 - px))
        for j in range(i + 1, num_sites):
            dist_bp = positions[j] - positions[i]
            if dist_bp < minimum:
                continue
            if dist_bp > maximum:
                break
            py = freqs[j]
            y = (Y[:, j] - 2 * py) / np.sqrt(2 * py * (1 - py))
            index = 0
            while index < nbins and right_edges_in_bp[index] < dist_bp:
                index += 1
            if index == nbins:
                continue
            xy = np.sum(x * y)
            x2y2 = np.sum(x * x * y * y)
            ld[index] += (xy**2 - x2y2) / (s * (s - 1))
            counter[index] += 1
    for k in range(nbins):
        if counter[k] > 0:
            ld[k] /= counter[k]
        else:
            ld[k] = np.nan
    return ld


def hapne_default(recombination_rate: float) -> dict:
    nbins = 25
    left_edges_in_cm = np.array([0.5 + 0.5 * i for i in range(nbins)])
    right_edges_in_cm = np.array([1.0 + 0.5 * i for i in range(nbins)])
    left_edges_in_bp = np.array(
        [x / 100.0 / recombination_rate for x in left_edges_in_cm]
    )
    right_edges_in_bp = np.array(
        [x / 100.0 / recombination_rate for x in right_edges_in_cm]
    )
    minimum = round(left_edges_in_bp[0])
    maximum = round(right_edges_in_bp[-1])
    return {
        "nbins": nbins,
        "left_edges_in_cm": left_edges_in_cm,
        "right_edges_in_cm": right_edges_in_cm,
        "left_edges_in_bp": left_edges_in_bp,
        "right_edges_in_bp": right_edges_in_bp,
        "minimum": minimum,
        "maximum": maximum,
    }


def ldbucket(
    ts: tskit.TreeSequence, recombination_rate=1e-8, maf_threshold=0.25, name="tskit"
) -> pd.DataFrame:
    """
    Function for processing linkage disequilibrium data from `tskit` data. Convenient when working with simulated datasets.

    Arguments:
        - ts: tskit.TreeSequence: Tree sequence object containing the genetic data.
        - recombination_rate: float: Recombination rate per base pair per generation.
        - maf_threshold: float: Minor allele frequency threshold for filtering sites.
        - name: str: Name of the contig (provided by the user).

    - Returns a pandas DataFrame with the following columns:
        - 'contig_name': name of the contig (provided by the user)
        - 'bin_index': index of the bin (starting from 0)
        - 'left_edge': left edge of the bin in Morgan
        - 'right_edge': right edge of the bin in Morgan
        - 'mean_ld': mean linkage disequilibrium within the bin (computed as E[X_iX_jY_iY_j])
    """
    # Compute bins for the given recombination rate
    bins = hapne_default(recombination_rate)
    # Compute allele frequencies and discard sites with MAF below threshold
    freqs = allele_frequencies(ts)[:, 0]  # shape: (n_sites,)
    mask = np.bitwise_or(freqs < maf_threshold, freqs > (1 - maf_threshold))
    sts = ts.delete_sites(np.where(mask)[0]).simplify(filter_sites=True)
    assert type(sts) is tskit.TreeSequence
    # Compute allele frequencies and genotype matrix
    freqs = freqs[~mask]
    Y = sts.genotype_matrix().T
    Y = Y[::2,] + Y[1::2,]
    observed_mean_ld = _ldbucket(
        sts.sites_position,
        freqs,
        Y,
        bins["nbins"],
        bins["minimum"],
        bins["maximum"],
        bins["right_edges_in_bp"],
    )
    return pd.DataFrame(
        {
            "contig_name": name,
            "bin_index": np.arange(bins["nbins"]),
            "left_bin": bins["left_edges_in_cm"] / 100,
            "right_bin": bins["right_edges_in_cm"] / 100,
            "mean": observed_mean_ld,
        }
    )


if __name__ == "__main__":
    import sys

    msg_error = (
        "Usage: python "
        + sys.argv[0]
        + " <tskit_file> <contig_name> <recombination_rate> <maf_threshold>"
    )
    if len(sys.argv) != 5:
        print(msg_error)
        sys.exit(1)
    tskit_file = sys.argv[1]
    contig_name = sys.argv[2]
    recombination_rate = float(sys.argv[3])
    maf_threshold = float(sys.argv[4])
    # Load the tree sequence from file
    ts = tskit.load(tskit_file)
    assert ts is not None
    df = ldbucket(ts, recombination_rate, maf_threshold, name=contig_name)
    df.to_csv(sys.stdout, index=False)
