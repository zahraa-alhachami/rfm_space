"""The project's single Silhouette estimator.

WHY THIS MODULE EXISTS
----------------------
Two scripts reported different numbers for the same quantity on Olist:

    allsets_commonspace_summary.csv   K=3  partition -0.0061  space +0.0673
    harness_decomposition.csv         K=3  partition -0.0049  space +0.0669

The preprocessing and the clustering were identical in both - log1p, then
StandardScaler, then PCA(n_components=2, random_state=42), then KMeans with
k-means++, n_init=10, random_state=42. The difference was the Silhouette call
alone:

    rfm_allsets_commonspace_test.py   silhouette_score(X, labels)
                                      -> exact, all 93,358 points
    rfm_commonspace_harness.py        silhouette_score(X, labels,
                                          sample_size=10000, random_state=42)
                                      -> a 10,000-point estimate

So one was exact and the other an estimate; they were not two different samples.
Both now call `silhouette` below, which fixes the rule in one place: exact up to
SUBSAMPLE_ABOVE points, and a seeded SUBSAMPLE_SIZE estimate above it.

WHY SUBSAMPLE AT ALL
--------------------
An exact Silhouette is O(n^2) in distance computations. At 93,358 customers that
is 8.7 billion pairs per call, and the harness makes dozens of calls per dataset.
The subsample keeps the pipeline tractable; the cost is sampling error, which is
measured rather than assumed - see `olist_sampling_uncertainty.csv` and the
Olist agreement check in rfm_consistency_audit.py, whose tolerance is set from
the measured standard deviation.

Scripts that need the exact value for a specific published figure ask for it
explicitly with `silhouette(X, labels, exact=True)`. That is what
rfm_allsets_commonspace_test.py does for the published 2x2 matrix: a figure that
goes into the write-up should not be one subsample draw. The harness keeps the
estimate, because it makes dozens of calls per dataset and the exact cost is
about 130 seconds per call on Olist.

So the two scripts deliberately differ on Olist, by about 0.0012 in the partition
effect: one is exact and the other is a seeded estimate. They are not expected to
agree to storage precision, and the audit's tolerance is sized for exactly that
difference - three times the standard deviation of the difference of two
independent estimates. An earlier version of this file described the exact call
as already in use when it was not; the behaviour was corrected rather than the
claim weakened.
"""

from sklearn.metrics import silhouette_score

import config

RS = config.RANDOM_STATE

# Above this many points the exact Silhouette is impractical for a pipeline that
# calls it dozens of times per dataset.
SUBSAMPLE_ABOVE = 20000
SUBSAMPLE_SIZE = 10000


def silhouette_kwargs(n, seed=RS, exact=False):
    """The keyword arguments this project uses for a dataset of n points.

    sklearn draws the subsample as permutation(n)[:sample_size], so the seed
    fully determines which points are scored. Passing it explicitly is what
    makes the estimate reproducible.
    """
    if exact or n <= SUBSAMPLE_ABOVE:
        return {}
    return dict(sample_size=SUBSAMPLE_SIZE, random_state=seed)


def silhouette(X, labels, seed=RS, exact=False):
    """Silhouette by the project's single rule. Returns a plain float."""
    return float(silhouette_score(X, labels, **silhouette_kwargs(len(X), seed, exact)))


def is_estimated(n, exact=False):
    """True when the value for n points is a subsample estimate, not exact."""
    return (not exact) and n > SUBSAMPLE_ABOVE
