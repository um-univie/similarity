import math

import numpy as np
import pytest

import similarity_rs as sr

rng = np.random.default_rng(0)


def test_isa_and_version():
    assert sr.isa() in {"Scalar", "Avx2", "Avx512", "Neon"}
    assert sr.__version__


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_vector_metrics(dtype):
    a = rng.random(37).astype(dtype)
    b = rng.random(37).astype(dtype)
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    assert sr.distance(a, b, "cosine") == pytest.approx(cos, rel=1e-5)
    assert sr.distance(a, b, "euclidean") == pytest.approx(np.linalg.norm(a - b), rel=1e-5)
    assert sr.distance(a, b, "manhattan") == pytest.approx(np.abs(a - b).sum(), rel=1e-5)
    assert math.isnan(sr.distance(np.zeros(3, dtype), np.ones(3, dtype), "cosine"))
    with pytest.raises(ValueError):
        sr.distance(a, b[:-1])
    with pytest.raises(ValueError):
        sr.distance(a, b, "nope")


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_cdist_pdist_top_k(dtype):
    x = rng.random((20, 9)).astype(dtype)
    y = rng.random((7, 9)).astype(dtype)
    d = sr.cdist(x, y, "euclidean")
    assert d.shape == (20, 7) and d.dtype == dtype
    ref = np.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1))
    np.testing.assert_allclose(d, ref, rtol=1e-5)

    p = sr.pdist(x, "sqeuclidean")
    iu = np.triu_indices(20, 1)
    full = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    np.testing.assert_allclose(p, full[iu], rtol=1e-4)

    idx, sc = sr.top_k(x[3], x, 3, "euclidean")
    assert idx[0] == 3 and sc[0] == 0
    assert list(idx) == list(np.argsort(full[3])[:3])

    with pytest.raises(ValueError):
        sr.cdist(x, y[:, :-1])
    with pytest.raises(ValueError):
        sr.cdist(x, y.astype(np.float64 if dtype == np.float32 else np.float32))
    with pytest.raises(ValueError):
        sr.top_k(x[0], x, 0)


def test_divergences():
    p = rng.random(50) + 0.01
    q = rng.random(50) + 0.01
    pn, qn = p / p.sum(), q / q.sum()
    assert sr.kl_divergence(p, q) == pytest.approx(np.sum(pn * np.log(pn / qn)), rel=1e-10)
    assert sr.shannon_entropy(p) == pytest.approx(-np.sum(pn * np.log(pn)), rel=1e-10)
    m = (pn + qn) / 2
    js = 0.5 * np.sum(pn * np.log(pn / m)) + 0.5 * np.sum(qn * np.log(qn / m))
    assert sr.js_divergence(p, q) == pytest.approx(js, rel=1e-10)
    assert sr.js_distance(p, q) == pytest.approx(math.sqrt(js / math.log(2)), rel=1e-10)
    assert sr.total_variation_distance(p, q) == pytest.approx(0.5 * np.abs(pn - qn).sum(), rel=1e-10)
    assert sr.hellinger_distance(p, p) == pytest.approx(0.0, abs=1e-7)
    assert sr.bhattacharyya_distance(p, q) >= 0
    assert sr.cross_entropy(p, q) == pytest.approx(-np.sum(pn * np.log(qn)), rel=1e-10)
    assert sr.kl_divergence(p, q[:-1]) is None


def make_spectrum(base, jitter=0.003):
    keep = rng.random(len(base)) < 0.7
    mz = base[keep] + rng.uniform(-jitter, jitter, keep.sum())
    extra = rng.uniform(50, 500, rng.integers(0, 6))
    mz = np.concatenate([mz, extra])
    inten = rng.uniform(0.1, 100, len(mz))
    return mz, inten


@pytest.fixture(scope="module")
def library():
    base = rng.uniform(50, 500, 40)
    lib = [make_spectrum(base) for _ in range(120)]
    queries = [make_spectrum(base) for _ in range(10)]
    return lib, queries


def test_pairwise_spectral(library):
    lib, _ = library
    (ma, ia), (mb, ib) = lib[0], lib[1]
    assert sr.entropy_similarity(ma, ia, ma, ia) == pytest.approx(1.0)
    s = sr.entropy_similarity(ma, ia, mb, ib, tolerance=0.01)
    assert 0.0 <= s <= 1.0
    c, n = sr.spectral_cosine(ma, ia, ma, ia)
    assert c == pytest.approx(1.0) and n == len(ma)
    with pytest.raises(ValueError):
        sr.entropy_similarity(ma, ia[:-1], mb, ib)


def test_flash_vs_library(library):
    lib, queries = library
    flash = sr.FlashEntropyIndex(lib, tolerance=0.01)
    sl = sr.SpectralLibrary(lib, metric="entropy", tolerance=0.01)
    assert len(flash) == len(sl) == len(lib)
    assert flash.n_peaks > 0
    for mz, inten in queries:
        scores = flash.search(mz, inten)
        hits = sl.search(mz, inten)
        dense = np.zeros(len(lib))
        for i, s, n in hits:
            assert n >= 1
            dense[i] = s
        np.testing.assert_allclose(scores, dense, atol=1e-10)
        idx, sc = flash.search_top_k(mz, inten, 5)
        assert list(idx) == [h[0] for h in sl.search(mz, inten, k=5)]
        assert np.all(np.diff(sc) <= 0)
    many = flash.search_many(queries, 3)
    assert len(many) == len(queries) and all(len(i) <= 3 for i, _ in many)


def test_precursor_filter_and_matrix(library):
    lib, queries = library
    prec = rng.uniform(100, 110, len(lib))
    qprec = rng.uniform(100, 110, len(queries))
    sl = sr.SpectralLibrary(
        lib, precursor_mz=prec, metric="cosine", tolerance=10, ppm=True, precursor_tolerance=0.5, min_matched_peaks=2
    )
    res = sl.search_many(queries, precursor_mz=qprec, k=0)
    for q, hits in enumerate(res):
        for i, s, n in hits:
            assert abs(prec[i] - qprec[q]) <= 0.5 and n >= 2 and 0 < s <= 1
    with pytest.raises(ValueError):
        sr.SpectralLibrary(lib, precursor_mz=prec[:-1])
    with pytest.raises(ValueError):
        sr.SpectralLibrary(lib, metric="nope")

    small = sr.SpectralLibrary(lib[:15], metric="cosine")
    m = small.similarity_matrix()
    assert m.shape == (15, 15)
    np.testing.assert_allclose(m, m.T)
    np.testing.assert_allclose(np.diag(m), 1.0)
