import torch as th

from criterion.custom import VAECriterion


def test_vae_criterion():
    crt = VAECriterion()

    recon = th.randn([2, 3, 4, 3, 24, 24])
    pred = th.randn([2, 3, 16])
    mse_target = th.randn([2, 4, 3, 24, 24])
    ce_target = (th.rand([2]) * 2).to(dtype=th.long)
    mean = th.randn(2, 32)
    var = th.rand(2, 32)
    ce, mse, kld = crt(recon, pred, mse_target, ce_target, mean, var)

    assert not ce.shape and not mse.shape and not kld.shape
