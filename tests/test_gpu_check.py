from tools.gpu_check import check_gpu


def test_check_gpu_never_raises_and_has_the_expected_shape():
    info = check_gpu()
    for key in ("ok", "gpu_name", "driver_version", "vram_mb", "cupy_version",
                "cuda_runtime", "compute_capability", "problems"):
        assert key in info
    assert isinstance(info["ok"], bool) and isinstance(info["problems"], list)
    # whatever machine this runs on, "ok" and "problems" must agree
    assert info["ok"] == (not info["problems"])
