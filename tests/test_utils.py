from pathlib import Path

from rl2048.utils import load_yaml, set_seed


def test_load_yaml_reads_mapping(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\nname: demo\n", encoding="utf-8")

    config = load_yaml(config_path)

    assert config == {"seed": 42, "name": "demo"}


def test_set_seed_makes_numpy_random_reproducible():
    import numpy as np

    set_seed(123)
    first = np.random.random()
    set_seed(123)
    second = np.random.random()

    assert first == second
