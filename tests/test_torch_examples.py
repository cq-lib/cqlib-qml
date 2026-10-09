"""Source example checks run before packaging; examples are not in the wheel."""
import pytest


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_runnable_training_example(method, tmp_path):
    from examples.torch_hybrid import train_and_restore
    checkpoint = tmp_path / f'{method}.pt'
    initial, final = train_and_restore(method, epochs=15, checkpoint=checkpoint)
    assert final < initial
    assert checkpoint.exists()
