"""Source example checks run before packaging; examples are not in the wheel."""
import pytest


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_runnable_training_example(method, tmp_path):
    from examples.torch_hybrid import train_and_restore
    checkpoint = tmp_path / f'{method}.pt'
    initial, final = train_and_restore(method, epochs=15, checkpoint=checkpoint)
    assert final < initial
    assert checkpoint.exists()


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
@pytest.mark.parametrize('encoding', ['angle', 'zz'])
def test_runnable_encoded_training_example(method, encoding, tmp_path):
    from examples.torch_hybrid import train_and_restore
    checkpoint = tmp_path / f'{encoding}-{method}.pt'
    initial, final = train_and_restore(method, epochs=15, checkpoint=checkpoint, encoding=encoding)
    assert final < initial
    assert checkpoint.exists()
