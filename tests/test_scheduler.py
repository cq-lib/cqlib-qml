import numpy as np
import pytest

from cqlib_qml.scheduler import (
    ConstantScheduler,
    ExponentialScheduler,
    NoamScheduler,
    KingScheduler,
    SchedulerInitializer,
)


class TestConstantScheduler:
    def test_init(self):
        scheduler = ConstantScheduler(lr=0.01)
        assert scheduler.lr == 0.01

    def test_learning_rate(self):
        scheduler = ConstantScheduler(lr=0.01)
        assert scheduler.learning_rate(step=0) == 0.01
        assert scheduler.learning_rate(step=100) == 0.01
        assert scheduler.learning_rate(step=1000) == 0.01

    def test_set_params_changes_lr(self):
        scheduler = ConstantScheduler(lr=0.01)
        scheduler.set_params({"lr": 0.5})
        assert scheduler.learning_rate(step=0) == 0.5
        assert scheduler.learning_rate(step=1000) == 0.5

    def test_set_params_unknown_key_ignored(self):
        scheduler = ConstantScheduler(lr=0.01)
        scheduler.set_params({"unknown": 1.0})
        assert scheduler.learning_rate(step=0) == 0.01


class TestExponentialScheduler:
    def test_init(self):
        scheduler = ExponentialScheduler(initial_lr=0.01, stage_length=100, decay=0.5)
        assert scheduler.initial_lr == 0.01
        assert scheduler.stage_length == 100
        assert scheduler.decay == 0.5

    def test_learning_rate_staircase_false(self):
        scheduler = ExponentialScheduler(initial_lr=0.01, stage_length=100, staircase=False, decay=0.5)
        lr = scheduler.learning_rate(step=50)
        assert lr == 0.01 * 0.5**0.5

    def test_learning_rate_staircase_true(self):
        scheduler = ExponentialScheduler(initial_lr=0.01, stage_length=100, staircase=True, decay=0.5)
        lr1 = scheduler.learning_rate(step=50)
        lr2 = scheduler.learning_rate(step=150)
        assert lr1 == 0.01
        assert lr2 == 0.005

    def test_str(self):
        scheduler = ExponentialScheduler(initial_lr=0.01, stage_length=100, decay=0.5)
        assert "ExponentialScheduler" in str(scheduler)

    def test_set_params_changes_initial_lr(self):
        scheduler = ExponentialScheduler(initial_lr=0.01, stage_length=10, decay=0.5)
        scheduler.set_params({"initial_lr": 2.0})
        assert scheduler.learning_rate(step=0) == 2.0

    def test_set_params_changes_decay(self):
        scheduler = ExponentialScheduler(initial_lr=1.0, stage_length=10, decay=0.1)
        scheduler.set_params({"decay": 0.5})
        assert scheduler.learning_rate(step=10) == 0.5


class TestNoamScheduler:
    def test_init(self):
        scheduler = NoamScheduler(model_dim=512, scale_factor=1, warmup_steps=4000)
        assert scheduler.model_dim == 512
        assert scheduler.scale_factor == 1
        assert scheduler.warmup_steps == 4000

    def test_learning_rate_warmup(self):
        scheduler = NoamScheduler(model_dim=512, scale_factor=1, warmup_steps=100)
        lr = scheduler.learning_rate(step=50)
        assert lr > 0
        expected = (512 ** (-0.5)) * (50 * 100 ** (-1.5))
        assert abs(lr - expected) < 1e-10

    def test_learning_rate_decay(self):
        scheduler = NoamScheduler(model_dim=512, scale_factor=1, warmup_steps=100)
        lr_at_peak = scheduler.learning_rate(step=100)
        lr_decay = scheduler.learning_rate(step=200)
        assert lr_decay < lr_at_peak

    def test_learning_rate_at_warmup_peak(self):
        scheduler = NoamScheduler(model_dim=512, scale_factor=1, warmup_steps=100)
        lr_peak = scheduler.learning_rate(step=100)
        expected = (512 ** (-0.5)) * (100 ** (-0.5))
        assert abs(lr_peak - expected) < 1e-10

    def test_learning_rate_increases_during_warmup(self):
        scheduler = NoamScheduler(model_dim=512, scale_factor=1, warmup_steps=100)
        lr_prev = scheduler.learning_rate(step=1)
        for step in range(2, 100):
            lr_curr = scheduler.learning_rate(step=step)
            assert lr_curr > lr_prev
            lr_prev = lr_curr

    def test_learning_rate_decreases_after_warmup(self):
        scheduler = NoamScheduler(model_dim=512, scale_factor=1, warmup_steps=100)
        lr_prev = scheduler.learning_rate(step=100)
        for step in range(101, 200, 10):
            lr_curr = scheduler.learning_rate(step=step)
            assert lr_curr < lr_prev
            lr_prev = lr_curr

    def test_str(self):
        scheduler = NoamScheduler(model_dim=512, scale_factor=1, warmup_steps=4000)
        assert "NoamScheduler" in str(scheduler)
        assert "model_dim=512" in str(scheduler)
        assert "warmup_steps=4000" in str(scheduler)

    def test_set_params_changes_warmup_steps(self):
        scheduler = NoamScheduler(model_dim=512, scale_factor=1, warmup_steps=4000)
        scheduler.set_params({"warmup_steps": 100})
        lr_peak = scheduler.learning_rate(step=100)
        expected = (512 ** (-0.5)) * (100 ** (-0.5))
        assert abs(lr_peak - expected) < 1e-10


class TestKingScheduler:
    def test_init(self):
        scheduler = KingScheduler(initial_lr=0.01, patience=100, decay=0.99)
        assert scheduler.initial_lr == 0.01
        assert scheduler.patience == 100
        assert scheduler.decay == 0.99

    def test_learning_rate_no_history(self):
        scheduler = KingScheduler(initial_lr=0.01, patience=10)
        lr = scheduler.learning_rate(step=0, cur_loss=0.5)
        assert lr == 0.01

    def test_learning_rate_with_history(self):
        scheduler = KingScheduler(initial_lr=0.01, patience=3, decay=0.5)
        losses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        for i, loss in enumerate(losses):
            lr = scheduler.learning_rate(step=i, cur_loss=loss)
        assert lr < 0.01

    def test_str(self):
        scheduler = KingScheduler(initial_lr=0.01, patience=100, decay=0.99)
        assert "KingScheduler" in str(scheduler)

    def test_set_params_changes_initial_lr(self):
        scheduler = KingScheduler(initial_lr=0.01, patience=100, decay=0.99)
        scheduler.set_params({"initial_lr": 0.5})
        lr = scheduler.learning_rate(step=0, cur_loss=1.0)
        assert lr == 0.5


class TestSchedulerInitializer:
    def test_from_none(self):
        init = SchedulerInitializer(None, lr=0.01)
        scheduler = init()
        assert isinstance(scheduler, ConstantScheduler)
        assert scheduler.lr == 0.01

    def test_from_str_constant(self):
        init = SchedulerInitializer("constant(lr=0.01)")
        scheduler = init()
        assert isinstance(scheduler, ConstantScheduler)
        assert scheduler.lr == 0.01

    def test_from_str_exponential(self):
        init = SchedulerInitializer("exponential(initial_lr=0.01, stage_length=100, decay=0.5)")
        scheduler = init()
        assert isinstance(scheduler, ExponentialScheduler)

    def test_from_str_noam(self):
        init = SchedulerInitializer("noam(model_dim=512, scale_factor=1, warmup_steps=4000)")
        scheduler = init()
        assert isinstance(scheduler, NoamScheduler)

    def test_from_str_king(self):
        init = SchedulerInitializer("king(initial_lr=0.01, patience=100, decay=0.99)")
        scheduler = init()
        assert isinstance(scheduler, KingScheduler)

    def test_from_dict(self):
        init = SchedulerInitializer({"hyperparameters": {"id": "ConstantScheduler", "lr": 0.01}})
        scheduler = init()
        assert isinstance(scheduler, ConstantScheduler)
        assert scheduler.lr == 0.01

    def test_from_dict_applies_nondefault_lr(self):
        init = SchedulerInitializer({"hyperparameters": {"id": "ConstantScheduler", "lr": 0.05}})
        scheduler = init()
        assert scheduler.learning_rate(step=0) == pytest.approx(0.05)

    def test_from_dict_exponential_applies_params(self):
        init = SchedulerInitializer({"hyperparameters": {
            "id": "ExponentialScheduler", "initial_lr": 1.0,
            "stage_length": 10, "staircase": False, "decay": 0.5,
        }})
        scheduler = init()
        assert scheduler.learning_rate(step=10) == pytest.approx(0.5)

    def test_from_dict_king_applies_initial_lr(self):
        init = SchedulerInitializer({"hyperparameters": {
            "id": "KingScheduler", "initial_lr": 0.5, "patience": 100, "decay": 0.99,
        }})
        scheduler = init()
        assert scheduler.learning_rate(step=0, cur_loss=1.0) == pytest.approx(0.5)

    def test_from_scheduler(self):
        constant = ConstantScheduler(lr=0.01)
        init = SchedulerInitializer(constant)
        scheduler = init()
        assert isinstance(scheduler, ConstantScheduler)
        assert scheduler.lr == 0.01
