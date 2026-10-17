from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from super_gradients.training.sg_trainer.sg_trainer import Trainer


@pytest.mark.parametrize("accumulate", [1, 2, 4])
@pytest.mark.parametrize("epoch", [0, 1])
@pytest.mark.parametrize("limit", [None, 2])
def test_accumulated_updates_match_mean_loss_including_partial_window(accumulate, epoch, limit):
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    expected = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    reference = torch.optim.SGD([expected], lr=0.1)
    targets = torch.tensor([1.0, 3.0, 5.0])[:limit]
    trainer = SimpleNamespace(
        training_params=SimpleNamespace(normalize_accumulated_gradients=True, clip_grad_norm=None),
        train_loader=[None] * 3, max_train_batches=limit, batch_accumulate=accumulate, max_epochs=3,
        scaler=torch.amp.GradScaler("cpu", enabled=False), optimizer=optimizer, ema=False,
        phase_callback_handler=MagicMock(),
    )
    for index, target in enumerate(targets):
        Trainer._backward_step(trainer, (parameter - target) ** 2, epoch, index, None)
    for window in targets.split(accumulate):
        ((expected - window) ** 2).mean().backward()
        reference.step()
        reference.zero_grad()
    torch.testing.assert_close(parameter, expected)
    assert trainer.phase_callback_handler.on_train_batch_gradient_step_end.call_count == (len(targets) + accumulate - 1) // accumulate
    assert parameter.grad is None or parameter.grad.item() == 0
