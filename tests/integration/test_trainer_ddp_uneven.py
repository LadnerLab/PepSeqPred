import logging
import socket
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from pepseqpred.core.models.ffnn import PepSeqFFNN
from pepseqpred.core.train.trainer import Trainer, TrainerConfig


pytestmark = pytest.mark.integration


def _rank_batches(rank: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    batch_count = 1 if rank == 0 else 3
    batches = []
    for batch_index in range(batch_count):
        x = torch.full(
            (1, 2, 2),
            float((rank + 1) * (batch_index + 1)),
            dtype=torch.float32,
        )
        y = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        mask = torch.ones((1, 2), dtype=torch.long)
        batches.append((x, y, mask))
    return batches


def _optimizer_payload(trainer: Trainer) -> list[dict[str, torch.Tensor | int]]:
    payload = []
    for group in trainer.optimizer.param_groups:
        for parameter in group["params"]:
            state = trainer.optimizer.state[parameter]
            payload.append({
                "step": int(state["step"].item()),
                "exp_avg": state["exp_avg"].detach().cpu().clone(),
                "exp_avg_sq": state["exp_avg_sq"].detach().cpu().clone(),
            })
    return payload


def _uneven_ddp_worker(
    rank: int,
    world_size: int,
    init_method: str,
    output_dir: str,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(1234)
        model = PepSeqFFNN(
            emb_dim=2,
            hidden_sizes=(4,),
            dropouts=(0.0,),
            num_classes=1,
            use_layer_norm=False,
            use_residual=False,
        )
        ddp_model = DDP(model)
        trainer = Trainer(
            model=ddp_model,
            train_loader=_rank_batches(rank),
            logger=logging.getLogger(f"uneven_ddp_rank_{rank}"),
            val_loader=None,
            config=TrainerConfig(
                epochs=2,
                batch_size=1,
                learning_rate=1e-2,
                device="cpu",
            ),
        )

        epoch_outputs = []
        for epoch in range(2):
            epoch_outputs.append(trainer._run_epoch(epoch, train=True))
        result = {
            "epochs": epoch_outputs,
            "model_state": {
                name: tensor.detach().cpu().clone()
                for name, tensor in trainer.model.state_dict().items()
            },
            "optimizer_state": _optimizer_payload(trainer),
        }
        torch.save(result, Path(output_dir) / f"rank_{rank}.pt")
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="PyTorch 2.4 Gloo rendezvous is unavailable in the Windows test environment",
)
def test_uneven_ddp_training_keeps_model_and_adam_state_synchronized(
    tmp_path: Path,
) -> None:
    world_size = 2
    output_dir = tmp_path / "rank_outputs"
    output_dir.mkdir()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    # Use the legacy TCPStore backend for portability across CPU PyTorch builds.
    init_method = f"tcp://127.0.0.1:{port}?use_libuv=0"

    mp.spawn(
        _uneven_ddp_worker,
        args=(world_size, init_method, str(output_dir)),
        nprocs=world_size,
        join=True,
    )

    rank_results = [
        torch.load(
            output_dir / f"rank_{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for rank in range(world_size)
    ]

    for rank, result in enumerate(rank_results):
        expected_real_batches = 1 if rank == 0 else 3
        expected_dummy_batches = 2 if rank == 0 else 0
        for epoch_output in result["epochs"]:
            assert epoch_output["synchronized_steps"] == 3
            assert epoch_output["optimizer_steps"] == 3
            assert epoch_output["zero_valid_steps"] == 0
            assert epoch_output["real_batches"] == expected_real_batches
            assert epoch_output["dummy_batches"] == expected_dummy_batches
            assert epoch_output["n_residues"] == 8

    rank_zero = rank_results[0]
    rank_one = rank_results[1]
    assert rank_zero["model_state"].keys() == rank_one["model_state"].keys()
    for name in rank_zero["model_state"]:
        torch.testing.assert_close(
            rank_zero["model_state"][name],
            rank_one["model_state"][name],
            rtol=0.0,
            atol=0.0,
        )

    assert len(rank_zero["optimizer_state"]) == len(rank_one["optimizer_state"])
    for state_zero, state_one in zip(
        rank_zero["optimizer_state"],
        rank_one["optimizer_state"],
    ):
        assert state_zero["step"] == 6
        assert state_one["step"] == 6
        torch.testing.assert_close(
            state_zero["exp_avg"], state_one["exp_avg"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            state_zero["exp_avg_sq"],
            state_one["exp_avg_sq"],
            rtol=0.0,
            atol=0.0,
        )
