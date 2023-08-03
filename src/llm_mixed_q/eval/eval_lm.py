import logging
import math

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def eval_lm_wikitext2(
    model,
    eval_dataloader,
    num_samples: int = None,
    progress_bar: bool = False,
    input_device: str = None,
):
    assert (
        num_samples is None or num_samples >= eval_dataloader.batch_size
    ), f"num_samples {num_samples} must be greater than batch_size {eval_dataloader.batch_size}"

    losses = []
    model.eval()

    if input_device is None:
        input_device = model.device
    if num_samples:
        num_batches = num_samples // eval_dataloader.batch_size
    else:
        num_batches = len(eval_dataloader)

    progress_bar = tqdm(
        eval_dataloader,
        desc="Evaluating on Wikitext-2",
        total=num_batches,
        disable=not progress_bar,
    )

    batch_size = eval_dataloader.batch_size
    seq_len = next(iter(eval_dataloader))["input_ids"].shape[1]
    num_samples = 0
    for i, batch in enumerate(eval_dataloader):
        assert (
            batch["input_ids"].shape[1] == seq_len
        ), f"sequence length is not a constant current seq_len = {batch['input_ids'].shape[1]} != {seq_len}"
        with torch.no_grad():
            batch = {
                k: v.to(input_device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(**batch)
        loss = outputs.loss.item() * batch_size * seq_len
        losses.append(loss)
        num_samples += batch_size

        progress_bar.update(1)
        if num_samples and i >= num_batches:
            break

    reduced_loss = sum(losses) / (seq_len * num_samples)
    try:
        perplexity = math.exp(reduced_loss)
    except OverflowError:
        perplexity = float("inf")

    results = {
        "loss": reduced_loss,
        "perplexity": perplexity,
        "num_samples": num_samples,
        "seq_len": seq_len,
        "batch_size": batch_size,
    }
    return results
