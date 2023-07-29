import torch
import evaluate as hf_evaluate
from tqdm import tqdm


def evaluate_cls_glue_fn(
    model,
    task,
    eval_dataloader,
    is_regression=False,
    num_samples: int = None,
    progress_bar: bool = False,
) -> dict:
    assert (
        num_samples is None or num_samples >= eval_dataloader.batch_size
    ), f"num_samples {num_samples} must be greater than batch_size {eval_dataloader.batch_size}"

    metric = hf_evaluate.load("glue", task)
    model.eval()
    device = model.device
    if num_samples:
        num_batches = num_samples // eval_dataloader.batch_size
    else:
        num_batches = len(eval_dataloader)

    progress_bar = tqdm(
        eval_dataloader,
        desc="Evaluating the best",
        total=num_batches,
        disable=not progress_bar,
    )

    for i, batch in enumerate(eval_dataloader):
        references = batch.pop("labels")
        with torch.no_grad():
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(**batch)

        predictions = (
            outputs.logits.argmax(dim=-1)
            if not is_regression
            else outputs.logits.squeeze()
        )
        references = references.to(outputs.logits.device)
        metric.add_batch(predictions=predictions, references=references)
        progress_bar.update(1)
        if num_samples and i >= num_batches:
            break
    results = metric.compute()
    return results
