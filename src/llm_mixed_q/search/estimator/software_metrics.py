import torch
import evaluate as hf_evaluate


def evaluate_cls_task(
    model, task, eval_dataloader, is_regression=False, num_samples: int = None
) -> dict:
    assert (
        num_samples is None or num_samples >= eval_dataloader.batch_size
    ), f"num_samples {num_samples} must be greater than batch_size {eval_dataloader.batch_size}"

    metric = hf_evaluate.load("glue", task)
    model.eval()
    device = model.device
    if num_samples:
        num_batches = num_samples // eval_dataloader.batch_size

    for i, batch in enumerate(eval_dataloader):
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
        references = batch["labels"]
        metric.add_batch(predictions=predictions, references=references)
        if num_samples and i >= num_batches:
            break
    results = metric.compute()
    return results
