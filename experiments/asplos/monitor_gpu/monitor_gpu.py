from io import StringIO
import sys
import torch
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import wandb
from transformers import AutoModel
from deepspeed.profiling.flops_profiler import get_model_profile
from torch.profiler import profile, record_function, ProfilerActivity


class StdoutCapturer(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def main():
    parser = ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seq_len", type=int, default=196)
    parser.add_argument("--warm_up", type=int, required=True)
    parser.add_argument("--log_dir", type=str, default=".")
    parser.add_argument("--model_parallelism", action="store_true")

    args = parser.parse_args()

    if args.project is None:
        args.project = "{}_monitor_gpu".format(args.model_name)

    if args.model_parallelism:
        model = AutoModel.from_pretrained(args.model_name, device_map="auto")
    else:
        model = AutoModel.from_pretrained(args.model_name)
        model.to("cuda")

    model.eval()

    wandb_logger = wandb.init(
        project=args.project,
        name=args.tag,
        config={
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "warm_up": args.warm_up,
            "model_parallelism": args.model_parallelism,
            "gpu name": torch.cuda.get_device_name(),
        },
    )

    print("============================ GPU Monitor ============================")
    print(
        f"model_name:{args.model_name}, batch_size:{args.batch_size}, seq_len:{args.seq_len}, warm_up:{args.warm_up}, model_parallelism:{args.model_parallelism}, gpu name:{torch.cuda.get_device_name()}"
    )

    inputs = {
        "input_ids": torch.randint(0, 1000, (args.batch_size, args.seq_len)),
    }

    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        for i in tqdm(range(args.warm_up)):
            if i < args.warm_up - 3:
                _ = model(**inputs)
                continue
            if i == args.warm_up - 3:
                with StdoutCapturer() as output:
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    ) as prof:
                        _ = model(**inputs)
                    print(prof.key_averages().table(sort_by="cuda_time_total"))
                    _ = model(**inputs)
                print("\n".join(output[-2:]))
            if i == args.warm_up - 2:
                flops, macs, params = get_model_profile(
                    model=model,
                    kwargs=inputs,
                    print_profile=True,
                    detailed=False,
                    as_string=False,
                )

    print(f"{flops=}, {macs=}, {params=}")
    print("=====================================================================")

    wandb_logger.finish()


if __name__ == "__main__":
    main()
