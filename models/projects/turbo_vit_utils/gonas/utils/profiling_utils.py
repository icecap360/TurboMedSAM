import time
from itertools import zip_longest
from typing import Sequence

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch import nn


def _latency_on_cpu(net, inputs, N, context) -> np.ndarray:
    """Measures inference latency of a network on CPU for N runs.

    Args:
        net: PyTorch model.
        inputs: Dummy inputs to be passed into model.
        N: Number of inference trials to run.
        context: Specifies which context manager should be used
                 {'none', 'no_grad', 'inference_mode'}.

    Returns:
        List of size N containing latencies.
    """
    net.to("cpu")
    net.eval()
    lat = []

    for _ in range(5):
        _ = net(inputs)
    if context == "none":
        from contextlib import nullcontext

        cm = nullcontext()
    elif context == "no_grad":
        cm = torch.no_grad()
    elif context == "inference_mode":
        cm = torch.inference_mode()
    else:
        raise ValueError("Unknown context mode")

    with cm:
        for n in range(N):
            start = time.time()
            _ = net(inputs)
            end = time.time()
            lat.append(end - start)

    lat = np.array(lat)
    return lat


def _latency_on_gpu(net, inputs, N=100, context="none", use_fp16=False):
    """Measures inference latency of a network on GPU for N runs.

    Args:
        net: PyTorch model.
        inputs: Dummy inputs to be passed into model.
        N: Number of inference trials to run.
        context: Specifies which context manager should be used
                 {'none', 'no_grad', 'inference_mode'}.

    Returns:
        Array of size N containing latencies.
    """
    device = torch.device("cuda:0")
    net.to(device)
    net.eval()
    timings = np.zeros((N, 1))
    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for _ in range(50):
        _ = net(inputs)

    if context == "none":
        from contextlib import nullcontext

        cm = nullcontext()
    elif context == "no_grad":
        cm = torch.no_grad()
    elif context == "inference_mode":
        cm = torch.inference_mode()
    else:
        raise ValueError("Unknown context mode")

    with torch.autocast(device.type, enabled=use_fp16):
        with cm:
            for n in range(N):
                starter.record()
                _ = net(inputs)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[n] = curr_time / 1000

    return timings


def latency(net, inputs, device, N=100, context="none", use_fp16=False):
    if device == "cpu":
        return _latency_on_cpu(net, inputs, N, context)

    if device == "cuda" or device == "cuda:0":
        return _latency_on_gpu(net, inputs, N, context, use_fp16)


def module_shape_printer(module, input, output):
    module._module_shapes = {
        "input_shapes": [inp.shape for inp in input],
        "output_shape": output.shape,
    }


def register_shape_printer(model):
    handles = []
    for name, module in model.named_modules():
        module._module_name = name  # Store name in the module
        handles.append(module.register_forward_hook(module_shape_printer))

    return handles


def unregister_hooks(handles):
    for handle in handles:
        handle.remove()


class ModelDiffer:
    def __init__(self, model1: nn.Module, model2: nn.Module, trace_flow: bool = False):
        self.model1 = model1
        self.model2 = model2
        self.trace_flow = trace_flow
        self.table = Table(show_header=True, header_style="bold magenta")
        self.table.add_column("Match", style="dim")
        self.table.add_column("Model 1 Layer")
        self.table.add_column("Model 2 Layer")

        if self.trace_flow:
            self.table.add_column("Model 1 Input Shapes")
            self.table.add_column("Model 2 Input Shapes")
            self.table.add_column("Model 1 Output Shape")
            self.table.add_column("Model 2 Output Shape")

    def _compare_modules(self, module1: nn.Module, module2: nn.Module, path: str):
        if not isinstance(
            module1, (nn.Sequential, nn.ModuleList, nn.ModuleDict)
        ) and not isinstance(module2, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            shapes1 = getattr(
                module1,
                "_module_shapes",
                {"input_shapes": "N/A", "output_shape": "N/A"},
            )
            shapes2 = getattr(
                module2,
                "_module_shapes",
                {"input_shapes": "N/A", "output_shape": "N/A"},
            )
            row_style = (
                "red" if str(module1) != str(module2) or shapes1 != shapes2 else "green"
            )
            match_icon = "[red]✘[/red]" if row_style == "red" else "[green]✔[/green]"

            if self.trace_flow:
                self.table.add_row(
                    match_icon,
                    f"[bold]{path}:[/bold] {str(module1)}",
                    f"[bold]{path}:[/bold] {str(module2)}",
                    str(shapes1["input_shapes"]),
                    str(shapes2["input_shapes"]),
                    str(shapes1["output_shape"]),
                    str(shapes2["output_shape"]),
                    style=row_style,
                )
            else:
                self.table.add_row(
                    match_icon,
                    f"[bold]{path}:[/bold] {str(module1)}",
                    f"[bold]{path}:[/bold] {str(module2)}",
                    style=row_style,
                )

        # Recursively compare child modules
        for (name1, child1), (name2, child2) in zip_longest(
            module1.named_children(), module2.named_children(), fillvalue=(None, None)
        ):
            if child1 is not None and child2 is not None:
                self._compare_modules(child1, child2, f"{path}/{name1}")
            elif child1 is not None:
                self._add_discrepancy_row(
                    f"[bold]{path}/{name1} exists in Model 1 but not in Model 2[/bold]"
                )
            else:
                self._add_discrepancy_row(
                    f"[bold]{path}/{name2} exists in Model 2 but not in Model 1[/bold]"
                )

    def _add_discrepancy_row(self, message):
        if self.trace_flow:
            self.table.add_row("[red]✘[/red]", message, "", "", "", "", "", style="red")
        else:
            self.table.add_row("[red]✘[/red]", message, "", style="red")

    def compare(self, verbose=False, dummy_input=None):
        if str(self.model1) == str(self.model2):
            print("The models are identical.")
            return

        if dummy_input is not None:
            if self.trace_flow is False:
                raise ValueError(
                    "Cannot trace flow of inputs without enabling trace_flow in the constructor."
                )

            handle1 = register_shape_printer(self.model1)
            handle2 = register_shape_printer(self.model2)

            if not isinstance(dummy_input, Sequence):
                dummy_input = (dummy_input,)

            # Execute a forward pass to get the shapes
            with torch.no_grad():
                _ = self.model1(*dummy_input)
                _ = self.model2(*dummy_input)

            unregister_hooks(handle1)
            unregister_hooks(handle2)

        console = Console(width=250)
        self._compare_modules(self.model1, self.model2, "Root")

        if verbose:
            console.print(self.table)
        else:
            headers = [c.header for c in self.table.columns]
            diff_table = Table(*headers, show_header=True, header_style="bold magenta")
            # diff_table.columns = self.table.columns
            num_rows = len(self.table.rows)
            for row_idx, row in enumerate(self.table.rows):
                cells = [col._cells[row_idx] for col in self.table.columns]
                if "red" in row.style:
                    diff_table.add_row(
                        *cells, style=row.style, end_section=row.end_section
                    )
            console.print(diff_table)
