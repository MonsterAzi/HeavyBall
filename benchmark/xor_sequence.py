import itertools
from typing import List

import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from heavyball.utils import set_torch
from utils import trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class Model(nn.Module):
    def __init__(self, size, depth):
        super().__init__()
        self.embed0 = nn.Embedding(2, size)
        self.embed1 = nn.Embedding(2, size)
        self.enc = nn.LSTM(size, size, depth, batch_first=True)
        self.dec = nn.LSTM(size, size, depth, batch_first=True)
        self.proj = nn.Linear(size, 1, bias=False)

    def forward(self, inp):
        i0, i1 = inp.chunk(2, 1)
        i0 = self.embed0(i0)
        i1 = self.embed1(i1)
        _, state = self.enc(i0)
        out, _ = self.dec(i1, state)
        return self.proj(out)


class ModelTransformer(nn.Module):
    def __init__(self, size, depth):
        super().__init__()
        self.embed0 = nn.Embedding(2, size)
        self.embed1 = nn.Embedding(2, size)
        self.enc = nn.TransformerEncoderLayer(d_model=size, nhead=1)
        self.dec = nn.TransformerDecoderLayer(d_model=size, nhead=1)

    def forward(self, inp):
        i0, i1 = inp.chunk(2, 1)
        i0 = self.embed0(i0)
        i1 = self.embed1(i1)
        out = self.enc(i0)
        out = self.dec(i1, out)
        return out


def win(loss):
    return loss < 0.1


@app.command()
def main(method: List[str] = typer.Option(['qr'], help='Eigenvector method to use (for SOAP)'),
         dtype: List[str] = typer.Option(["float32"], help='Data type to use'), length: int = 128, size: int = 128,
         depth: int = 1, batch: int = 128, steps: int = 500_000, weight_decay: float = 0, opt: List[str] = typer.Option(
            ['PrecondSchedulePaLMForeachSOAP'],
            help='Optimizers to use')):
    dtype = [getattr(torch, d) for d in dtype]

    for args in itertools.product(method, dtype, [(length, size, depth, batch)], opt, [weight_decay]):
        m, d, (l, s, dp, b), o, wd = args
        torch.manual_seed(0x1239121)

        model = Model(s, dp)

        def data():
            inp = torch.randn((b, l, 1), device='cuda', dtype=d)
            inp = inp > 0
            i0, i1 = inp.chunk(2, 1)
            xored = torch.logical_xor(i0, i1)
            return inp.long().squeeze(-1), xored.to(d)

        trial(model, data, torch.nn.functional.binary_cross_entropy_with_logits, win, steps, o, d, s, b, wd, m, l, dp,
              failure_threshold=10, base_lr=0.001)


if __name__ == '__main__':
    app()
