# %%
"""
A synthetic DCGAN run, laid out like `DCGANTrainer` in ARENA's [0.5] VAEs & GANs day, to show what
liveplot looks like on that training loop without a GPU. Nothing is trained: the losses are made up
to behave the way the course describes (lossD starting at ln 4 and falling, lossG starting at ln 2
and rising, D(x) and D(G(z)) pulling apart from 1/2), and the "generator" returns ten real CelebA
faces behind noise that fades as training goes on.

Run it cell by cell in the VS Code / Cursor interactive window or Jupyter to see it live: the loss
curves and the samples side by side in one output, with a tqdm bar underneath. Run it as a script
(`python examples/dcgan_synthetic.py`) and nothing is drawn on screen, but it records the run to
`dcgan_synthetic.gif`.

Needs torch and Pillow, and fetches ten CelebA faces (about 70 kB, cached in ~/.cache/liveplot)
through Hugging Face's dataset viewer, not the 1.4 GB dataset.

The parts to copy into the real trainer are marked `# liveplot:`.
"""

import io
import json
import math
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as t
from PIL import Image

from liveplot import LivePlot

MAIN = __name__ == "__main__"
CELEB_IMAGE_SIZE = 64


# %%
# The data: ten CelebA faces, prepared as the course's `get_dataset("CELEB")` does (resize the short
# side to 64, centre-crop to 64x64, uint8), then mapped to [-1, 1] like TANH_RANGE_TRANSFORM.


def celeba_faces(n: int = 10) -> t.Tensor:
    """(n, 3, 64, 64) uint8: the first n images of nielsr/CelebA-faces, the dataset the course uses."""
    cache = Path.home() / ".cache" / "liveplot" / f"celeba_faces_{n}.npy"
    if not cache.exists():
        url = f"https://datasets-server.huggingface.co/rows?dataset=nielsr/CelebA-faces&config=default&split=train&offset=0&length={n}"
        rows = json.load(urllib.request.urlopen(url, timeout=60))["rows"]
        faces = []
        for row in rows:
            img = Image.open(io.BytesIO(urllib.request.urlopen(row["row"]["image"]["src"], timeout=60).read())).convert("RGB")
            scale = CELEB_IMAGE_SIZE / min(img.size)  # transforms.Resize(64): the short side to 64
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
            left, top = (img.width - CELEB_IMAGE_SIZE) // 2, (img.height - CELEB_IMAGE_SIZE) // 2
            img = img.crop((left, top, left + CELEB_IMAGE_SIZE, top + CELEB_IMAGE_SIZE))  # CenterCrop(64)
            faces.append(np.asarray(img).transpose(2, 0, 1))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, np.stack(faces))
    return t.from_numpy(np.load(cache))


def tanh_range(x: t.Tensor) -> t.Tensor:
    """TANH_RANGE_TRANSFORM on a uint8 batch: [0, 255] -> [-1, 1]."""
    return x.float() / 127.5 - 1


# %%
# The trainer. Same shape as the course's DCGANTrainer; the two training steps and the generator are
# fakes, and everything liveplot adds is marked.


@dataclass
class DCGANArgs:
    batch_size: int = 64
    epochs: int = 3
    batches_per_epoch: int = 150  # stands in for len(self.trainloader)
    log_every_n_steps: int = 15  # 250 in the course, with ~3000 batches per epoch
    seconds_per_step: float = 0.03  # stands in for the GPU's time per batch
    seed: int = 0


class SyntheticDCGANTrainer:
    def __init__(self, args: DCGANArgs):
        self.args = args
        self.gen = t.Generator().manual_seed(args.seed)
        self.faces = tanh_range(celeba_faces(10))  # what a well-trained netG(self.fixed_noise) would give
        self.fixed_noise = t.randn(self.faces.shape, generator=self.gen)  # the fake generator's own noise
        self.total_steps = args.epochs * args.batches_per_epoch
        self.d_state = t.zeros(2)  # slowly wandering offsets that make D(x), D(G(z)) look like a real run

    def _progress(self) -> float:
        return self.step / self.total_steps

    def training_step_discriminator(self) -> tuple[float, float, float]:
        """Returns lossD and the discriminator's mean outputs D(x) (real) and D(G(z)) (fake)."""
        gap = 0.26 * (1 - math.exp(-6 * self._progress()))  # D learns to separate real from fake, fast then slow
        self.d_state = 0.9 * self.d_state + 0.04 * t.randn(2, generator=self.gen)
        spike = 0.12 * float(t.rand(1, generator=self.gen) < 0.01)  # the odd batch D gets badly wrong
        d_real = min(max(0.5 + gap + float(self.d_state[0]) - spike, 0.02), 0.98)
        d_fake = min(max(0.5 - gap + float(self.d_state[1]) + spike, 0.02), 0.98)
        lossD = -(math.log(d_real) + math.log(1 - d_fake))  # the course's lossD, at the batch-mean outputs
        return lossD, d_real, d_fake

    def training_step_generator(self, d_fake: float) -> float:
        """lossG = -log D(G(z)), after the discriminator's step has moved D(G(z)) a little."""
        d_fake = min(max(d_fake + 0.03 * float(t.randn(1, generator=self.gen)), 0.02), 0.98)
        return -math.log(d_fake)

    @t.inference_mode()
    def log_samples(self) -> None:
        """netG(self.fixed_noise), faked: the faces, behind noise that fades as training goes on."""
        quality = 1 - math.exp(-4 * self._progress())
        output = quality * self.faces + (1 - quality) * self.fixed_noise
        # Clip values to make the visualization clearer (as the course does)
        output = output.clamp(output.quantile(0.01), output.quantile(0.99))
        self.ax_samples.imshow(output, rows=2)  # liveplot: replaces LiveImage.update / wandb.Image

    def train(self, **plot_kwargs) -> LivePlot:
        self.step = 0
        # liveplot: one figure for the curves and the samples, and the tqdm bar underneath it (this
        # replaces `self.live_image = LiveImage()` and `progress_bar = tqdm(total=...)`)
        self.plot, (ax_loss, self.ax_samples) = LivePlot.subplots(
            1, 2, total=self.total_steps, figsize=(12, 4), width_ratios=(1, 1.25), **plot_kwargs
        )
        ax_loss.plot("lossD", "lossG")
        ax_loss.set_smooth(0.6)  # per-batch GAN losses are noisy: wandb's smoothing, raw values faded behind
        ax_loss.axhline(math.log(4), "ln 4: lossD, D at chance")  # where lossD starts, and where a perfect G ends
        ax_loss.axhline(math.log(2), "ln 2: lossG, D at chance", linestyle=":")  # ... and the same for lossG
        ax_loss.twinx().plot("D(x)", "D(G(z))").set_ylim(0, 1)
        ax_loss.set_title("losses (left), discriminator outputs (right)")
        ax_loss.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=3)  # under the panel, off the curves
        self.ax_samples.set_title("netG(fixed_noise)")

        for epoch in range(self.args.epochs):
            self.plot.set_description(f"epoch {epoch}")  # liveplot: was progress_bar.set_description
            for _ in range(self.args.batches_per_epoch):  # for img_real, label in self.trainloader:
                lossD, d_real, d_fake = self.training_step_discriminator()
                lossG = self.training_step_generator(d_fake)

                # liveplot: log the step's numbers and advance the bar (was progress_bar.update())
                self.plot.log({"lossD": lossD, "lossG": lossG, "D(x)": d_real, "D(G(z))": d_fake})
                self.plot.update()
                self.step += 1

                if self.step % self.args.log_every_n_steps == 0:
                    self.log_samples()
                time.sleep(self.args.seconds_per_step)

        self.log_samples()  # the final samples, whatever the step count
        self.plot.finish()
        return self.plot


# %%
# Run it. In a notebook or interactive window you get the live figure with a bar underneath; as a
# script, the run is recorded to a GIF instead (pass a path to choose where).

if MAIN:
    in_notebook = "ipykernel" in sys.modules
    gif = None if in_notebook else (sys.argv[1] if len(sys.argv) > 1 else "dcgan_synthetic.gif")
    plot = SyntheticDCGANTrainer(DCGANArgs()).train(**({"record": gif, "refresh_seconds": 0.25} if gif else {}))
    if gif:
        print(f"{len(plot.frames)} frames -> {gif}")
