"""Tables teacher defaults inherited from v1.1, with explicit smoke overrides."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TrainConfig:
    num_envs: int = 4096
    iterations: int = 20000
    buffer_steps: int = 1000
    utd: float = .4
    horizon: int = 32
    hist: int = 10
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    lr: float = 2e-4
    gamma: float = .99
    lam: float = .95
    action_noise: float = .1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    warmup_steps: int = 64
    wm_only_iters: int = 500
    lr_decay_iters: int = 250000
    ckpt_every: int = 500
    seed: int = 0
    amp: str = "bf16"
    experiment_kind: str = "formal"
    wm_hidden: tuple = (2048, 2048, 2048)
    actor_hidden: tuple = (2048, 1024, 512, 256)
    critic_hidden: tuple = (256, 128)
    contact_hidden: int = 128

    def validate(self):
        for name in ("num_envs", "iterations", "buffer_steps", "horizon", "hist", "wm_window",
                     "wm_batches", "wm_batch", "pi_batch", "critic_epochs", "critic_minibatches",
                     "warmup_steps", "lr_decay_iters", "ckpt_every", "contact_hidden"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.wm_only_iters) is not int or self.wm_only_iters < 0:
            raise ValueError("wm_only_iters must be nonnegative")
        if self.hist != 10 or self.buffer_steps < self.wm_window:
            raise ValueError("Tables retains ten history frames and complete WM windows")
        if self.experiment_kind not in ("formal", "smoke"):
            raise ValueError("Unknown experiment kind")
        if self.experiment_kind == "formal" and self.wm_only_iters != 500:
            raise ValueError("Formal from-scratch tables training requires 500 WM-only iterations")
        if not (0 < self.gamma < 1 and 0 <= self.lam <= 1):
            raise ValueError("Invalid gamma/lambda")
        if any(not math.isfinite(v) or v <= 0 for v in (self.lr, self.utd)):
            raise ValueError("Invalid learning settings")
        if self.action_noise < 0 or not math.isfinite(self.action_noise) or self.amp not in ("none", "bf16"):
            raise ValueError("Invalid action noise/AMP")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Invalid seed")
        for name in ("wm_hidden", "actor_hidden", "critic_hidden"):
            if not getattr(self, name) or any(type(n) is not int or n < 1 for n in getattr(self, name)):
                raise ValueError("Hidden sizes must be nonempty positive integer sequences")
        if self.experiment_kind == "formal" and (tuple(self.wm_hidden), tuple(self.actor_hidden), tuple(self.critic_hidden), self.contact_hidden) != (
                (2048, 2048, 2048), (2048, 1024, 512, 256), (256, 128), 128):
            raise ValueError("Formal tables model sizes must preserve the v1.1 hidden layers")
        return self

    def stage(self, iteration):
        if type(iteration) is not int or iteration < 1:
            raise ValueError("Iteration is one-based")
        return "wm_only" if iteration <= self.wm_only_iters else "actor_critic"

    def lr_at(self, iteration):
        if iteration >= self.lr_decay_iters:
            return self.lr * .03
        return self.lr * (.03 + .97 * .5 * (1 + math.cos(math.pi * iteration / self.lr_decay_iters)))

    def to_dict(self):
        return asdict(self)
