"""
Wrapper mỏng quanh wandb.

Mục tiêu: nếu chưa cài wandb / không có mạng / cfg.wandb.enabled = False
thì training vẫn chạy bình thường, chỉ là không log. Không bao giờ raise.

Cách dùng:
    logger = WandbLogger(cfg, phase="phase1")
    logger.log({"train/step_loss": loss, "train/lr": lr}, step=global_step)
    logger.log_epoch(epoch, {"train/loss": .., "val/loss": ..})
    logger.finish()
"""

from typing import Optional


class WandbLogger:
    def __init__(self, cfg, phase: str, run_name: Optional[str] = None):
        self.phase = phase
        self.run = None
        self.wandb = None
        self.enabled = bool(getattr(cfg, "wandb", None) and cfg.wandb.enabled)

        if not self.enabled:
            print("[wandb] Tắt (cfg.wandb.enabled = False).")
            return

        try:
            import wandb
        except ImportError:
            print("[wandb] Chưa cài. Chạy `pip install wandb` rồi `wandb login`. Bỏ qua logging.")
            self.enabled = False
            return

        if run_name is None:
            run_name = getattr(cfg.wandb, f"run_name_{phase}", None)
        if run_name is None:
            run_name = f"{cfg.train.experiment_name}-{phase}"

        try:
            self.run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=run_name,
                mode=cfg.wandb.mode,
                job_type=phase,
                group=cfg.train.experiment_name,
                config={**cfg.to_dict(), "phase": phase},
            )
            self.wandb = wandb

            # ── Trục x cho biểu đồ ────────────────────────────────────────
            # Cách wandb phân giải (sdk/internal/handler.py):
            #   1. Tra tên CHÍNH XÁC trong _metric_defines  → thắng mọi wildcard
            #   2. Nếu không có, duyệt _metric_globs theo THỨ TỰ CHÈN và lấy
            #      match ĐẦU TIÊN  → glob hẹp phải đăng ký TRƯỚC glob rộng
            #
            # LỖI CŨ: "train/*" đăng ký sau "train/step_*" thì không sao cho
            # step_loss, nhưng nó nuốt luôn train/amp_scale (không có định nghĩa
            # riêng) → amp_scale bị vẽ trên trục epoch, mà trong vòng lặp train
            # thì "epoch" không được log cùng lúc → cả ~200 điểm của 1 epoch dồn
            # vào 1 giá trị x. Biểu đồ trông phẳng lì và giấu sạch dao động.
            wandb.define_metric("global_step")
            wandb.define_metric("epoch")

            # (a) Glob hẹp trước, glob rộng sau
            wandb.define_metric("train/step_*", step_metric="global_step")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("test/*", step_metric="epoch")

            # (b) Tên chính xác — luôn thắng glob, không phụ thuộc thứ tự
            for _k in ("train/lr", "train/grad_norm",
                       "train/amp_scale", "train/scale_dropped"):
                wandb.define_metric(_k, step_metric="global_step")

            print(f"[wandb] Run: {self.run.name}  →  {self.run.url}")
        except Exception as e:
            print(f"[wandb] init thất bại ({e}) → chạy tiếp không logging.")
            self.enabled = False
            self.run = None

    # ──────────────────────────────
    def log(self, data: dict, step: Optional[int] = None):
        if not self.enabled or self.run is None:
            return
        try:
            payload = dict(data)
            if step is not None:
                payload["global_step"] = step
            self.wandb.log(payload)
        except Exception as e:
            print(f"[wandb] log lỗi: {e}")

    def log_epoch(self, epoch: int, data: dict):
        if not self.enabled or self.run is None:
            return
        try:
            self.wandb.log({**data, "epoch": epoch})
        except Exception as e:
            print(f"[wandb] log_epoch lỗi: {e}")

    def watch(self, model, log_freq: int = 200):
        if not self.enabled or self.run is None:
            return
        try:
            self.wandb.watch(model, log="all", log_freq=log_freq)
        except Exception as e:
            print(f"[wandb] watch lỗi: {e}")

    def set_summary(self, data: dict):
        if not self.enabled or self.run is None:
            return
        try:
            for k, v in data.items():
                self.run.summary[k] = v
        except Exception as e:
            print(f"[wandb] summary lỗi: {e}")

    def log_artifact(self, path: str, name: str, artifact_type: str = "model"):
        if not self.enabled or self.run is None:
            return
        try:
            art = self.wandb.Artifact(name=name, type=artifact_type)
            art.add_file(str(path))
            self.run.log_artifact(art)
        except Exception as e:
            print(f"[wandb] artifact lỗi: {e}")

    def finish(self):
        if not self.enabled or self.run is None:
            return
        try:
            self.wandb.finish()
        except Exception:
            pass


def flatten_metrics(metrics: dict, prefix: str) -> dict:
    """
    Chuyển dict metrics lồng nhau ({'per_class': {...}, 'macro': {...}, 'loss': x})
    thành dict phẳng để wandb vẽ được: 'val/mass_auc', 'val/macro_ap', 'val/loss', ...
    """
    out = {}
    for key, value in metrics.items():
        if key == "per_class" and isinstance(value, dict):
            for cls_name, cls_metrics in value.items():
                for m_name, m_val in cls_metrics.items():
                    if isinstance(m_val, (int, float)):
                        out[f"{prefix}/{cls_name}_{m_name}"] = m_val
        elif key == "macro" and isinstance(value, dict):
            for m_name, m_val in value.items():
                if isinstance(m_val, (int, float)):
                    out[f"{prefix}/{m_name}"] = m_val
        elif isinstance(value, (int, float)):
            out[f"{prefix}/{key}"] = value
    return out
