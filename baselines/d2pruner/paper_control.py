"""D2Pruner paper control with local calibration prior."""

from pathlib import Path
import torch

from baselines.d2pruner.adapter import D2PrunerBaseline

DUALSIGNAL_ROOT = Path(__file__).resolve().parents[2] / "qcal"

class D2PrunerPaper(D2PrunerBaseline):
    """D2Pruner with a dualsignal-local debiasing prior.

    BEA's reference class looks for the bias prior under BEA. This subclass
    keeps all generated calibration artifacts under dualsignal.
    """

    def __init__(self, *args, bias_path: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_path = Path(bias_path) if bias_path else (
            DUALSIGNAL_ROOT / "runners/calibration"
            / "d2_qwen25vl_attention_bias.pt"
        )

    def _load_bias(self, n_vis):
        if not self.bias_path.exists():
            return None
        payload = torch.load(self.bias_path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict):
            bias = payload.get("bias")
        else:
            bias = payload
        if bias is None:
            return None
        bias = bias.float()
        if bias.shape[0] != n_vis:
            bias = torch.nn.functional.interpolate(
                bias.unsqueeze(0).unsqueeze(0),
                size=n_vis,
                mode="linear",
                align_corners=False,
            ).squeeze()
        return bias
