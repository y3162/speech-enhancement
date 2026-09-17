import torch
import torch.nn as nn


class MagPhaseNetwork(nn.Module):
    def scale_inference(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        energy = audio.pow(2).sum(dim=1).clamp_min(1e-12)
        alpha = torch.sqrt(audio.size(1) / energy).unsqueeze(1)
        return audio * alpha, alpha

    def unscale_inference(
        self,
        audio: torch.Tensor,
        factor: torch.Tensor,
    ) -> torch.Tensor:
        return audio / factor

    @torch.inference_mode()
    def enhance(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if noisy_audio.ndim == 1:
            noisy_audio = noisy_audio.unsqueeze(0)
            squeeze = True
        elif noisy_audio.ndim != 2:
            raise ValueError(
                f"noisy_audio must be [B, T] or [T], got shape {tuple(noisy_audio.shape)}"
            )
        length = noisy_audio.size(1)
        scaled, factor = self.scale_inference(noisy_audio)
        pred = self(self.processor.encode(scaled))
        audio = self.processor.decode(pred, length=length)
        enhanced = self.unscale_inference(audio, factor)
        if squeeze:
            enhanced = enhanced.squeeze(0)
        return enhanced
