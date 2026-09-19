from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import nemo.collections.asr as nemo_asr
import torch
import torch.nn as nn
from nemo.collections.asr.parts.numba.rnnt_loss import TDTLossNumba
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
from nemo.core.utils import numba_utils
from nemo.utils import logging as nemo_logging
from omegaconf import open_dict

nemo_logging.set_verbosity(nemo_logging.ERROR)
numba_utils.set_numba_compat_strictness(False)


def clip_encoded_time(
    encoded: torch.Tensor,
    encoded_length: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make encoded T equal max(encoded_length) so TDT/RNNT length checks pass."""
    if encoded.ndim != 3:
        raise ValueError(f"encoded must be [B, D, T], got {tuple(encoded.shape)}")
    time = encoded.size(-1)
    clipped_length = encoded_length.to(dtype=torch.int64).clamp(min=0, max=time)
    cut = int(clipped_length.max().item()) if clipped_length.numel() else 0
    if time != cut:
        encoded = encoded[:, :, :cut]
    return encoded, clipped_length


@dataclass(frozen=True)
class Token:
    token_id: int
    token: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class Recognition:
    text: str
    encoded_length: int
    samples_per_encoder_frame: int
    tokens: list[Token]
    frame_token_index: list[int]


@dataclass(frozen=True)
class EncodeResult:
    encoded: torch.Tensor
    encoded_length: torch.Tensor
    layer_outputs: dict[int, torch.Tensor]


class ParakeetTDT06BV2(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.model: Any = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
        self.model.preprocessor.featurizer.use_grads = True
        tdt_kwargs = self.model.cfg.loss.tdt_kwargs
        self.tdt_loss = TDTLossNumba(
            blank=int(self.model.loss._blank),
            durations=[int(duration) for duration in tdt_kwargs.durations],
            reduction="none",
            sigma=float(tdt_kwargs.sigma),
            omega=float(tdt_kwargs.get("omega", 0.0)),
        )
        decoding_cfg = self.model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy_batch"
            decoding_cfg.compute_timestamps = False
            decoding_cfg.tdt_include_token_duration = True
        self.model.change_decoding_strategy(decoding_cfg)
        self.samples_per_encoder_frame = int(self.model.preprocessor.featurizer.hop_length) * int(
            self.model.encoder.subsampling_factor
        )
        self.eval()

    def train(
        self,
        mode: bool = True,
    ) -> nn.Module:
        super().train(mode)
        self.model.eval()
        self._set_decoder_backward_enabled(mode)
        for parameter in self.model.parameters():
            parameter.requires_grad_(self.training)
        return self

    def _set_decoder_backward_enabled(
        self,
        enabled: bool,
    ) -> None:
        if enabled:
            self.model.decoder.train()
            for module in self.model.decoder.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
        else:
            self.model.decoder.eval()

    def _empty_recognition(
        self,
    ) -> Recognition:
        return Recognition(
            text="",
            encoded_length=0,
            samples_per_encoder_frame=self.samples_per_encoder_frame,
            tokens=[],
            frame_token_index=[],
        )

    def _resolve_layer_indices(
        self,
        layers: Sequence[int] | None,
    ) -> list[int]:
        if not layers:
            return []
        n_layers = len(self.model.encoder.layers)
        resolved: list[int] = []
        seen: set[int] = set()
        for index in layers:
            layer_index = index + n_layers if index < 0 else int(index)
            if not 0 <= layer_index < n_layers:
                raise ValueError(f"encoder layer index {index} is out of range for {n_layers} layers")
            if layer_index not in seen:
                seen.add(layer_index)
                resolved.append(layer_index)
        return resolved

    def encode(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        layers: Sequence[int] | None = None,
    ) -> EncodeResult:
        layer_indices = self._resolve_layer_indices(layers)
        batch_size = waveforms.shape[0]
        valid = lengths > 0
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            encoded = waveforms.new_zeros(batch_size, int(self.model.encoder.d_model), 0) + waveforms.sum() * 0
            encoded_length = torch.zeros(batch_size, device=lengths.device, dtype=torch.int64)
            return EncodeResult(
                encoded=encoded,
                encoded_length=encoded_length,
                layer_outputs=dict.fromkeys(layer_indices, encoded),
            )
        idx = None
        waveforms_valid = waveforms
        lengths_valid = lengths
        if n_valid != batch_size:
            idx = valid.nonzero(as_tuple=False).squeeze(1)
            waveforms_valid = waveforms.index_select(0, idx)
            lengths_valid = lengths.index_select(0, idx)
        # AudioPreprocessor.forward is @torch.no_grad(); FilterbankFeatures keeps waveform grads.
        features, feature_lengths = FilterbankFeatures.forward(
            self.model.preprocessor.featurizer,
            waveforms_valid.to(dtype=torch.float32),
            lengths_valid,
        )
        layer_outputs_valid: dict[int, torch.Tensor] = {}
        handles = []

        def capture_layer(layer_index: int) -> Any:
            def hook(_module: nn.Module, _args: Any, out: torch.Tensor) -> None:
                layer_outputs_valid[layer_index] = out.transpose(1, 2)

            return hook

        try:
            for layer_index in layer_indices:
                handles.append(self.model.encoder.layers[layer_index].register_forward_hook(capture_layer(layer_index)))
            encoded_valid, encoded_length_valid = self.model.encoder(
                audio_signal=features,
                length=feature_lengths,
            )
        finally:
            for handle in handles:
                handle.remove()
        encoded_length_valid = encoded_length_valid.to(dtype=torch.int64)
        encoded_valid, encoded_length_valid = clip_encoded_time(encoded_valid, encoded_length_valid)
        cut = encoded_valid.size(-1)
        for layer_index, layer_tensor in list(layer_outputs_valid.items()):
            if layer_tensor.size(-1) != cut:
                layer_outputs_valid[layer_index] = layer_tensor[:, :, :cut]
        if idx is None:
            return EncodeResult(
                encoded=encoded_valid,
                encoded_length=encoded_length_valid,
                layer_outputs=layer_outputs_valid,
            )
        d_model = encoded_valid.shape[1]
        time = encoded_valid.shape[2]
        index = idx.view(-1, 1, 1).expand_as(encoded_valid)
        encoded = encoded_valid.new_zeros(batch_size, d_model, time).scatter(0, index, encoded_valid)
        encoded_length = encoded_length_valid.new_zeros(batch_size).scatter(0, idx, encoded_length_valid)
        layer_outputs = {}
        for layer_index, layer_tensor in layer_outputs_valid.items():
            layer_index_map = idx.view(-1, 1, 1).expand_as(layer_tensor)
            layer_outputs[layer_index] = layer_tensor.new_zeros(
                batch_size, layer_tensor.shape[1], layer_tensor.shape[2]
            ).scatter(0, layer_index_map, layer_tensor)
        return EncodeResult(encoded=encoded, encoded_length=encoded_length, layer_outputs=layer_outputs)

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        layers: Sequence[int] | None = None,
    ) -> EncodeResult:
        return self.encode(waveforms, lengths, layers)

    def _timed_recognition(
        self,
        prediction: Any,
        encoded_length: int,
    ) -> Recognition:
        def as_ints(value: Any) -> list[int]:
            if value is None:
                return []
            if torch.is_tensor(value):
                if value.numel() == 0:
                    return []
                value = value.detach().cpu().tolist()
            return [int(v) for v in value]

        token_ids = as_ints(prediction.y_sequence)
        starts = as_ints(prediction.timestamp)
        durations = as_ints(getattr(prediction, "token_duration", None))
        if not (len(token_ids) == len(starts) == len(durations)):
            raise RuntimeError(
                f"y_sequence/timestamp/token_duration lengths differ: {len(token_ids)}, {len(starts)}, {len(durations)}"
            )
        blank_id = int(self.model.loss._blank)
        kept = []
        for token_id, start, duration in zip(token_ids, starts, durations):
            if token_id >= blank_id:
                continue
            start_offset = max(int(start), 0)
            if start_offset >= encoded_length:
                continue
            end_offset = min(start_offset + max(int(duration), 1), encoded_length)
            kept.append((token_id, start_offset, end_offset))
        token_texts = self.model.decoding.decode_ids_to_tokens([token_id for token_id, _, _ in kept])
        tokens = [
            Token(
                token_id=token_id,
                token=token_text,
                start_offset=start_offset,
                end_offset=end_offset,
            )
            for (token_id, start_offset, end_offset), token_text in zip(kept, token_texts)
        ]
        frame_token_index = [-1] * encoded_length
        for token_index, token in enumerate(tokens):
            for frame in range(token.start_offset, token.end_offset):
                frame_token_index[frame] = token_index
        return Recognition(
            text=prediction.text,
            encoded_length=encoded_length,
            samples_per_encoder_frame=self.samples_per_encoder_frame,
            tokens=tokens,
            frame_token_index=frame_token_index,
        )

    def recognize_encoded(
        self,
        encoded: torch.Tensor,
        encoded_length: torch.Tensor,
    ) -> list[Recognition]:
        batch_size = encoded.shape[0]
        valid = encoded_length > 0
        if not bool(valid.any()):
            return [self._empty_recognition() for _ in range(batch_size)]
        idx = None
        encoded_valid = encoded
        encoded_length_valid = encoded_length
        if not bool(valid.all()):
            idx = valid.nonzero(as_tuple=False).squeeze(1)
            encoded_valid = encoded.index_select(0, idx)
            encoded_length_valid = encoded_length.index_select(0, idx)
        encoded_lengths = encoded_length_valid.detach().cpu().tolist()
        decoder_training = self.model.decoder.training
        self.model.decoder.eval()
        with torch.no_grad():
            predictions = self.model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded_valid.detach(),
                encoded_lengths=encoded_length_valid,
                return_hypotheses=True,
            )
        self._set_decoder_backward_enabled(decoder_training)
        timed = [
            self._timed_recognition(prediction, int(encoded_lengths[i])) for i, prediction in enumerate(predictions)
        ]
        if idx is None:
            return timed
        results = [self._empty_recognition() for _ in range(batch_size)]
        for source, destination in enumerate(idx.tolist()):
            results[destination] = timed[source]
        return results

    def recognize(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> list[Recognition]:
        with torch.no_grad():
            encoded = self.encode(waveforms, lengths)
        return self.recognize_encoded(encoded.encoded, encoded.encoded_length)

    def ids_to_text(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        return self.model.tokenizer.ids_to_text(list(token_ids))

    def loss(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        texts: Sequence[str],
    ) -> torch.Tensor:
        return self.loss_from_ids(waveforms, lengths, [self.model.tokenizer.text_to_ids(text) for text in texts])

    def loss_from_ids(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        token_ids: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        batch_size = waveforms.shape[0]
        if len(token_ids) != batch_size:
            raise ValueError(f"token_ids length {len(token_ids)} != batch {batch_size}")
        nonempty = [len(ids) > 0 for ids in token_ids]
        valid = (lengths > 0) & torch.tensor(nonempty, device=lengths.device, dtype=torch.bool)
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            return waveforms.new_zeros(batch_size) + waveforms.sum() * 0
        idx = None
        token_id_lists = list(token_ids)
        if n_valid != batch_size:
            idx = valid.nonzero(as_tuple=False).squeeze(1)
            waveforms = waveforms.index_select(0, idx)
            lengths = lengths.index_select(0, idx)
            token_id_lists = [token_id_lists[int(i)] for i in idx.tolist()]
        target_lengths_cpu = torch.tensor([len(ids) for ids in token_id_lists], dtype=torch.int64)
        max_length = int(target_lengths_cpu.max().item())
        targets_cpu = torch.zeros((len(token_id_lists), max_length), dtype=torch.int64)
        for i, ids in enumerate(token_id_lists):
            if ids:
                targets_cpu[i, : len(ids)] = torch.tensor(ids, dtype=torch.int64)
        target_lengths = target_lengths_cpu.to(device=waveforms.device, non_blocking=True)
        targets = targets_cpu.to(device=waveforms.device, non_blocking=True)
        encoded = self.encode(waveforms, lengths)
        decoder_outputs, _, _ = self.model.decoder(
            targets=targets,
            target_length=target_lengths,
        )
        logits = self.model.joint.joint(
            encoded.encoded.transpose(1, 2),
            decoder_outputs.transpose(1, 2),
        )
        nll_valid = self.tdt_loss(
            acts=logits,
            labels=targets.to(dtype=torch.int64),
            act_lens=encoded.encoded_length.to(dtype=torch.int64),
            label_lens=target_lengths.to(dtype=torch.int64),
        )
        if idx is None:
            return nll_valid
        return nll_valid.new_zeros(batch_size).scatter(0, idx, nll_valid)
