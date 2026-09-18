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


class FrozenParakeetTDT06BV2(nn.Module):
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

    def _encode_valid(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        layer_indices: Sequence[int],
    ) -> EncodeResult:
        features, feature_lengths = FilterbankFeatures.forward(
            self.model.preprocessor.featurizer,
            waveforms.to(dtype=torch.float32),
            lengths,
        )
        layer_outputs: dict[int, torch.Tensor] = {}
        handles = []

        def capture_layer(layer_index: int) -> Any:
            def hook(_module: nn.Module, _args: Any, out: torch.Tensor) -> None:
                layer_outputs[layer_index] = out.transpose(1, 2)

            return hook

        try:
            for layer_index in layer_indices:
                handles.append(self.model.encoder.layers[layer_index].register_forward_hook(capture_layer(layer_index)))
            encoded, encoded_length = self.model.encoder(
                audio_signal=features,
                length=feature_lengths,
            )
        finally:
            for handle in handles:
                handle.remove()
        return EncodeResult(
            encoded=encoded,
            encoded_length=encoded_length.to(dtype=torch.int64),
            layer_outputs=layer_outputs,
        )

    def _empty_encoded(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = waveforms.shape[0]
        d_model = int(self.model.encoder.d_model)
        encoded = waveforms.new_zeros(batch_size, d_model, 0) + waveforms.sum() * 0
        encoded_length = torch.zeros(batch_size, device=lengths.device, dtype=torch.int64)
        return encoded, encoded_length

    def _scatter_encoded(
        self,
        encoded_valid: torch.Tensor,
        encoded_length_valid: torch.Tensor,
        idx: torch.Tensor | None,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if idx is None:
            return encoded_valid, encoded_length_valid
        d_model = encoded_valid.shape[1]
        time = encoded_valid.shape[2]
        index = idx.view(-1, 1, 1).expand_as(encoded_valid)
        encoded = encoded_valid.new_zeros(batch_size, d_model, time).scatter(0, index, encoded_valid)
        encoded_length = encoded_length_valid.new_zeros(batch_size).scatter(0, idx, encoded_length_valid)
        return encoded, encoded_length

    def encode(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        layers: Sequence[int] | None = None,
    ) -> EncodeResult:
        layer_indices = self._resolve_layer_indices(layers)
        batch_size = waveforms.shape[0]
        valid = lengths > 0
        if not bool(valid.any()):
            encoded, encoded_length = self._empty_encoded(waveforms, lengths)
            return EncodeResult(
                encoded=encoded,
                encoded_length=encoded_length,
                layer_outputs=dict.fromkeys(layer_indices, encoded),
            )
        idx = None
        waveforms_valid = waveforms
        lengths_valid = lengths
        if not bool(valid.all()):
            idx = valid.nonzero(as_tuple=False).squeeze(1)
            waveforms_valid = waveforms.index_select(0, idx)
            lengths_valid = lengths.index_select(0, idx)
        encoded_valid = self._encode_valid(waveforms_valid, lengths_valid, layer_indices)
        encoded, encoded_length = self._scatter_encoded(
            encoded_valid.encoded,
            encoded_valid.encoded_length,
            idx,
            batch_size,
        )
        layer_outputs = {}
        for layer_index, layer_tensor in encoded_valid.layer_outputs.items():
            layer_outputs[layer_index], _ = self._scatter_encoded(
                layer_tensor,
                encoded_valid.encoded_length,
                idx,
                batch_size,
            )
        return EncodeResult(encoded=encoded, encoded_length=encoded_length, layer_outputs=layer_outputs)

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        layers: Sequence[int] | None = None,
    ) -> EncodeResult:
        return self.encode(waveforms, lengths, layers)

    def _to_int_list(
        self,
        value: Any,
    ) -> list[int]:
        if value is None:
            return []
        if torch.is_tensor(value):
            if value.numel() == 0:
                return []
            return [int(v) for v in value.detach().cpu().tolist()]
        return [int(v) for v in value]

    def _timed_recognition(
        self,
        prediction: Any,
        encoded_length: int,
        return_text: bool = True,
    ) -> Recognition:
        token_ids = self._to_int_list(prediction.y_sequence)
        starts = self._to_int_list(prediction.timestamp)
        durations = self._to_int_list(getattr(prediction, "token_duration", None))
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
        if return_text:
            token_texts = self.model.decoding.decode_ids_to_tokens([token_id for token_id, _, _ in kept])
            text = prediction.text
        else:
            token_texts = [""] * len(kept)
            text = ""
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
            text=text,
            encoded_length=encoded_length,
            samples_per_encoder_frame=self.samples_per_encoder_frame,
            tokens=tokens,
            frame_token_index=frame_token_index,
        )

    def _scatter_recognition(
        self,
        timed: list[Recognition],
        idx: torch.Tensor | None,
        batch_size: int,
    ) -> list[Recognition]:
        if idx is None:
            return timed
        results = [self._empty_recognition() for _ in range(batch_size)]
        for source, destination in enumerate(idx.tolist()):
            results[destination] = timed[source]
        return results

    def _rnnt_predictions(
        self,
        encoded: torch.Tensor,
        encoded_length: torch.Tensor,
    ) -> Any:
        decoder_training = self.model.decoder.training
        self.model.decoder.eval()
        with torch.no_grad():
            predictions = self.model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded.detach(),
                encoded_lengths=encoded_length,
                return_hypotheses=True,
            )
        self._set_decoder_backward_enabled(decoder_training)
        return predictions

    def recognize_encoded(
        self,
        encoded: torch.Tensor,
        encoded_length: torch.Tensor,
        *,
        return_text: bool = True,
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
        predictions = self._rnnt_predictions(encoded_valid, encoded_length_valid)
        timed = [
            self._timed_recognition(prediction, int(encoded_lengths[i]), return_text=return_text)
            for i, prediction in enumerate(predictions)
        ]
        return self._scatter_recognition(timed, idx, batch_size)

    def recognize(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        *,
        return_text: bool = True,
    ) -> list[Recognition]:
        with torch.no_grad():
            encoded = self.encode(waveforms, lengths)
        return self.recognize_encoded(encoded.encoded, encoded.encoded_length, return_text=return_text)

    def _pad_token_ids(
        self,
        token_ids: Sequence[Sequence[int]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_lengths = torch.tensor([len(ids) for ids in token_ids], device=device, dtype=torch.int64)
        max_length = int(target_lengths.max().item())
        padded = torch.zeros((len(token_ids), max_length), device=device, dtype=torch.int64)
        for i, ids in enumerate(token_ids):
            if ids:
                padded[i, : len(ids)] = torch.tensor(ids, device=device, dtype=torch.int64)
        return padded, target_lengths

    def loss(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        texts: Sequence[str],
    ) -> torch.Tensor:
        batch_size = waveforms.shape[0]
        token_id_lists = [self.model.tokenizer.text_to_ids(text) for text in texts]
        valid = (lengths > 0) & torch.tensor(
            [len(ids) > 0 for ids in token_id_lists],
            device=lengths.device,
            dtype=torch.bool,
        )
        if not bool(valid.any()):
            return waveforms.new_zeros(batch_size) + waveforms.sum() * 0
        idx = None
        if not bool(valid.all()):
            idx = valid.nonzero(as_tuple=False).squeeze(1)
            waveforms = waveforms.index_select(0, idx)
            lengths = lengths.index_select(0, idx)
            token_id_lists = [token_id_lists[int(i)] for i in idx.tolist()]
        targets, target_lengths = self._pad_token_ids(token_id_lists, waveforms.device)
        encoded_valid = self._encode_valid(waveforms, lengths, ())
        decoder_outputs, _, _ = self.model.decoder(
            targets=targets,
            target_length=target_lengths,
        )
        logits = self.model.joint.joint(
            encoded_valid.encoded.transpose(1, 2),
            decoder_outputs.transpose(1, 2),
        )
        nll_valid = self.tdt_loss(
            acts=logits,
            labels=targets.to(dtype=torch.int64),
            act_lens=encoded_valid.encoded_length.to(dtype=torch.int64),
            label_lens=target_lengths.to(dtype=torch.int64),
        )
        if idx is None:
            return nll_valid
        return nll_valid.new_zeros(batch_size).scatter(0, idx, nll_valid)
