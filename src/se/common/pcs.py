import librosa
import numpy as np
import scipy

PCS400 = np.ones(201)
PCS400[0:3] = 1
PCS400[3:5] = 1.070175439
PCS400[5:8] = 1.182456140
PCS400[8:10] = 1.287719298
PCS400[10:110] = 1.4
PCS400[110:130] = 1.322807018
PCS400[130:160] = 1.238596491
PCS400[160:190] = 1.161403509
PCS400[190:202] = 1.077192982


def _magnitude_phase(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    signal_length = signal.shape[0]
    n_fft = 400
    padded = librosa.util.fix_length(signal, size=signal_length + n_fft // 2)
    spec = librosa.stft(
        padded,
        n_fft=400,
        hop_length=100,
        win_length=400,
        window=scipy.signal.windows.hamming(400),
    )
    mag = PCS400 * np.transpose(np.log1p(np.abs(spec)), (1, 0))
    phase = np.angle(spec)
    return np.transpose(mag, (1, 0)), phase, signal_length


def _to_wav(mag: np.ndarray, phase: np.ndarray, signal_length: int) -> np.ndarray:
    mag = np.expm1(mag)
    reconstructed = mag * np.exp(1j * phase)
    return librosa.istft(
        reconstructed,
        hop_length=100,
        win_length=400,
        window=scipy.signal.windows.hamming(400),
        length=signal_length,
    )


def cal_pcs(signal_wav: np.ndarray) -> np.ndarray:
    mag, phase, signal_length = _magnitude_phase(signal_wav.squeeze())
    pcs = _to_wav(mag, phase, signal_length)
    return pcs / np.max(np.abs(pcs))
