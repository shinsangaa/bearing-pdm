"""진동 신호의 특징, FFT/STFT와 그림을 계산한다. 데이터 파일은 읽지 않는다."""
import math
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT


def hann_fft(x, fs):
    """한 채널과 샘플링 주파수를 받아 단측 FFT의 주파수·진폭 배열을 반환한다."""
    if len(x) < 3:
        raise ValueError("한 윈도 FFT에는 최소 3개 샘플이 필요합니다.")
    # 평균 제거로 직류 성분(DC)을 줄인 뒤 한 윈도(Hann Window)로 누설을 완화한다.
    window = np.hanning(len(x))
    centered = x - np.mean(x)
    amplitude = 2 * np.abs(np.fft.rfft(centered * window)) / window.sum()
    # 음의 주파수 짝이 없는 0Hz와 나이퀴스트 주파수는 두 배 보정을 되돌린다.
    amplitude[0] /= 2
    if len(x) % 2 == 0:
        amplitude[-1] /= 2
    return np.fft.rfftfreq(len(x), d=1 / fs), amplitude


def stft_spectrogram(x, fs, nperseg=1024, overlap=0.75):
    """한 채널을 짧게 나누어 시간·주파수·진폭 데시벨 배열과 설정을 반환한다.

    단시간 푸리에 변환(STFT)은 한 윈도를 일정 간격으로 옮기며 FFT를 반복한다.
    창 길이를 늘리면 주파수 간격이 좁아지지만 빠른 시간 변화는 구분하기 어려워진다.
    """
    if not 8 <= nperseg <= len(x):
        raise ValueError(f"--nperseg는 8 이상, 샘플 수({len(x)}) 이하여야 합니다.")
    if not 0 <= overlap < 1:
        raise ValueError("--overlap은 0 이상 1 미만이어야 합니다.")
    hop = max(1, round(nperseg * (1 - overlap)))
    # 양의 주파수 쪽 진폭을 보정한다. fs에는 고정값 대신 CSV 메타데이터 값을 넣는다.
    transform = ShortTimeFFT(np.hanning(nperseg), hop, fs,
                             fft_mode="onesided2X", scale_to="magnitude")
    # 창 전체가 실제 신호 안에 들어가는 구간만 선택해 경계의 0 채움 효과를 제외한다.
    start = math.ceil((nperseg // 2) / hop)
    stop = (len(x) - (nperseg - nperseg // 2)) // hop + 1
    if start >= stop:
        raise ValueError("완전한 STFT 창이 없습니다. --nperseg를 줄여 주세요.")
    spectrum = transform.stft(x - np.mean(x), p0=start, p1=stop)
    times = transform.t(len(x), p0=start, p1=stop)
    # 두 상태 모두 진폭 1을 기준으로 변환한다. 파일별 최댓값으로 정규화하지 않는다.
    # 센서 단위가 확정되지 않았으므로 dB는 물리적인 절대 소음 수준을 뜻하지 않는다.
    db = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-12))
    settings = dict(nperseg=nperseg, hop=hop, overlap_actual=1 - hop / nperseg,
                    window_seconds=nperseg / fs, frequency_step_hz=fs / nperseg,
                    time_step_seconds=hop / fs, frames=len(times))
    return transform.f, times, db, settings


def configure_font():
    """설치된 한글 글꼴을 찾아 제목과 축 이름의 글자 깨짐을 줄인다."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    raise RuntimeError("한글 글꼴이 없습니다. 맑은 고딕 등 한글 글꼴을 설치해 주세요.")


def signal_figure(signal, fs=20000, title="NASA IMS", nperseg=1024):
    """같은 실제 측정 신호의 파형·FFT·STFT를 표시한다."""
    configure_font()
    freq, amplitude = hann_fft(signal, fs)
    sfreq, times, db, settings = stft_spectrogram(signal, fs, nperseg)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), layout="constrained")
    axes[0].plot(np.arange(len(signal)) / fs, signal, linewidth=.5)
    axes[0].set(xlabel="측정 내 시간 (s)", ylabel="원본 진폭", title=title)
    axes[1].plot(freq, amplitude, linewidth=.6)
    axes[1].set(xlabel="주파수 (Hz)", ylabel="Hann FFT 진폭")
    image = axes[2].pcolormesh(times, sfreq, db, shading="auto", cmap="magma")
    axes[2].set(xlabel="측정 내 시간 (s)", ylabel="주파수 (Hz)", title="STFT: 원본 진폭 1 기준 dB")
    fig.colorbar(image, ax=axes[2], label="dB")
    return fig, settings





def signal_features(x, fs):
    """한 채널의 신호 x와 샘플링 주파수 fs(Hz)를 받아 특징 딕셔너리를 반환한다.

    x는 시간 열을 제외한 1차원 측정값 배열이다. 호출 전에 결측값과 시간 간격을 검사한다.
    시간 영역의 크기·분포 특징과 고속 푸리에 변환(Fast Fourier Transform, FFT)의 주파수 특징을 계산한다.
    """
    # 실효값(Root Mean Square, RMS): 제곱 → 평균 → 제곱근 순서로 신호의 전체적인 크기를 구한다.
    rms = float(np.sqrt(np.mean(x * x)))
    # 모집단 표준편차(Standard Deviation): 평균 주변의 흔들림을 계산한다.
    std = float(np.std(x, ddof=0))
    # 절대 최댓값(Peak): 양수·음수 중 부호와 관계없이 가장 큰 크기를 구한다.
    peak = float(np.max(np.abs(x)))
    # 평균을 빼서 일정한 오프셋인 직류 성분(DC)을 줄인다. RMS는 원래 신호로 계산한다.
    centered = x - np.mean(x)
    # 첨도(Kurtosis): 표준화한 값의 4제곱 평균이다. 정규분포 기준은 3이다.
    # 상수 신호는 표준편차가 0이므로 나눗셈 대신 None을 반환한다.
    kurtosis = float(np.mean((centered / std) ** 4)) if std else None
    # 한 윈도(Hann Window): 양 끝을 완만하게 줄여 주파수 성분이 주변으로 퍼지는 누설을 줄인다.
    # 길이가 2이면 한 윈도의 합이 0이 되므로 모든 값이 1인 사각 창을 사용한다.
    window = np.hanning(len(x)) if len(x) > 2 else np.ones(len(x))
    # 실수 입력용 FFT로 양의 주파수 쪽을 구하고, 절댓값으로 각 성분의 크기를 계산한다.
    # 단측 스펙트럼의 두 배 보정과 창의 합을 이용해 진폭을 보정한다. 0Hz는 순위에서 제외한다.
    spectrum = np.abs(np.fft.rfft(centered * window)) * 2 / window.sum()
    if len(x) % 2 == 0: # 짝수 길이 신호에는 나이퀴스트 주파수 bin이 있다
        spectrum[-1] /= 2 # 이 bin은 음의 주파수 짝이 없어 두 배 보정을 되돌린다
    freq = np.fft.rfftfreq(len(x), 1 / fs) # 각 FFT bin에 대응하는 Hz 값을 만든다
    # 0Hz를 뺀 진폭을 내림차순으로 정렬한다. +1은 원래 배열의 위치로 되돌리는 보정이다.
    # 상위 3개 주파수 구간(bin)은 한 봉우리의 이웃일 수도 있어 별개의 고장 성분을 뜻하지 않는다.
    indices = np.argsort(spectrum[1:])[::-1][:3] + 1 if std else []
    # 진폭 제곱을 주파수 중심의 가중치로 사용한다. 별도로 정규화한 전력 스펙트럼 밀도는 아니다.
    power = spectrum[1:] ** 2
    # 파고율(Crest Factor)은 Peak/RMS, 주파수 중심은 진폭 제곱으로 가중한 평균 주파수다.
    # 평균에서 8표준편차를 넘는 값의 개수는 검토용으로 기록하며 측정값을 삭제하지 않는다.
    result = dict(rms=rms, std=std, peak=peak, kurtosis=kurtosis,
                  crest_factor=peak / rms if rms else None,
                  fft_centroid_hz=float(np.sum(freq[1:] * power) / power.sum()) if power.sum() else None,
                  constant_signal=std == 0,
                  outlier_8std_count=int(np.count_nonzero(np.abs(centered) > 8 * std)) if std else 0)
    for rank in range(3):
        # 사용할 주파수 성분이 없으면 해당 주파수와 진폭을 None으로 남긴다.
        idx = int(indices[rank]) if rank < len(indices) else None
        result[f"fft_peak_{rank + 1}_hz"] = float(freq[idx]) if idx is not None else None
        result[f"fft_peak_{rank + 1}_amplitude"] = float(spectrum[idx]) if idx is not None else None
    return result
