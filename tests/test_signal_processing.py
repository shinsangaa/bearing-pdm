import unittest
import numpy as np
from src.signal_processing import hann_fft, stft_spectrogram, signal_features


class SignalProcessingTests(unittest.TestCase):

    def setUp(self):
        self.fs = 4000
        self.x = 3 * np.sin(2 * np.pi * 125 * np.arange(12000) / self.fs)

    def test_fft_frequency_amplitude_and_offset(self):
        freq, amplitude = hann_fft(self.x + 10, self.fs)
        self.assertEqual(freq[np.argmax(amplitude)], 125)
        self.assertAlmostEqual(float(amplitude.max()), 3, places=5)

    def test_stft_frequency_amplitude_and_boundaries(self):
        freq, times, db, settings = stft_spectrogram(self.x, self.fs)
        np.testing.assert_array_equal(freq[np.argmax(db, axis=0)], 125)
        np.testing.assert_allclose(db.max(axis=0), 20 * np.log10(3), atol=0.01)
        self.assertTrue(np.all(times * self.fs >= 512))
        self.assertTrue(np.all(times * self.fs <= len(self.x) - 512))
        self.assertEqual(db.shape, (513, 43))
        self.assertEqual(settings['hop'], 256)

    def test_invalid_stft_settings(self):
        for size, overlap in ((0, 0.75), (20000, 0.75), (1024, 1.0)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                stft_spectrogram(self.x, self.fs, size, overlap)


class PreprocessingTests(unittest.TestCase):

    def test_known_sine(self):
        x = 3 * np.sin(2 * np.pi * 50 * np.arange(2000) / 2000)
        result = signal_features(x, 2000)
        self.assertAlmostEqual(result['rms'], 3 / np.sqrt(2))
        self.assertAlmostEqual(result['kurtosis'], 1.5)
        self.assertAlmostEqual(result['crest_factor'], np.sqrt(2))
        self.assertEqual(result['fft_peak_1_hz'], 50)
        self.assertAlmostEqual(result['fft_peak_1_amplitude'], 3, places=5)

    def test_constant(self):
        result = signal_features(np.zeros(20), 20)
        self.assertTrue(result['constant_signal'])
        self.assertIsNone(result['crest_factor'])
        self.assertIsNone(result['kurtosis'])
