# Audio Configuration
sample_rate = 16000
window_size = 0.025
window_stride = 0.01
window = 'hamming'
noise_dir = None
noise_prob = 0.4
noise_min = 0.0
noise_max = 0.5
audio_conf = dict(sample_rate=sample_rate,
        window_size=window_size,
        window_stride=window_stride,
        window=window,
        noise_dir=noise_dir,
        noise_prob=noise_prob,
        noise_levels=(noise_min, noise_max))

rDot_config = {
        'train_manifest': '/home/muncok/DL/
