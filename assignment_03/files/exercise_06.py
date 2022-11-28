import numpy as np
from utils import plot_time_and_freq, check_arrays

# Your solution starts here.
def dft_1d(y_time):
    """Transforms a given signal from time to frequency domain.

    Args:
        y_time: A numpy array with shape (n,) representing the imput signal in time domain.

    Returns:
        A numpy array with shape (n,) representing the signal in frequency domain.
    """
    n, = y_time.shape
    
    y_freq = np.zeros_like(y_time) # TODO: Exercise 6a)
    x = np.arange(n)
    u = x.reshape((n,1))
    e = np.exp(-2j * np.pi * u * x / n)

    y_freq = np.dot(e,y_time)

    return y_freq

def idft_1d(y_freq):
    """Transforms a given signal from frequency to time domain.

    Args:
        y_freq: A numpy array with shape (n,) representing the imput signal in frequency domain.

    Returns:
        A numpy array with shape (n,) representing the signal in time domain.
    """
    n, = y_freq.shape
    
    y_time = np.zeros_like(y_freq) # TODO: Exercise 6b)
    x = np.arange(n)
    u = x.reshape((n,1))
    e = np.exp(2j * np.pi * u * x / n)

    y_time = 1/n*np.dot(e,y_freq)

    return y_time

def dft_1d_denoise(y_time, threshold=0.25):
    """Applies a threshold to filter out frequencies with low amplitudes.
    
    Args:
        y_time: A numpy array with shape (n,) representing the imput signal in time domain.

    Returns:
        A numpy array with shape (n,) representing the denoised signal in time domain.
    """
    n, = y_time.shape
    y_freq = dft_1d(y_time)
    # TODO: Exercise 6c)
    psd = y_freq * np.conj(y_freq)/n
    y_freq = np.where(psd/np.sqrt(n)<threshold, 0, y_freq)

    return idft_1d(y_freq)

def box_filter_1d_time(y_time, w):
    """Applies the mean filter of size 2*w+1 to the input signal.
    
    This function uses convolution to apply the mean filter in time domain.
    

    Args:
        y_time: A numpy array with shape (n,) representing the imput signal in time domain.

    Returns:
        A numpy array with shape (n,) representing the filtered signal in time domain.
    """
    # pad signal periodically 
    y_time_padded = np.concatenate((y_time[-w:], y_time, y_time[:w]))
    n, = y_time.shape
    
    y_time_filtered = np.zeros_like(y_time) # TODO: Exercise 6d)

    for i in range(n):
        y_time_filtered[i] = np.mean(y_time_padded[i:2*w+1+i])

    
    return y_time_filtered

def box_filter_1d_freq(y_time, w):
    """Applies the mean filter of size 2*w+1 to the input signal.
    
    This function uses dft and exploits the convolution theorem to apply the mean filter in frequency domain.
    

    Args:
        y_time: A numpy array with shape (n,) representing the imput signal in time domain.

    Returns:
        A numpy array with shape (n,) representing the filtered signal in time domain.
    """
    y_freq = dft_1d(y_time)
    
    y_freq_filtered = np.zeros_like(y_freq) # TODO: Exercise 6d)
    kernel = np.full(2*w+1, 1/(2*w+1), dtype=np.complex128)
    n, = y_time.shape
    n = (n-(2*w+1))/2
    filter = np.concatenate((np.zeros(int(n)), kernel, np.zeros(int(n))))
    y_freq_filter = dft_1d(filter)
    y_freq_filtered = y_freq*y_freq_filter
    
    return idft_1d(y_freq_filtered)
# Your solution ends here.

def main():
    """Do not change this function at all."""
    t = np.linspace(0,2*np.pi, 128, dtype=np.complex128)[:-1]
    y_time = np.sin(2*t) + np.cos(11*t) + 1
    y_freq = dft_1d(y_time)
    plot_time_and_freq(t, y_time, y_freq, 'Original Signal')
    
    rng = np.random.default_rng(313373)

    y_time_noisy = y_time + 0.5 * rng.standard_normal(y_time.shape)
    y_freq_noisy = dft_1d(y_time_noisy)
    plot_time_and_freq(t, y_time_noisy, y_freq_noisy, 'Noisy Signal')
    
    y_time_denoised = dft_1d_denoise(y_time_noisy, threshold=0.25)
    y_freq_denoised = dft_1d(y_time_denoised)
    plot_time_and_freq(t, y_time_denoised, y_freq_denoised, 'Denoised Signal')
    
    y_filtered1_time = box_filter_1d_time(y_time, w=5)
    y_filtered1_freq = dft_1d(y_filtered1_time)
    plot_time_and_freq(t, y_filtered1_time, y_filtered1_freq, 'Filtered Signal (Time)')
    
    y_filtered2_time = box_filter_1d_freq(y_time, w=5)
    y_filtered2_freq = dft_1d(y_filtered2_time)
    plot_time_and_freq(t, y_filtered2_time, y_filtered2_freq, 'Filtered Signal (Frequency)')
    assets = np.load('.assets.npz')
    check_arrays(
        'Exercise 6',
        [
            'a) Discrete Fourier Transform',
            'b) Inverse Discrete Fourier Transform',
            'c) Signal Denoising',
            'd) Box-Filter (Time)',
            'd) Box-Filter (Frequency)'
        ],
        [
            y_freq,
            idft_1d(y_freq),
            y_time_denoised,
            np.abs(y_filtered1_freq),
            np.abs(y_filtered2_freq),
        ],
        [
            np.fft.fft(y_time),
            y_time,
            assets['y_time_denoised'],
            assets['y_filtered1_freq_abs'],
            assets['y_filtered2_freq_abs']
        ],
    )
    input('Press ENTER to quit.')

if __name__ == '__main__':
    main()
