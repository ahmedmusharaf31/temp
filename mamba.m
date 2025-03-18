% Read and convert image to grayscale
img = im2double(rgb2gray(imread('sample.png')));

% Apply Fourier Transform
F = fft2(img);
F_shifted = fftshift(F);

% Add periodic noise (simulate sine wave noise)
[M, N] = size(img);
[x, y] = meshgrid(1:N, 1:M);
noise = sin(2 * pi * 10 * x / N) + sin(2 * pi * 10 * y / M); % Periodic noise
F_noisy = F_shifted + fftshift(fft2(noise)); % Add noise in frequency domain

% Create Notch Filter to remove noise
D0 = 10; % Cutoff frequency
[u, v] = meshgrid(-N/2:N/2-1, -M/2:M/2-1);
D1 = sqrt((u - 10).^2 + (v - 10).^2); % Centered at (10,10)
D2 = sqrt((u + 10).^2 + (v + 10).^2); % Symmetric point
notch_filter = 1 - exp(-((D1.^2 + D2.^2) / (2 * (D0^2)))); % Gaussian Notch

% Apply Notch Filter
F_filtered = F_noisy .* notch_filter;

% Inverse Fourier Transform
img_filtered = real(ifft2(ifftshift(F_filtered)));

% Display Results
figure;
subplot(1,3,1), imshow(img, []), title('Original Image');
subplot(1,3,2), imshow(log(1 + abs(F_noisy)), []), title('With Periodic Noise');
subplot(1,3,3), imshow(img_filtered, []), title('After Notch Filtering');
