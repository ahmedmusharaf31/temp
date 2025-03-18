R = img(:,:,1); 
G = img(:,:,2); 
B = img(:,:,3);

[X, map] = rgb2ind(img, 256);


RGB_img = ind2rgb(X, map);

img = imread('filename'); 
imshow(img); 
imwrite(img, 'filename');

R = img(:,:,1); 
G = img(:,:,2); 
B = img(:,:,3);


gray = rgb2gray(img); 
[X, map] = rgb2ind(img, 256); 
RGB_img = ind2rgb(X, map);


clc; clear; close all; 
% Load an RGB image 
img = imread('peppers.png'); 
% Display the original image 
figure; imshow(img); title('Original RGB Image'); 
% Extract individual R, G, and B channels 
R = img(:,:,1); G = img(:,:,2); B = img(:,:,3); 
% Display the separate color channels 
figure; 
subplot(1,3,1), imshow(R), title('Red Channel'); 
subplot(1,3,2), imshow(G), title('Green Channel'); 
subplot(1,3,3), imshow(B), title('Blue Channel'); 
% Convert RGB to Grayscale 
gray_img = rgb2gray(img);


% Convert RGB to Indexed with 256 colors 
[X, map] = rgb2ind(img, 256); 
% Convert Indexed back to RGB 
RGB_img = ind2rgb(X, map); 
% Display grayscale, indexed, and reconstructed images 
figure; 
subplot(2,2,1), imshow(img), title('Original RGB Image'); 
subplot(2,2,2), imshow(gray_img), title('Grayscale Image'); 
subplot(2,2,3), imshow(X, map), title('Indexed Image'); 
subplot(2,2,4), imshow(RGB_img), title('Reconstructed RGB from Indexed'); 
% Display the colormap used 
figure; colormap(map); colorbar; title('Colormap Used in Indexed Image');


YCbCr = rgb2ycbcr(img); 
RGB_reconstructed = ycbcr2rgb(YCbCr);


HSV = rgb2hsv(img); 
RGB_reconstructed = hsv2rgb(HSV);