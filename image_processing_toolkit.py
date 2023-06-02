import cv2
import numpy as np
import os
from cv2_rolling_ball import subtract_background_rolling_ball


def apply_gaussian_blur(image, kernel_size):
    """
    Apply Gaussian blur to the input image.

    Args:
    image: Input image
    kernel_size: Kernel size, must be a positive odd number

    Returns:
    Blurred image
    """
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("Kernel size must be a positive odd number.")

    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image


def apply_mean_blur(image, kernel_size):
    """
    Apply mean blur to the input image.

    Args:
    image: Input image
    kernel_size: Kernel size, must be a positive odd number

    Returns:
    Blurred image
    """
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("Kernel size must be a positive odd number.")

    blurred_image = cv2.blur(image, (kernel_size, kernel_size))
    return blurred_image


def apply_denoising(image, denoising_method):
    """
    Apply denoising algorithm to the input image.

    Args:
    image: Input image
    denoising_method: Denoising method name, supports "gaussian" and "mean"

    Returns:
    Denoised image
    """
    if denoising_method == "gaussian":
        denoised_image = apply_gaussian_blur(image, 3)
    elif denoising_method == "mean":
        denoised_image = apply_mean_blur(image, 3)
    else:
        raise ValueError("Invalid denoising method. Supported methods: 'gaussian', 'mean'.")

    return denoised_image


def extract_dark_regions(image):
    """
    Extract dark regions from the input image.

    Args:
    image: Input image

    Returns:
    Extracted dark regions (binary image)
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, 0, 240, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return threshold_image


def add_background(image):
    """
    Add a light gray background to the input image.

    Args:
    image: Input image

    Returns:
    Image with background
    """
    background = np.ones_like(image, dtype=np.uint8) * 230
    blended_image = cv2.addWeighted(image, 0.55, background, 0.45, 10)
    return blended_image


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                               None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]),
                                         None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def merge_videos(input_paths, output_path):
    cap = cv2.VideoCapture(input_paths[0])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    cap.release()

    for input_path in input_paths:
        cap = cv2.VideoCapture(input_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()


def rolling_ball_single_images(img_path, save_path, ratio, radius=40, show_window=False):
    filename_with_extension = os.path.basename(img_path)
    file_name, _ = os.path.splitext(filename_with_extension)
    img = cv2.imread(img_path, 0)
    img_a = img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)
    final_img, background_img = subtract_background_rolling_ball(img, radius, light_background=True,
                                                                  use_paraboloid=True, do_presmooth=True)
    equalized_image = cv2.equalizeHist(img_a)
    equalized_image_b = cv2.equalizeHist(final_img)
    if ratio > 0:
        imagestack = stackImages(ratio, ([img_a, background_img, final_img],
                                         [equalized_image, equalized_image_b, clahe_img]))
        print("writing the processing images")
        cv2.imwrite(f'{save_path}/{file_name}_treat_process.png', imagestack)

    print("rolling ball on your image\nprocessing>>>%s" % file_name)
    cv2.imwrite(f'{save_path}/{file_name}_treated.png', final_img)

    if show_window:
        cv2.imshow('imagestack', imagestack)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
