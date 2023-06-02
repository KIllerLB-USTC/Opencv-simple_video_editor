import cv2
import numpy as np
import os
from cv2_rolling_ball import subtract_background_rolling_ball
def apply_gaussian_blur(image, kernel_size):
    """
    对输入的图像应用高斯模糊。

    参数：
    image: 输入的图像
    kernel_size: 内核大小，必须是正奇数

    返回值：
    模糊后的图像
    """
    # 确保内核大小是正奇数
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("Kernel size must be a positive odd number.")

    # 应用高斯模糊
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return blurred_image

def apply_mean_blur(image, kernel_size):
    """
    对输入的图像应用均值模糊。

    参数：
    image: 输入的图像
    kernel_size: 内核大小，必须是正奇数

    返回值：
    模糊后的图像
    """
    # 确保内核大小是正奇数
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("Kernel size must be a positive odd number.")

    # 应用均值模糊
    blurred_image = cv2.blur(image, (kernel_size, kernel_size))

    return blurred_image

def apply_denoising(image, denoising_method):
    """

    对输入的图像应用降噪算法。

    参数：
    image: 输入的图像
    denoising_method: 降噪算法名称，支持 "gaussian" 和 "mean"

    返回值：
    降噪后的图像
    """
    if denoising_method == "gaussian":
        # 使用高斯模糊进行降噪
        denoised_image = apply_gaussian_blur(image, 3)
    elif denoising_method == "mean":
        # 使用均值模糊进行降噪
        denoised_image = apply_mean_blur(image, 3)
    else:
        raise ValueError("Invalid denoising method. Supported methods: 'gaussian', 'mean'.")

    return denoised_image

def extract_dark_regions(image):
    """
    提取输入图像中的暗部区域。

    参数：
    image: 输入的图像

    返回值：
    提取的暗部区域图像（二值图像）
    """
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用自适应阈值化来提取暗部区域
    _, threshold_image = cv2.threshold(gray_image, 0, 240, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return threshold_image

def add_background(image):
    """
    为输入的图像添加浅灰色背景。

    参数：
    image: 输入的图像

    返回值：
    添加背景后的图像
    """
    # 创建与图像大小相同的浅灰色背景
    background = np.ones_like(image, dtype=np.uint8) * 230

    # 将图像和背景进行融合
    blended_image = cv2.addWeighted(image, 0.55, background, 0.45, 10)

    return blended_image

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def merge_videos(input_paths, output_path):
    # 获取第一个视频的信息
    cap = cv2.VideoCapture(input_paths[0])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    cap.release()

    # 逐个读取视频并写入合并视频
    for input_path in input_paths:
        cap = cv2.VideoCapture(input_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

def rolling_ball_single_images(img_path,save_path,ratio,radius=40,show_window=False):
    filename_with_extension = os.path.basename(img_path)
    file_name, _ = os.path.splitext(filename_with_extension)
    img = cv2.imread(img_path,0)
    img_a = img.copy()
    #img_b = np.zeros_like(img_a)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(img)
    final_img , background_img = subtract_background_rolling_ball(img, radius,light_background=True,use_paraboloid=True,do_presmooth=True)
    #final_grey = cv2.cvtColor(final_img,cv2.COLOR_BGR2GRAY)
    #img_binary = cv2.threshold(final_img,0,255,cv2.THRESH_BINARY_INV)
    equalized_image = cv2.equalizeHist(img_a)
    equalized_image_b = cv2.equalizeHist(final_img)
    if ratio >0:
        imagestack = stackImages(ratio, ([img_a,background_img, final_img],
                                           [equalized_image,equalized_image_b,clahe_img]))
        print("writing the processing images")
        cv2.imwrite(f'{save_path}/{file_name}_treat_process.png', imagestack)

    print("roling rolloing rolling ball on your image bk\n procesing>>>%s" % file_name)
    cv2.imwrite(f'{save_path}/{file_name}_treated.png', final_img)


    if show_window:
        cv2.imshow('imagestack', imagestack)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


