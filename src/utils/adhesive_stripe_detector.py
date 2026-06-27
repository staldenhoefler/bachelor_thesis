import cv2
import numpy as np
import warnings
from skimage.filters import threshold_multiotsu


def realign_images(img_before, img_after, max_orb_features=10000, top_n_matches=50):
    """Realigns an 'after' image to match the perspective of a 'before' image.

    This function uses the ORB (Oriented FAST and Rotated BRIEF) feature
    detector to find corresponding keypoints between two images. It then
    calculates a partial affine transformation from the best matches and
    warps the 'after' image to align it with the 'before' image.

    Args:
        img_before (np.ndarray): The reference image in BGR format.
        img_after (np.ndarray): The image to be realigned, in BGR format.
        max_orb_features (int, optional): The maximum number of ORB features to detect.
            Defaults to 10000.
        top_n_matches (int, optional): The number of best feature matches to use for
            estimating the transformation. Defaults to 50.

    Returns:
        np.ndarray: The `img_after` warped to align with the `img_before` image.
    """
    gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_orb_features)
    kp1, des1 = orb.detectAndCompute(gray_before, None)
    kp2, des2 = orb.detectAndCompute(gray_after, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < top_n_matches:
        warnings.warn(f"Not enough matches found: {len(matches)}. Using all available matches.")
        top_n_matches = len(matches)

    good_matches = matches[:top_n_matches]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    M, inliers = cv2.estimateAffinePartial2D(pts2, pts1)

    h, w = gray_before.shape
    aligned = cv2.warpAffine(img_after, M, (w, h))

    image_before_aligned = cut_upper_part(img_before, amount_pixels=800)
    image_after_aligned = cut_upper_part(aligned, amount_pixels=800)

    image_before_aligned = cut_arround_image(image_before_aligned, percentage=0.00)
    image_after_aligned = cut_arround_image(image_after_aligned, percentage=0.00)

    return image_before_aligned, image_after_aligned


def find_contours_of_adhesive_stripes(image_before, image_after, max_orb_features=10000, top_n_matches=1000, threshold_percentage_of_image=0.001, padding_hight_percentage = 0.05, padding_width_percentage = 0.05, min_aspect_ratio: int=3, max_aspect_ration: int=5, filter_of_local_density_map: float=0.10, p:int=80, q:int=80):
    """Identifies the bounding boxes of adhesive stripes by comparing two images.

    This function detects regions of significant change between a 'before' and an
    'after' image. The process involves:
    1.  Aligning the 'after' image to the 'before' image to ensure a consistent perspective.
    2.  Calculating the absolute difference to highlight added or changed objects.
    3.  Applying a series of image processing steps (Canny edge detection, dilation,
        closing, and local density filtering) to create a clean binary mask of the changes.
    4.  Finding contours in the mask and filtering out noise based on a minimum area threshold.
    5.  Returning a list of padded bounding boxes for the detected contours.

    Args:
        image_before (np.ndarray): The reference image without stripes (BGR format).
        image_after (np.ndarray): The image containing the adhesive stripes (BGR format).
        max_orb_features (int, optional): Max ORB features to use for initial alignment.
            Passed to `realign_images`. Defaults to 10000.
        top_n_matches (int, optional): Number of feature matches to use for alignment.
            Passed to `realign_images`. Defaults to 50.
        threshold_percentage_of_image (float, optional): The minimum area a contour must have,
            as a percentage of the total image area, to be considered a stripe.
            Defaults to 0.01 (1%).
        padding_hight_percentage (float, optional): The percentage of bounding box height
            to add as vertical padding to the final bounding boxes. Defaults to 0.25.
        padding_width_percentage (float, optional): The percentage of bounding box width
            to add as horizontal padding. Defaults to 0.05.

    Returns:
        list[list[int]]: A list of detected bounding boxes, where each box is a
        list in the format [x, y, width, height].
    """
    if max_orb_features > 0 and top_n_matches > 0:
        image_before_aligned, image_after_aligned = realign_images(image_before, image_after, max_orb_features=max_orb_features,
                                                               top_n_matches=top_n_matches)
    else:
        image_before_aligned, image_after_aligned = image_before, image_after

    image_abs_diff = cv2.absdiff(image_before_aligned, image_after_aligned)

    image_abs_diff_gray = cv2.cvtColor(image_abs_diff, cv2.COLOR_BGR2GRAY)

    # image_abs_diff_cut = image_abs_diff[0:3800, 100:2300]

    def multi_otsu_segment(stripe_gray, n_classes=2):
        thresholds = threshold_multiotsu(stripe_gray, classes=n_classes)
        regions = np.digitize(stripe_gray, bins=thresholds)
        return regions, thresholds

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_abs_diff_gray)

    regions, treshhold = multi_otsu_segment(enhanced, n_classes=3)

    binary_mask = (regions == regions.max()).astype(np.uint8)
    k, l = 30 ,30
    kernel = np.ones((k, l), dtype=np.float32)
    edges = cv2.filter2D(binary_mask, ddepth=cv2.CV_16S, kernel=kernel)
    abs_edges = cv2.convertScaleAbs(edges)

    mask_first_alg = (abs_edges > 250)
    mask_first_alg = mask_first_alg.astype(np.uint8)

    norm = image_abs_diff_gray / 255.0

    kernel = np.ones((p, q), dtype=np.float32)
    local_sum = cv2.filter2D(norm.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE)
    local_density_map = local_sum / (p * q)

    mask_second_alg = (local_density_map > filter_of_local_density_map).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    clean_second_alg = cv2.morphologyEx(mask_second_alg, cv2.MORPH_CLOSE, kernel, iterations=2)

    clean_second_alg = np.array(clean_second_alg)

    clean = (mask_first_alg * clean_second_alg * image_abs_diff_gray) / 3
    clean = clean.astype(np.uint8)

    p, q = 70, 70
    kernel = np.ones((p, q), dtype=np.float32)
    local_sum = cv2.filter2D(clean.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE)
    combination = local_sum / (p * q)
    combination = combination.astype(np.uint8)

    cnts, _ = cv2.findContours(combination, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = binary_mask.shape[0] * binary_mask.shape[1]
    cnts = [c for c in cnts if cv2.contourArea(c) > threshold_percentage_of_image * image_area]

    contours_list = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / h
        if aspect_ratio < (1/min_aspect_ratio) and aspect_ratio > (1/max_aspect_ration):
            contours_list.append(c)

    objects = []
    for c in contours_list:
        x, y, w, h = cv2.boundingRect(c)
        padding_x = int(w * padding_width_percentage)
        padding_y = int(h * padding_hight_percentage)
        #x = x + 100
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(image_after.shape[1] - x, w + 2 * padding_x)
        h = min(image_after.shape[0] - y, h + 2 * padding_y)
        objects.append([x, y, w, h])

    return image_before_aligned, image_after_aligned, objects


def cut_upper_part(image, amount_pixels: int = None, percentage: float = None):
    """Cuts the upper part of an image by a specified number of pixels or percentage.

    Args:
        image (np.ndarray): The input image from which the upper part will be cut.
        amount_pixels (int, optional): The number of pixels to cut from the top.
            If specified, `percentage` should be None. Defaults to None.
        percentage (float, optional): The percentage of the image height to cut from the top.
            If specified, `amount_pixels` should be None. Defaults to None.

    Returns:
        np.ndarray: The image with the upper part cut off.
    """
    if amount_pixels is not None and percentage is not None:
        raise ValueError("Specify either amount_pixels or percentage, not both.")

    if amount_pixels is not None:
        return image[amount_pixels:, :, :]

    if percentage is not None:
        amount_pixels = int(image.shape[0] * percentage)
        return image[amount_pixels:, :, :]

    return image

def cut_arround_image(image, percentage: float=0.05):
    """Cuts a percentage of the image from all sides.

    Args:
        image (np.ndarray): The input image to be cut.
        percentage (float, optional): The percentage of the image size to cut from each side.
            Defaults to 0.05 (5%).

    Returns:
        np.ndarray: The image with the specified percentage cut from all sides.
    """
    if not (0 <= percentage < 1):
        raise ValueError("Percentage must be between 0 and 1.")

    height, width = image.shape[:2]
    cut_height = int(height * percentage)
    cut_width = int(width * percentage)

    return image[cut_height:height - cut_height, cut_width:width - cut_width, :]



def find_substrate_plate(img):
    import cv2
    import numpy as np
    from skimage.filters import threshold_multiotsu

    def multi_otsu_segment(stripe_gray, n_classes=2):
        thresholds = threshold_multiotsu(stripe_gray, classes=n_classes)
        regions = np.digitize(stripe_gray, bins=thresholds)
        return regions, thresholds

    image_abs_diff_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    image_abs_diff_gray = cv2.convertScaleAbs(image_abs_diff_gray, alpha=0.05, beta=0.1)

    regions, treshhold = multi_otsu_segment(image_abs_diff_gray, n_classes=2)

    binary_mask = (regions == regions.max()).astype(np.uint8)
    p, q = 150, 150
    kernel = np.ones((p, q), dtype=np.float32)
    local_sum = cv2.filter2D(binary_mask.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE)
    local_density_map = local_sum / (p * q)
    local_sum = cv2.filter2D(local_density_map.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE)
    local_density_map = local_sum / (p * q)

    # make binary mask
    binary_mask = (local_density_map < 0.975).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = binary_mask.shape[0] * binary_mask.shape[1]
    threshold_percentage_of_image = 0.1
    cnts = [c for c in cnts if cv2.contourArea(c) > threshold_percentage_of_image * image_area]

    max_sized_cnt = max(cnts, key=cv2.contourArea, default=None)
    x, y, w, h = cv2.boundingRect(max_sized_cnt)

    return img[y:y + h, x:x + w, :]