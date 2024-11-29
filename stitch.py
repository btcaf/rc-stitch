import numpy as np
import cv2
import os
import itertools
import re
import matplotlib.pyplot as plt
import subprocess


def load_images(folder_path: str) -> list[np.ndarray]:
    files = filter(lambda x: re.fullmatch(r'img\d+\.png', x), os.listdir(folder_path))
    files = sorted(files, key=lambda x: int(x[3:-4]))

    return [cv2.imread(os.path.join(folder_path, filename)) for filename in files]


### TASK 1 FUNCTIONS ###


def calibrate_and_compute_error(obj_points: np.ndarray, all_corners: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.float64]:
    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, all_corners, gray.shape[::-1], None, None)

    error = 0
    for i in range(len(obj_points)):
        img_points, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        img_points = img_points.reshape(-1, 2)
        error += cv2.norm(all_corners[i], img_points, cv2.NORM_L2) / len(img_points)

    error /= len(obj_points)

    return camera_matrix, dist_coeffs, error


def calibrate_camera_separate(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.float64]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    all_corners = []
    obj_points = []
    objp = np.array([
        [0, 168, 0],
        [0, 0, 0],
        [168, 0, 0],
        [168, 168, 0]
    ], dtype=np.float32)

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, _, _ = detector.detectMarkers(gray)

        for marker_corners in corners:
            all_corners.append(marker_corners.reshape(4, 2).astype(np.float32))
            obj_points.append(objp)

    return calibrate_and_compute_error(obj_points, all_corners, gray)


def get_together_obj_points(ids):
    position_dict = {
        28: (0, 0),
        29: (0, 168 + 70),
        23: (168 + 70, 0),
        24: (168 + 70, 168 + 70),
        18: (2 * (168 + 70), 0),
        19: (2 * (168 + 70), 168 + 70),
    }

    obj_points = []
    for id in ids:
        obj_points.extend([
            [position_dict[id][0], position_dict[id][1] + 168, 0],
            [position_dict[id][0], position_dict[id][1], 0],
            [position_dict[id][0] + 168, position_dict[id][1], 0],
            [position_dict[id][0] + 168, position_dict[id][1] + 168, 0]
        ])

    return np.array(obj_points, dtype=np.float32)


def calibrate_camera_together(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.float64]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    all_corners = []
    obj_points = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        corners = np.hstack(corners).reshape(-1, 2).astype(np.float32)
        all_corners.append(corners)
        obj_points.append(get_together_obj_points(ids.flatten()))


    return calibrate_and_compute_error(obj_points, all_corners, gray)


def undistort_images(images: list[np.ndarray], camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> list[np.ndarray]:
    new_images = []
    for img in images:
        h, w = img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0, (w, h))
        map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), 5)
        res = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        x, y, w, h = roi
        res = res[y:y+h, x:x+w]
        new_images.append(res)
    return new_images


### TASK 2 FUNCTIONS ###


def transform_image(img: np.ndarray, homography: np.ndarray, img_to_pad: np.ndarray = None, display: bool = False) -> tuple[np.ndarray, np.ndarray, np.float32, np.float32]:
    if img_to_pad is None:
        img_to_pad = img
    dst_bounds = np.array([homography.dot(np.array([x, y, 1])) for x, y in itertools.product([0, img.shape[1]], [0, img.shape[0]])])
    dst_bounds = dst_bounds / dst_bounds[:, 2].reshape(-1, 1)
    dst_bounds = dst_bounds[:, :2]

    min_x, min_y = np.min(dst_bounds, axis=0)
    max_x, max_y = np.max(dst_bounds, axis=0)

    total_min_x, total_min_y = min(min_x, 0), min(min_y, 0)
    total_max_x, total_max_y = max(max_x, img_to_pad.shape[1]), max(max_y, img_to_pad.shape[0])

    total_min_x = int(total_min_x + np.sign(total_min_x))
    total_min_y = int(total_min_y + np.sign(total_min_y))
    total_max_x = int(total_max_x + np.sign(total_max_x))
    total_max_y = int(total_max_y + np.sign(total_max_y))

    total_shape = (total_max_y - total_min_y + 1, total_max_x - total_min_x + 1)

    dst = np.zeros((total_shape[0], total_shape[1], 4), dtype=np.uint8)
    for x, y in itertools.product(range(total_min_x, total_max_x + 1), range(total_min_y, total_max_y + 1)):
        try:
            src_coords = np.linalg.solve(homography, np.array([x, y, 1]))
        except np.linalg.LinAlgError:
            # matrix is singular, return empty image
            break
        
        src_coords /= src_coords[2]
        src_coords = src_coords[:2]

        src_coords = np.round(src_coords).astype(int)
        if 0 <= src_coords[0] < img.shape[1] and 0 <= src_coords[1] < img.shape[0]:
            dst[y - total_min_y, x - total_min_x] =  \
                img[src_coords[1], src_coords[0]] \
                if img.shape[2] == 4 \
                else np.append(img[src_coords[1], src_coords[0]], 1)

    img_to_pad = cv2.cvtColor(img_to_pad, cv2.COLOR_BGR2BGRA)
    new_img = np.zeros((total_shape[0], total_shape[1], 4), dtype=np.uint8)
    new_img[-total_min_y:(img_to_pad.shape[0] - total_min_y), -total_min_x:(img_to_pad.shape[1] - total_min_x)] = img_to_pad

    if display:
        cv2.imshow("original", new_img)
        cv2.imshow("dst", dst)
        cv2.waitKey(0)
    
    return dst, new_img, total_min_x, total_min_y


### TASK 3 FUNCTIONS ###


def compute_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    A = []
    for src, dst in zip(src_points, dst_points):
        x, y = src
        u, v = dst
        A.append([x, y, 1, 0, 0, 0, -x*u, -y*u, -u])
        A.append([0, 0, 0, x, y, 1, -x*v, -y*v, -v])

    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    eigenvector = V[-1, :]

    return eigenvector.reshape(3, 3)


def test_compute_homography_single():
    homography = np.random.rand(3, 3)
    homography /= np.linalg.norm(homography)

    src_points = np.random.rand(4, 2)
    src_points = np.hstack((src_points, np.ones([src_points.shape[0],1])))
    dst_points = src_points.dot(homography.T)
    dst_points = dst_points / dst_points[:, 2].reshape(-1, 1)

    computed_homography = compute_homography(src_points[:, :2], dst_points[:, :2])

    assert np.allclose(homography, computed_homography) or np.allclose(homography, -computed_homography)


def test_compute_homography(iters: int = 100):
    for _ in range(iters):
        test_compute_homography_single()


### TASK 4 FUNCTIONS ###


def show_images(images: list[np.ndarray]):
    plt_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    for i, img in enumerate(plt_images):
        plt.figure()
        plt.imshow(img)
        plt.title(f"Image {i}")
    plt.show()


# matching pairs between img1.png and img2.png, found by hand
PAIRS = [
    [[517, 374], [624, 381]],
    [[926, 360], [1043, 366]],
    [[1058, 193], [1188, 190]],
    [[136, 47], [276, 76]],
    [[1084, 448], [1221, 460]]
]


### TASK 5 FUNCTIONS ###


# dst is only padded, src is transformed
def get_transformed_images(src: np.ndarray, dst: np.ndarray, pairs: list[list[list[int]]], shift_x: np.float32=0, shift_y: np.float32=0) -> tuple[np.ndarray, np.ndarray, np.float32, np.float32]:
    src_points = np.array([pair[0] for pair in pairs], dtype=np.float32)
    dst_points = np.array([pair[1] for pair in pairs], dtype=np.float32)

    homography = compute_homography(src_points, dst_points)
    homography = np.array([[1, 0, shift_x], [0, 1, shift_y], [0, 0, 1]]) @ homography
    src_new, dst_new, shift_x, shift_y = transform_image(src, homography, dst, display=False)
    return src_new, dst_new, shift_x, shift_y


# returns boolean mask of common points
def get_common_points(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    return np.logical_and(img1[:, :, 3] > 0, img2[:, :, 3] > 0)


# correction coefficients for img2 wrt img1
def local_correction_coeff(img1: np.ndarray, img2: np.ndarray, common_points: np.ndarray, gamma: np.float32) -> np.ndarray:
    val_1 = np.sum((img1[common_points][:, :3] / 256) ** gamma, axis=0)
    val_2 = np.sum((img2[common_points][:, :3] / 256) ** gamma, axis=0)
    return val_1 / val_2


def global_correction_coeff(local_coeffs: np.ndarray) -> np.ndarray:
    return np.sum(local_coeffs, axis=0) / np.sum(local_coeffs ** 2, axis=0)


def apply_correction(img: np.ndarray, local_coeffs: np.ndarray, global_coeffs: np.ndarray, gamma: np.float32) -> np.ndarray:
    new_img = img.copy()
    new_img[:, :, :3] = np.minimum((global_coeffs * local_coeffs) ** (1 / gamma) * img[:, :, :3], 255)
    return new_img


def pixel_cost(pixel1: np.ndarray, pixel2: np.ndarray) -> np.float32:
    diff = np.abs(pixel1[:, :-1] - pixel2[:, :-1])
    return (0.11 * diff[:, 0] + 0.59 * diff[:, 1] + 0.3 * diff[:, 2]) ** 2



def stitch_corrected_images(img1: np.ndarray, img2: np.ndarray, common_points: np.ndarray, is_reverse: bool) -> np.ndarray:
    dp_table = np.full_like(common_points, np.inf, dtype=np.float32)
    first = 0
    while first < dp_table.shape[0] and not np.any(common_points[first]):
        first += 1
    dp_table[first] = pixel_cost(img1[first], img2[first])
    dp_table[first][~common_points[first]] = np.inf

    for i in range(first + 1, dp_table.shape[0]):
        minima = np.minimum(dp_table[i - 1], np.roll(dp_table[i - 1], 1), np.roll(dp_table[i - 1], -1))
        dp_table[i] = pixel_cost(img1[i], img2[i]) + minima
        dp_table[i][~common_points[i]] = np.inf

    sewing_line = []
    last_min_idx = None
    for i in reversed(range(dp_table.shape[0])):
        if np.all(dp_table[i] == np.inf):
            continue
        if last_min_idx == None:
            last_min_idx = np.argmin(dp_table[i])
            sewing_line.append((i, last_min_idx))

        last_min_idx = np.argmin(dp_table[i, max(last_min_idx - 1, 0):min(last_min_idx + 2, dp_table.shape[1])]) + max(last_min_idx - 1, 0)
        if dp_table[i, last_min_idx] == np.inf:
            break
        sewing_line.append((i, last_min_idx))
    
    sewing_line = list(reversed(sewing_line))

    new_img = img1 + img2
    new_img[:, :, 3] = np.logical_or(img1[:, :, 3], img2[:, :, 3]).astype(np.uint8) * 255

    for i, j in sewing_line:
        if is_reverse:
            new_img[i, :j, :3] = img1[i, :j, :3]
            new_img[i, j:, :3] = img2[i, j:, :3]
        else:
            new_img[i, :j, :3] = img2[i, :j, :3]
            new_img[i, j:, :3] = img1[i, j:, :3]

    return new_img



def stitch_images(img1: np.ndarray, img2: np.ndarray, gamma: np.float32, pairs: list[list[list[int]]], is_reverse: bool) -> np.ndarray:
    img1, img2, _, _ = get_transformed_images(img1, img2, pairs)
    common_points = get_common_points(img1, img2)
    local_coeffs = local_correction_coeff(img1, img2, common_points, gamma)
    global_coeffs = global_correction_coeff(local_coeffs)
    img1 = apply_correction(img1, np.array([1, 1, 1]), global_coeffs, gamma)
    img2 = apply_correction(img2, local_coeffs, global_coeffs, gamma)
    res = stitch_corrected_images(img1, img2, common_points, is_reverse)

    return res


### TASK 6 FUNCTIONS ###


def get_pairs(path: str) -> list[list[list[int]]]:
    npz = np.load(path)
    pairs = []
    for i, point in enumerate(npz['keypoints0']):
        if npz['matches'][i] != -1:
            pairs.append([point, npz['keypoints1'][npz['matches'][i]]])

    return pairs


def compute_homography_ransac(pairs: list[list[list[int]]], threshold: np.float32, max_iters: int) -> np.ndarray:
    best_inliers = 0
    best_homography = None

    for _ in range(max_iters):
        sample = np.random.choice(len(pairs), 4, replace=False)
        src_points = np.array([pairs[i][0] for i in sample], dtype=np.float32)
        dst_points = np.array([pairs[i][1] for i in sample], dtype=np.float32)

        homography = compute_homography(src_points, dst_points)

        inliers = 0
        for src, dst in pairs:
            src = np.append(src, 1)
            dst = np.append(dst, 1)
            src = homography.dot(src)
            src = src / src[2]
            if np.linalg.norm(src - dst) < threshold:
                inliers += 1
        if inliers > best_inliers:
            best_inliers = inliers
            best_homography = homography

    return best_homography


def get_transformed_images_ransac(src: np.ndarray, dst: np.ndarray, pairs: list[list[list[int]]], shift_x: np.float32=0, shift_y: np.float32=0) -> tuple[np.ndarray, np.ndarray, np.float32, np.float32]:
    homography = compute_homography_ransac(pairs, 3, 100)
    homography = np.array([[1, 0, shift_x], [0, 1, shift_y], [0, 0, 1]]) @ homography
    src_new, dst_new, shift_x, shift_y = transform_image(src, homography, dst, display=False)
    return src_new, dst_new, shift_x, shift_y


def stich_images_superglue(img1: np.ndarray, img2: np.ndarray, gamma: np.float32, is_reverse: bool, path: str) -> np.ndarray:
    pairs = get_pairs(path)
    img1, img2, _, _ = get_transformed_images_ransac(img1, img2, pairs)
    common_points = get_common_points(img1, img2)
    local_coeffs = local_correction_coeff(img1, img2, common_points, gamma)
    global_coeffs = global_correction_coeff(local_coeffs)
    img1 = apply_correction(img1, np.array([1, 1, 1]), global_coeffs, gamma)
    img2 = apply_correction(img2, local_coeffs, global_coeffs, gamma)
    res = stitch_corrected_images(img1, img2, common_points, is_reverse)

    return res


### TASK 7 FUNCTIONS ###


def get_pairs_auto(images: list[np.ndarray]) -> list[list[list[list[int]]]]:
    os.makedirs("tmp")
    for i, img in enumerate(images):
        cv2.imwrite(f"tmp/img{i}.png", img)
    with open("tmp/pairs.txt", "w") as f:
        for i in range(len(images) - 1):
            f.write(f"img{i}.png img{i + 1}.png 0 0\n")

    subprocess.run(
        [
            "./SuperGluePretrainedNetwork/match_pairs.py", 
            "--input_pairs", "tmp/pairs.txt" , 
            "--input_dir", "tmp", 
            "--output_dir", "tmp", 
            "--resize", "1280", "720",
            "--viz"
         ]
    )
    pairs = [get_pairs(f"tmp/img{i}_img{i + 1}_matches.npz") for i in range(len(images) - 1)]
    subprocess.run(["rm", "-r", "tmp"])
    return pairs


# run with SuperGluePretrainedNetwork in the working directory
def stitch_all_median(images: list[np.ndarray], gamma: np.float32) -> np.ndarray:
    median = len(images) // 2
    all_pairs = get_pairs_auto(images)
    first_half_pairs = all_pairs[:median]
    second_half_pairs = get_pairs_auto(list(reversed(images[median:])))

    local_coeffs = [np.array([1, 1, 1])]
    for i in range(len(images) - 1):
        img1, img2 = images[i], images[i + 1]
        img1, img2, _, _ = get_transformed_images_ransac(img1, img2, all_pairs[i])
        common_points = get_common_points(img1, img2)
        local_coeffs.append(local_correction_coeff(img1, img2, common_points, gamma))

    global_coeffs = global_correction_coeff(np.array(local_coeffs))

    corrected_images = []
    for i in range(len(images)):
        corrected_images.append(apply_correction(images[i], local_coeffs[i], global_coeffs, gamma))

    stitched = corrected_images[0]
    shift_x, shift_y = 0, 0
    for i in range(1, median + 1):
        img1, img2 = stitched, corrected_images[i]
        img1, img2, shift_x, shift_y = get_transformed_images_ransac(img1, img2, first_half_pairs[i - 1], shift_x, shift_y)
        common_points = get_common_points(img1, img2)
        stitched = stitch_corrected_images(img1, img2, common_points, False)

    stitched2 = corrected_images[-1]
    shift_x2, shift_y2 = 0, 0
    for i in range(len(corrected_images) - 2, median, -1):
        img1, img2 = stitched2, corrected_images[i]
        img1, img2, shift_x2, shift_y2 = get_transformed_images_ransac(img1, img2, second_half_pairs[i - median], shift_x2, shift_y2)
        common_points = get_common_points(img1, img2)
        stitched2 = stitch_corrected_images(img1, img2, common_points, True)

    if second_half_pairs:
        img1, img2 = stitched2, stitched
        img1, img2, shift_x, shift_y = get_transformed_images_ransac(img1, img2, second_half_pairs[0], shift_x2 - shift_x, shift_y2 - shift_y)
        common_points = get_common_points(img1, img2)
        stitched = stitch_corrected_images(img1, img2, common_points, True)

    return stitched


### MAIN ###


def main():
    ## TASK 1 ##

    print("Task 1:")
    calibration_images = load_images("calibration")
    print("Calibrating camera by using separate tags")
    _, _, err = calibrate_camera_separate(calibration_images)
    print("Reprojection error:", err)

    print("Calibrating camera by using all tags together")
    camera_matrix, dist_coeffs, err = calibrate_camera_together(calibration_images)
    print("Reprojection error:", err)

    ## TASK 2 ##

    print("Task 2 example:")
    transform_image(calibration_images[0], np.array([[0.7, -0.7, 0], [0.7, 0.7, 0], [0, 0, 1]]), img_to_pad=None, display=True)
    cv2.destroyAllWindows()

    ## TASK 3 ##

    test_compute_homography()
    print("Task 3 tests passed")

    ## TASK 4 ##
    
    # nothing to show here

    ## TASK 5 ##
    
    print("Task 5:")
    stiching_images = load_images("stitching")
    undistorted_images = undistort_images(stiching_images, camera_matrix, dist_coeffs)

    stitched_image = stitch_images(undistorted_images[0], undistorted_images[1], 2.2, PAIRS, False)
    cv2.imshow("Task 5", stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("task_5_stitched.jpg", stitched_image)

    ## TASK 6 ##

    print("Task 6:")
    stitched_image = stich_images_superglue(undistorted_images[0], undistorted_images[1], 2.2, False, "img1_img2_matches.npz")
    cv2.imshow("Task 6", stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("task_6_stitched.jpg", stitched_image)

    # ## TASK 7 ##

    print("Task 7:")
    stiched_image = stitch_all_median(undistorted_images[:5], 2.2)
    cv2.imshow("Task 7", stiched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("task_7_stitched.jpg", stiched_image)

if __name__ == "__main__":
    main()