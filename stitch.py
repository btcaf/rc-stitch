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


def calibrate_camera_separate(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
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
        corners, ids, _ = detector.detectMarkers(gray)

        for marker_corners in corners:
            all_corners.append(marker_corners.reshape(4, 2).astype(np.float32))
            obj_points.append(objp)

        # img = cv2.aruco.drawDetectedMarkers(img, corners, ids)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, all_corners, gray.shape[::-1], None, None)

    error = 0
    for i in range(len(obj_points)):
        img_points, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        # print(all_corners[i].shape, img_points.shape, obj_points[i].shape)
        # print(all_corners[i], img_points)
        img_points = img_points.reshape(-1, 2)
        error += cv2.norm(all_corners[i], img_points, cv2.NORM_L2) / len(img_points)

    error /= len(obj_points)

    return camera_matrix, dist_coeffs, error


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


    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, all_corners, gray.shape[::-1], None, None)

    error = 0
    for i in range(len(obj_points)):
        img_points, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        img_points = img_points.reshape(-1, 2)
        error += cv2.norm(all_corners[i], img_points, cv2.NORM_L2) / len(img_points)

    error /= len(obj_points)

    return camera_matrix, dist_coeffs, error


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


def transform_image(img: np.ndarray, homography: np.ndarray, img_to_pad: np.ndarray, display: bool = False) -> tuple[np.ndarray, np.ndarray, np.float32, np.float32]:
    print(img.shape)
    dst_bounds = np.array([homography.dot(np.array([x, y, 1])) for x, y in itertools.product([0, img.shape[1]], [0, img.shape[0]])])
    dst_bounds = dst_bounds / dst_bounds[:, 2].reshape(-1, 1)
    dst_bounds = dst_bounds[:, :2]

    min_x, min_y = np.min(dst_bounds, axis=0)
    max_x, max_y = np.max(dst_bounds, axis=0)

    total_min_x, total_min_y = min(min_x, 0), min(min_y, 0)
    total_max_x, total_max_y = max(max_x, img.shape[1]), max(max_y, img.shape[0])

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
        # print(x, y, src_coords)

        src_coords = np.round(src_coords).astype(int)
        if 0 <= src_coords[0] < img.shape[1] and 0 <= src_coords[1] < img.shape[0]:
            dst[y - total_min_y, x - total_min_x] =  \
                img[src_coords[1], src_coords[0]] \
                if img.shape[2] == 4 \
                else np.append(img[src_coords[1], src_coords[0]], 1)

    # move original wrt new bounds
    img_to_pad = cv2.cvtColor(img_to_pad, cv2.COLOR_BGR2BGRA)
    new_img = np.zeros((total_shape[0], total_shape[1], 4), dtype=np.uint8)
    new_img[-total_min_y:(img_to_pad.shape[0] - total_min_y), -total_min_x:(img_to_pad.shape[1] - total_min_x)] = img_to_pad

    if display:
        cv2.imshow("original", new_img)
        cv2.imshow("dst", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
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
    val_1 = np.sum(img1[common_points][:, :3] ** gamma)
    val_2 = np.sum(img2[common_points][:, :3] ** gamma)
    return val_1 / val_2


def global_correction_coeff(local_coeffs: np.ndarray) -> np.ndarray:
    return np.sum(local_coeffs) / np.sum(local_coeffs ** 2)


def apply_correction(img: np.ndarray, local_coeffs: np.ndarray, global_coeffs: np.ndarray, gamma: np.float32) -> np.ndarray:
    new_img = img.copy()
    new_img[:, :, :3] = (global_coeffs * local_coeffs) ** (1 / gamma) * img[:, :, :3]
    return new_img


def pixel_cost(pixel1: np.ndarray, pixel2: np.ndarray) -> np.float32:
    diff = np.abs(pixel1[:, :-1] - pixel2[:, :-1])
    return (0.11 * diff[:, 0] + 0.59 * diff[:, 1] + 0.3 * diff[:, 2]) ** 2



def stitch_corrected_images(img1: np.ndarray, img2: np.ndarray, common_points: np.ndarray) -> np.ndarray:
    dp_table = np.zeros_like(common_points, dtype=np.float32)
    dp_table[0] = pixel_cost(img1[0], img2[0])
    dp_table[0][~common_points[0]] = np.inf

    for i in range(1, dp_table.shape[0]):
        minima = np.minimum(dp_table[i - 1], np.roll(dp_table[i - 1], 1), np.roll(dp_table[i - 1], -1))
        minima[minima == np.inf] = 0
        dp_table[i] = pixel_cost(img1[i], img2[i]) + minima
        dp_table[i][~common_points[i]] = np.inf

    # plt.imshow(dp_table)
    # plt.show()

    sewing_line = []
    last_min_idx = None
    for i in reversed(range(dp_table.shape[0])):
        if np.all(dp_table[i] == np.inf):
            continue
        if last_min_idx == None:
            last_min_idx = np.argmin(dp_table[i])
            sewing_line.append((i, last_min_idx))
        # print(i, last_min_idx)
        # print(dp_table.shape)
        last_min_idx = np.argmin(dp_table[i, max(last_min_idx - 1, 0):min(last_min_idx + 2, dp_table.shape[1])]) + max(last_min_idx - 1, 0)
        if dp_table[i, last_min_idx] == np.inf:
            break
        sewing_line.append((i, last_min_idx))
    
    sewing_line = list(reversed(sewing_line))
    # print("Line", list(sewing_line))

    new_img = img1.copy()

    for i, j in sewing_line:
        # print(i, j)
        new_img[i, :j, :3] = img2[i, :j, :3]

    for i, j in sewing_line:
        new_img[i, j] = [0, 0, 255, 1]
        # print(i, j, new_img[i, j])

    return new_img



def stitch_images(img1: np.ndarray, img2: np.ndarray, gamma: np.float32, pairs: list[list[list[int]]]) -> np.ndarray:
    img1, img2, _, _ = get_transformed_images(img1, img2, pairs)
    common_points = get_common_points(img1, img2)
    local_coeffs = local_correction_coeff(img1, img2, common_points, gamma)
    global_coeffs = global_correction_coeff(local_coeffs)
    img2 = apply_correction(img2, local_coeffs, global_coeffs, gamma)
    res = stitch_corrected_images(img1, img2, common_points)

    # cv2.imshow("stitched", res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    show_images([img1, img2, res])
    return res


### TASK 6 FUNCTIONS ###


def get_pairs(path: str) -> list[list[list[int]]]:
    npz = np.load(path)
    pairs = []
    for i, point in enumerate(npz['keypoints0']):
        if npz['matches'][i] != -1:
            pairs.append([point, npz['keypoints1'][npz['matches'][i]]])

    return pairs


### TASK 7 FUNCTIONS ###


def get_pairs_auto(images: list[np.ndarray]) -> list[list[list[list[int]]]]:
    os.makedirs("tmp")
    for i, img in enumerate(images):
        cv2.imwrite(f"tmp/img{i}.png", img)
    with open("tmp/pairs.txt", "w") as f:
        for i in range(len(images) - 1):
            f.write(f"img{i}.png img{i + 1}.png\n")

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
    os.system("rm -r tmp")
    return pairs


# run with SuperGluePretrainedNetwork in the working directory
def stitch_all(images: list[np.ndarray], gamma: np.float32) -> np.ndarray:
    all_pairs = get_pairs_auto(images)

    # # find local coefficients
    # local_coeffs = []
    # for i in range(len(images) - 1):
    #     img1, img2 = images[i], images[i + 1]
    #     img1, img2, _, _ = get_transformed_images(img1, img2, all_pairs[i])
    #     common_points = get_common_points(img1, img2)
    #     local_coeffs.append(local_correction_coeff(img1, img2, common_points, gamma))

    # # find global coefficients
    # global_coeffs = global_correction_coeff(np.array(local_coeffs))

    # # apply correction
    # corrected_images = [images[0]]
    # for i in range(1, len(images)):
    #     corrected_images.append(apply_correction(images[i], local_coeffs[i - 1], global_coeffs, gamma))
    corrected_images = images


    # stitch images
    stitched = corrected_images[0]
    shift_x, shift_y = 0, 0
    # TODO debug code
    all_stitched = [stitched]
    for i in range(1, len(corrected_images)):
        # transform
        img1, img2 = stitched, corrected_images[i]
        img1, img2, shift_x, shift_y = get_transformed_images(img1, img2, all_pairs[i - 1], shift_x, shift_y)
        common_points = get_common_points(img1, img2)
        stitched = stitch_corrected_images(img1, img2, common_points)
        all_stitched.append(stitched)

    show_images([img for img in corrected_images] + all_stitched)



### MAIN ###


def main():
    calibration_images = load_images("calibration")
    camera_matrix, dist_coeffs, err = calibrate_camera_together(calibration_images)
    
    stiching_images = load_images("stitching")
    undistorted_images = undistort_images(stiching_images, camera_matrix, dist_coeffs)
    first_two = undistorted_images[:2]
    pairs = get_pairs("img1_img2_matches.npz")
    # stitch_images(first_two[0], first_two[1], 2.0, pairs)
    # stitch_all(undistorted_images[:5], 2.0)
    stitch_images(first_two[0], first_two[1], 2.0, PAIRS)

if __name__ == "__main__":
    main()