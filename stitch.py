import numpy as np
import cv2
import os
import itertools


def load_images(folder_path: str) -> list[np.ndarray]:
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder_path, filename))
            images.append(img)
    return images


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


### TASK 2 FUNCTIONS ###


def transform_image(img: np.ndarray, homography: np.ndarray, display: bool = False) -> np.ndarray:
    dst = np.zeros_like(img)
    for x, y in itertools.product(range(img.shape[1]), range(img.shape[0])):
        try:
            src_coords = np.linalg.solve(homography, np.array([x, y, 1]))
        except np.linalg.LinAlgError:
            dst = np.zeros_like(img)
            break
        
        src_coords /= src_coords[2]
        src_coords = src_coords[:2]

        src_coords = np.round(src_coords).astype(int)
        if 0 <= src_coords[0] < img.shape[1] and 0 <= src_coords[1] < img.shape[0]:
            dst[y, x] = img[src_coords[1], src_coords[0]]

    if display:
        cv2.imshow("original", img)
        cv2.imshow("dst", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return dst


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


### MAIN ###


def main():
    calibration_images = load_images("calibration")
    camera_matrix, dist_coeffs, err = calibrate_camera_separate(calibration_images)
    print(err)

    img = cv2.imread("calibration/img1.png")
    cv2.imshow("img", img)
    cv2.waitKey(0)

    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0, (w, h))
    map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), 5)
    res = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    x, y, w, h = roi
    res = res[y:y+h, x:x+w]

    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()