from gettext import find
import cv2
import yaml
import numpy as np

def main(m, n, photo_nums, device=0, Lambda=0.3):
    cap = cv2.VideoCapture(device)
    grid_size = (m, n)
    grid_point = np.zeros((1, np.prod(grid_size), 3), np.float32)
    grid_point[0, :, :2] = np.indices(grid_size).T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    if not cap.isOpened():
        print("Cannot open camera")
        return
 
    sum = 0
    pre = [[0,0]]
    while sum < photo_nums:
        t, frame = cap.read()
        print(frame.shape)
        if not t:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv2.namedWindow('frame', 0)
        cv2.resizeWindow('frame', 960, 480)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(gray.shape)
        ret, corners = cv2.findChessboardCorners(
            gray, 
            grid_size, 
            flags = cv2.CALIB_CB_FAST_CHECK
            )
        
        if ret == True:
            flag, new_point = count_distance(pre, corners, grid_size, Lambda)
            if flag:
                pre.append(new_point)
                objpoints.append(grid_point)
                corners2 = cv2.cornerSubPix(
                    gray,corners, 
                    (11,11), 
                    (-1,-1), 
                    (cv2.TERM_CRITERIA_EPS + 
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
                imgpoints.append(corners2)
                cv2.putText(frame, 'yes', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 5,(255,255,255),2,cv2.LINE_AA)
                cv2.drawChessboardCorners(frame, grid_size, corners2, ret)
                sum += 1
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    calibration(objpoints, imgpoints, gray.shape[::-1], './camera_parameter2.yaml')

    cap.release()
    cv2.destroyAllWindows()


def calibration(objpoints, imgpoints, img_shape, yaml_path):
    res = {}
    photo_nums = len(objpoints)
    K = np.zeros((3,3))
    D = np.zeros((4,1))

    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(photo_nums)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(photo_nums)]
    retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        img_shape,
        K,
        D,
        rvecs,
        tvecs,
        None,None)
    
    res["internal_reference"] = K.tolist()
    res["distortion"] = D.tolist()

    with open(yaml_path, 'w') as f:
        yaml.dump(res, f)


def load_interparamter(yaml_path):
    with open(yaml_path) as f:
        paramter = yaml.load(f, Loader=yaml.FullLoader)
    K = np.array(paramter['internal_reference'])
    D = np.array(paramter['distortion'])
    return K, D


def test_single_img(img_path, yaml_path):
    K, D = load_interparamter('./camera_parameter.yaml')

    img = cv2.imread(img_path)

    DIM = (1920,1080)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imshow('window', undistorted_img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        exit(0)
    elif key == ord('s'):
        cv2.imwrite('10_1.jpg', undistorted_img)
    cv2.destroyAllWindows()

def count_distance(pre, corners, grip_size, Lambda):
    centor = np.prod(grip_size)

    base = np.sqrt(np.power(corners[0][0][0]-corners[centor-1][0][0],2)+np.power(corners[0][0][1]-corners[centor-1][0][1],2))

    for i in range(len(pre)):
        com = np.sqrt(np.power(corners[centor//2][0][0]-pre[i][0],2)+np.power(corners[centor//2][0][1]-pre[i][1],2))
        if com > Lambda*base:
            continue
        else:
            return False, corners[centor//2][0]

    return True, corners[centor//2][0]

def calibrate_video(yaml_path, device=0, H=False):
    K, D = load_interparamter(yaml_path)
    cap = cv2.VideoCapture(device)
    with open('./homography_parameter.yaml') as f:
        h = yaml.load(f, Loader=yaml.FullLoader)
    h1 = np.array(h['homography'])
    while 1:
        t, frame = cap.read()
        cv2.namedWindow('frame', 0)
        cv2.resizeWindow('frame', 960, 480)
        if not t:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        DIM = frame.shape[:2][::-1]
        print(DIM)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        if H:
            undistorted_img = cv2.warpPerspective(undistorted_img, h1, (undistorted_img.shape[1], undistorted_img.shape[0]))

        cv2.imshow('frame', undistorted_img)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def findhomography():
    dict = {}

    src = np.array([[650,967],[1209,961],[801,753],[1112,735]])
    dst = np.array([[1248,49],[669,60],[1246,454],[681,463]])
 
    H, _ = cv2.findHomography(src, dst)
    dict["homography"] = H.tolist()
    with open('./homography_parameter.yaml', 'w') as f:
        yaml.dump(dict, f)

    return H

def warphomography(img):
    H = findhomography()
    out = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    return out



if __name__ == '__main__':
    # main(6,9,15, Lambda=0.12)

    test_single_img('./10.jpg', './camera_parameter.yaml')

    # calibrate_video('./camera_parameter.yaml', H=True)

    # img = cv2.imread('./11_1.jpg')
    # out = warphomography(img)
    # cv2.namedWindow('new', 0)
    # cv2.resizeWindow('new', 960, 480)
    # cv2.imshow('new', out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
