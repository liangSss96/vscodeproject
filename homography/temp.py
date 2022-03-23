#coding:utf-8

# This code only tested in OpenCV 3.4.2!
import cv2
import numpy as np

# 读取图片
im1 = cv2.imread('2.jpg')
im2 = cv2.imread('1.jpg')

# 计算SURF特征点和对应的描述子，kp存储特征点坐标，des存储对应描述子
surf = cv2.xfeatures2d.SURF_create()
kp1, des1 = surf.detectAndCompute(im1, None)
kp2, des2 = surf.detectAndCompute(im2, None)

# 匹配特征点描述子
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)


print(type(matches))
print("**************")
# print(matches.shape)
print("**************")
# print(matches)

# 提取匹配较好的特征点
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# 通过特征点坐标计算单应性矩阵H
# （findHomography中使用了RANSAC算法剔初错误匹配）
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

print(H)

# 使用单应性矩阵计算变换
h, w, d = im1.shape
pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, H)
img2 = cv2.polylines(im2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)


# 关键点连线
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

im3 = cv2.drawMatches(im1, kp1, im2, kp2, good, None, **draw_params)


out = cv2.warpPerspective(im2, np.linalg.inv(H), (im2.shape[1]+im1.shape[1], im2.shape[0]))

# out1 = cv2.warpPerspective(im2, np.eye(3), (im2.shape[1]+im1.shape[1], im2.shape[0]))


direct=out.copy()
direct[0:im1.shape[0], 0:im1.shape[1]] = im1


rows,cols=im1.shape[:2]
left, right = (0,0)

# 确定重合的列的范围
for col in range(0,cols):
    if im1[:, col].any() and out[:, col].any():#开始重叠的最左端
        left = col
        break
for col in range(cols-1, 0, -1):
    if im1[:, col].any() and out[:, col].any():#重叠的最右一列
        right = col
        break


print(im1.shape[:2],left, right)
res = np.zeros([rows, cols, 3], np.uint8)
for row in range(0, rows):
    for col in range(0, cols):
        if not im1[row, col].any():#如果没有原图，用旋转的填充
            res[row, col] = out[row, col]
        elif not out[row, col].any():
            res[row, col] = im1[row, col]
        else:
            srcImgLen = float(abs(col - left))
            testImgLen = float(abs(col - right))
            alpha = srcImgLen / (srcImgLen + testImgLen)
            res[row, col] = np.clip(im1[row, col] * (1-alpha) + out[row, col] * alpha, 0, 255)

out[0:im1.shape[0], 0:im1.shape[1]]=res

# out[0:im1.shape[0], 0:im1.shape[1]]=im1[0:im1.shape[0], 0:im1.shape[1]]

cv2.namedWindow('frame', 0)
cv2.resizeWindow('frame', 960, 480)
cv2.imshow('frame', out)
key = cv2.waitKey(0)
if key == ord('q'):
    exit()
cv2.destroyAllWindows()