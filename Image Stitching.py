import cv2
import numpy as np

point_color = (0, 0, 255)
img_show = None
points = []

# np.set_printoptions(suppress=True)

def draw_circle_on_click(event, x_, y_, flags, param):
    global img_show, point_color, points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img_show, (x_, y_), 2, point_color, -1)
        points.append([x_, y_, 1])

def transform_points(homography, points_):
    trans_points_ = np.matmul(homography, points_.T).T
    return trans_points_ / trans_points_[:, -1][:, np.newaxis]


def bilinear_interpolation(img_in, points):
    img_ = img_in.astype(float)
    
    x = points[:, 0]
    y = points[:, 1]
    
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, img_.shape[1] - 1)
    x1 = np.clip(x1, 0, img_.shape[1] - 1)
    y0 = np.clip(y0, 0, img_.shape[0] - 1)
    y1 = np.clip(y1, 0, img_.shape[0] - 1)

    # TODO: implement bilinear interpolation
    a = x-x0
    b = y-y0
    
    interpolated = ((img_[y0,x0,:]*np.array((1-a)*(1-b)).reshape(-1,1))
                    + (np.array(a*(1-b)).reshape(-1,1)*img_[y0,x1,:])
                    + (np.array(a*b).reshape(-1,1)*img_[y1,x1,:])
                    + (np.array((1-a)*b).reshape(-1,1)*img_[y1,x0,:])
                    )
    
    return interpolated


def image_transform(img_in, Homog):
    print('[+] Transform Image')
    h, w, _ = img_in.shape
    corners = np.array([
        [0., 0., 1.], [w - 1., 0., 1.],
        [0., h - 1., 1.], [w - 1., h - 1., 1.]])
    corners_trans = transform_points(Homog, corners)

    min_x, min_y = np.min(corners_trans[:, 0]), np.min(corners_trans[:, 1])
    max_x, max_y = np.max(corners_trans[:, 0]), np.max(corners_trans[:, 1])
    
    img_w, img_h = int(max_x - min_x), int(max_y - min_y)
    base_point = np.array([min_x, min_y, 0.])

    indices = np.array([[x_, y_, 1.] for y_ in np.arange(img_h) for x_ in np.arange(img_w)], dtype=np.float)
    indices_origin = transform_points(np.linalg.inv(Homog), indices + base_point)
    
    img_interpolated = bilinear_interpolation(img_in, indices_origin).astype(int)
    img_interpolated = img_interpolated.reshape(img_h, img_w, 3)
    
    print('[+] Image Shape: {} --> {}'.format(img_in.shape, img_interpolated.shape))
    print('[+] Complete')
    return img_interpolated, base_point

def combine_images(imgs_list, base_points_):
    base_points = np.floor(base_points_).astype(int)

    base_min = np.min(base_points, axis=0)
    base_max = np.max(
        [[b[0]+img.shape[1], b[1]+img.shape[0]] for b, img in list(zip(base_points, imgs_list))], axis=0)
    w = base_max[0] - base_min[0]
    h = base_max[1] - base_min[1]
    comb = np.ones((h, w, 3), dtype=int)*255

    for img, base in list(zip(imgs_list, base_points)):
        mask = np.where(np.sum(img, axis=-1) != 0)
        comb[mask[0]+base[1]-base_min[1], mask[1]+base[0]-base_min[0]] = img[mask[0], mask[1]]

    return comb


def find_homography(source_points, target_points):
    # points normalization
    std_1 = np.sqrt(np.sum(np.var(source_points[:, :2], axis=0)) / 2)
    mean_1 = np.mean(source_points, axis=0)
    std_2 = np.sqrt(np.sum(np.var(target_points[:, :2], axis=0)) / 2)
    mean_2 = np.mean(target_points, axis=0)
    
    # TODO: implement normalization
    T_image1 = np.array([
        [1/std_1, 0, -mean_1[0]/std_1],
        [0, 1/std_1, -mean_1[1]/std_1],
        [0,0,1]
        ])
    T_image2 = np.array([
        [1/std_2, 0, -mean_2[0]/std_2],
        [0, 1/std_2, -mean_2[1]/std_2],
        [0,0,1]
        ])

    source_points_n = np.matmul(T_image1, source_points.T).T
    target_points_n = np.matmul(T_image2, target_points.T).T

    print('----- Normalization Params ----------------------')
    print('points shape:', source_points_n.shape, target_points_n.shape)

    # find homography
    N = len(source_points_n)
    A = np.zeros((2 * N, 9), dtype=np.float64)

    # x' = Hx -> Ah = 0
    for i in range(N):
        x_, y_, _ = target_points_n[i]  # after
        X = source_points_n[i]  # before
        # TODO: implement homography
        A[2 * i] =       [X[0],X[1],X[2],0,0,0,-x_*X[0],-x_*X[1],-x_*X[2]]
        A[(2 * i) + 1] = [0,0,0,X[0],X[1],X[2],-y_*X[0],-y_*X[1],-y_*X[2]]
        print("Target: {}, Source: {}".format((x_, y_), X[:2]))
    
    print('----- Data Matrix Constructed ----------------------')
    print('A:', A.shape)

    # TODO: implement homography (tip: use np.linalg.svd)
    U,s,Vh = np.linalg.svd(A)
    H = Vh[np.argmin(s)].reshape(3,3)
    H = np.linalg.inv(T_image2)@H@T_image1
    
    print('----- Homography1 Matrix Constructed ----------------------')
    print('H:', H)
    print('H_norm:', H / np.linalg.norm(H))

    trans = np.matmul(H, source_points.T).T
    trans = trans / trans[:, -1][:, np.newaxis]
    print('points transformed', trans[:5])
    print('points origin(img2)', target_points[:5])
    
    return H


def main():
    # Arguments for image option
    import argparse
    
    # np.set_printoptions(suppress=True)
    
    parser = argparse.ArgumentParser(description='Image name')
    parser.add_argument('--imgs', type=str, default='c')
    parser.add_argument('--ratio', type=float, default=0.2)
    args = parser.parse_args()
    print('[+] Image Alignment Start... Image option:', args.imgs)

    global img_show, points, point_color
    # 1. Prepare points
    image1 = cv2.imread('imgs/{}1.jpg'.format(args.imgs))
    image2 = cv2.imread('imgs/{}2.jpg'.format(args.imgs))
    image3 = cv2.imread('imgs/{}3.jpg'.format(args.imgs))

    img_ratio = args.ratio
    if img_ratio != 1.0:
        image1 = cv2.resize(image1, dsize=(0, 0), fx=img_ratio, fy=img_ratio)
        image2 = cv2.resize(image2, dsize=(0, 0), fx=img_ratio, fy=img_ratio)
        image3 = cv2.resize(image3, dsize=(0, 0), fx=img_ratio, fy=img_ratio)

    H, W, C = image1.shape  # (4032, 3024, 3) - (H, W, C)
    
    cv2.namedWindow('image')    
    cv2.setMouseCallback('image', draw_circle_on_click)

    # 1) get points (1 -> 2)
    points = []
    img_show = np.concatenate((image1, image2), axis=1)
    point_color = (0, 0, 255)
    while (cv2.waitKey(1) & 0xFF) != 27:
        point_color = (0, 0, 255) if len(points) % 2 == 0 else (255, 0, 0)
        cv2.imshow('image', img_show)
    
    img1_points = np.array([points[i] for i in range(0, len(points), 2)], dtype=np.float)
    img2_points_1 = np.array([points[i] for i in range(1, len(points), 2)], dtype=np.float)-np.array([W,0,0])
        
    print('[+] Points Collected...')
    print('Num of img1_points:', len(img1_points))
    print('Num of img2_points:', len(img2_points_1))
    
    # 1) get points (3 -> 1)
    points = []
    img_show = np.concatenate((image3, image2), axis=1)
    point_color = (0, 0, 255)
    while (cv2.waitKey(1) & 0xFF) != 27:
        point_color = (0, 0, 255) if len(points) % 2 == 0 else (255, 0, 0)
        cv2.imshow('image', img_show)
    cv2.destroyAllWindows()
    
    img3_points = np.array([points[i] for i in range(0, len(points), 2)], dtype=np.float)
    img2_points_2 =  np.array([points[i] for i in range(1, len(points), 2)], dtype=np.float)-np.array([W,0,0])
    
    print('[+] Points Collected...')
    print('Num of img3_points:', len(img3_points))
    print('Num of img2_points:', len(img2_points_2))
    
    print('[+] Finding Homography...')
    H1_2 = find_homography(img1_points, img2_points_1)
    H3_2 = find_homography(img3_points, img2_points_2)
    
    print('[+] Homography Found!')
    print('H1_2:', H1_2)
    print('H3_2:', H3_2)
    
    print('[+] Warp Image...')
    img1_warped, img1_base = image_transform(image1, H1_2)
    img2_warped, img2_base = image2.copy(), np.array([0., 0., 1.])
    img3_warped, img3_base = image_transform(image3, H3_2)

    print('[+] Finished!')
    img_comb = combine_images([img1_warped, img3_warped, img2_warped], [img1_base, img3_base, img2_base])

    cv2.imwrite('combine%s.png'%args.imgs.upper(),img_comb)
    cv2.imshow('Combined', img_comb.astype(np.uint8))
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    main()