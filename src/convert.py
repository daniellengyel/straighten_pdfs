import cv2
import numpy as np
import yaml
import os

def get_lines(img_in_path, config, should_save=True):
    img = cv2.imread(img_in_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,config["threshold1"],config["threshold2"],apertureSize = config["apertureSize"])

    num_points = config["num_points"]
    succeeded = False
    while not succeeded:
        try:
            lines = cv2.HoughLines(edges,config["rho"],config["theta"],num_points) # binary search the best points.
            assert lines.shape[0] > 0
            succeeded = True
        except:
            num_points -= 50
    line_repr = []

    # find lines
    for l in lines:
        rho = l[0][0]
        theta = l[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho

        line_repr.append([a, b, x0, y0])

        x1 = int(0)
        y1 = int(get_y(a, b, x0, y0, x1))
        x2 = int(img.shape[1])
        y2 = int(get_y(a, b, x0, y0, x2))

        #     x1 = int(x0 + 5000*(-b))
        #     y1 = int(y0 + 5000*(a))
        #     x2 = int(x0 - 5000*(-b))
        #     y2 = int(y0 - 5000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    if should_save:
        cv2.imwrite("./temp_lines.jpg",img)

    return line_repr

# Find line with nth lowest y coordinate (highest from top). 0 indexed
def find_n_line(lines, x_compare, nth):
    sort_lines = []
    for l in lines:
        a, b, x0, y0 = l
        # TODO check for too vertical lines
        if a > b:
            continue
        y_compare = get_y(a, b, x0, y0, x_compare)
        sort_lines.append([y_compare, l])
    sort_lines.sort(key=lambda x: x[0])

    return sort_lines[nth][1]

# y is assumed to be at x_compare
def find_close_lines(lines, y, x_compare, threshold):
    close_lines = []
    for l in lines:
        a, b, x0, y0 = l
        y0_adjusted = get_y(a, b, x0, y0, x_compare)
        if abs(y0 - y) < threshold:
            close_lines.append(l)
    return close_lines

def average_line(lines):
    return np.mean(lines, axis=0)

def get_y(a, b, x0, y0, x):
    if b < 1e-2:
        return y0
    return -a*(x - x0)/b + y0

def dis(x0, y0, x1, y1):
    return pow((x0-x1)**2 + (y0-y1)**2, 0.5)

def is_y_neighborhood_colored(img, p, delta):
    for i in range(delta+1):
        if is_black(img, [p[0]+i, p[1]]):
            return True
        if is_black(img, [p[0]-i, p[1]]):
            return True

def is_black(img, p):
    return all(img[p[0], p[1]] < 200)

def find_line_start_end(line, img, config):
    a, b, x0, y0 = line

    x_threshold, y_threshold = config["x_threshold"], config["y_threshold"]

    line_end_points = []
    start = None
    last_seen = None
    for x in range(img.shape[1]):
        y = int(get_y(a, b, x0, y0, x))

        p = [y, x] # ?
        if is_y_neighborhood_colored(img, p, y_threshold):
            if start is None:
                start = p
                last_seen = p
            else:
                last_seen = p
        else:
            if last_seen is not None:
                if dis(last_seen[1], last_seen[0], x, y) > x_threshold:
                    line_end_points.append([start, last_seen])
                    start = None
                    last_seen = None
    if last_seen is not None:
        line_end_points.append([start, last_seen])

    return line_end_points

def _connect_endpoints(line_end_points, x0, x1):
    l = line_end_points[0]
    start = None
    end = None
    for i in range(len(line_end_points)):
        l = line_end_points[i]
        if (x0 <= l[0][1]) and (start is None):
            start = l[0]
        if (x1 <= l[1][1]) and (end is None):
            end = line_end_points[i-1][1]

    if (end is None) and (x1 >= line_end_points[-1][1][1]):
        end = line_end_points[-1][1]
    return [start, end]

def find_accurate_y(img, p, delta):
    start = None
    last_seen = None
    for i in range(-delta, delta+1):
        if is_black(img, [p[0]+i, p[1]]):
            if start is None:
                start = p[0]+i
                last_seen = p[0]+i
            else:
                last_seen = p[0]+i
    return int((start + last_seen)/2.)

# def process_image(from_p1, from_p2, to_p1, to_p2):
#     from_angle =

def get_perpendicular(p1, p2, dist):
    x_vec = p2[1] - p1 [1]
    y_vec = p2[0] - p1[0]
    y = -x_vec*dist/pow((x_vec**2 + y_vec**2), 0.5)
    x = y_vec*dist/pow((x_vec**2 + y_vec**2), 0.5)

#     y = round(y)
#     x = round(x)
    return [int(y), int(x)]

def convert(config):
    img = cv2.imread(config["img_in_path"])
    lines = get_lines(config["img_in_path"], config["find_lines"], False)


    # top_line = find_n_line(lines, 0, config["nth_line"])
    close_lines = find_close_lines(lines, config["ref_line"]["y"], config["ref_line"]["x"], config["ref_line"]["y_threshold"])


    avg_line = average_line(close_lines)
    start_end_multiple = find_line_start_end(avg_line, img, config["find_line_start_end"])
    # connect endpoints according to config
    if (config["find_line_start_end"]["x0"] == "") or (config["find_line_start_end"]["x1"] == ""):
        line_end_point = start_end_multiple[-1]
    else:
        line_end_point = _connect_endpoints(start_end_multiple, config["find_line_start_end"]["x0"], config["find_line_start_end"]["x1"])

    start, end = line_end_point

    x0 = start[1]
    y0 = find_accurate_y(img, start, 7)
    x1 = end[1]
    y1 = find_accurate_y(img, end, 7)

#     config = {
#     "p1": (217, 386 ), # x, y
#     "p2": (3136, 386   ),
#     "p3": (217, 386 + 20 )
#     }
#     config = {
#     "p1": (217, 386 ), # x, y
#     "p2": (4200, 386   ),
#     "p3": (217, 386 + 20 )
#     }

    ref_points = config["ref_points"]

    img = cv2.imread(config["img_in_path"])
    rows,cols,ch = img.shape

    from_p1 = [x0 , y0]
    from_p2 = [x1, y1]
    from_p3_vector = get_perpendicular(from_p1, from_p2, ref_points["perp_length"])

    from_p3 = [from_p1[0] + from_p3_vector[0], from_p1[1] + ref_points["perp_length"]]

    pts1 = np.float32([[from_p1, from_p2, from_p3]])
    ref_point_p3 = [ref_points["p1"][0], ref_points["p1"][1] + ref_points["perp_length"]]
    pts2 = np.float32([ref_points["p1"],ref_points["p2"],ref_point_p3])

    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))

#     cv2.line(dst,config["p1"],config["p2"],(0,0,255),2)

    # Crop
    # y
#     dst = dst[config["p1"][1] - 10:2120, :]
    dst = dst[ref_points["p1"][1] - 10:config["crop_lengths"]["y_length"], :]

    # right x
    dst = dst[:, :ref_points["p2"][0] + config["crop_lengths"]["pad"]]
    # left x
    dst = dst[:, ref_points["p1"][0] - config["crop_lengths"]["pad"]:]
    cv2.imwrite(config["img_out_path"], dst)


def main():
    folder_name = "116-1995"
    # folder_name = "119-1992"
    # folder_name = "44-198"
    # folder_name = "45-1987"
    folder_path = "../" + folder_name

    with open("./configs/1.yml", "r") as f:
        config = yaml.load(f)

    if not os.path.exists(folder_path + "_corrected"):
        os.mkdir(folder_path + "_corrected")

    # always have a fresh log
    log_path = "./logs/{}.txt".format(folder_name)
    if os.path.exists(log_path):
        os.remove(log_path)

    end_page = 332
    # end_page = 388
    # end_page = 245
    # end_page = 226
    for i in range(1, end_page + 1):
        im_path = folder_path + "/" + "Page_{0:0=3d}.jpg".format(i)
        # im_path = folder_path + "/" + "44 - 1986_Page_{0:0=3d}.jpg".format(i)
        # im_path = folder_path + "/" + "45 - 1987_Page_{0:0=3d}_Image_0001.jpg".format(i)


        print(im_path)

        out_path = folder_path + "_corrected" + "/{}.jpg".format(i)
        config["img_in_path"] = im_path
        config["img_out_path"] = out_path
        try:
            convert(config)
        except:
            print("Problem with current file. Skipping.")
            with open(log_path, "a+") as f:
                f.write("File {} didn't work. \n".format(i))

if __name__ == "__main__":
    main()
