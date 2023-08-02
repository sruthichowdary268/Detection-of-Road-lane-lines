import cv2
import numpy as np

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def canny(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)

def draw_lines(image, lines, color=(255, 0, 0), thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def average_slope_intercept(lines):
    left_lines = []
    right_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

    left_slope, left_intercept = np.average(left_lines, axis=0)
    right_slope, right_intercept = np.average(right_lines, axis=0)

    return [(left_slope, left_intercept), (right_slope, right_intercept)]

def get_lane_lines(image, lines):
    lane_lines = []
    if lines is not None:
        lane_lines = average_slope_intercept(lines)

    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))

    left_line = make_line_points(y1, y2, lane_lines[0])
    right_line = make_line_points(y1, y2, lane_lines[1])

    return np.array([[left_line, right_line]])

def make_line_points(y1, y2, line):
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return x1, y1, x2, y2

def lane_detection(image):
    gray_image = grayscale(image)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = canny(blurred_image, 50, 150)

    height = image.shape[0]
    width = image.shape[1]
    vertices = np.array([[(100, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 20, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    lines = hough_lines(masked_edges, 1, np.pi / 180, 20, 10, 5)
    lane_lines = get_lane_lines(image, lines)

    line_image = np.zeros_like(image)
    draw_lines(line_image, lane_lines)

    return cv2.addWeighted(image, 0.8, line_image, 1, 0)

# Example usage:
if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"
    image = cv2.imread(image_path)
    lane_detected_image = lane_detection(image)
    cv2.imshow("Lane Detection", lane_detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
