import cv2
import numpy as np


kernel = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], np.uint8)


def preprocess_img(img):
    outerBox = cv2.GaussianBlur(img, (11, 11), 0)
    outerBox = cv2.adaptiveThreshold(
        outerBox, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    outerBox = cv2.bitwise_not(outerBox)
    outerBox = cv2.dilate(outerBox, kernel)
    return outerBox


def merge_related_lines(lines, img):
    for current in lines:
        current = current[0]

        if current[0] == 0 and current[1] == -100:
            continue

        p1 = current[0]
        theta1 = current[1]

        pt1current, pt2current = {x: 0, y: 0}, {x: 0, y: 0}
        if np.pi * 45 / 180 < theta1 < np.pi * 135 / 180:
            pt1current[x] = 0
            pt1current[y] = p1 / np.sin(theta1)

            pt2current[x] = img.shape[1]
            pt2current[y] = -pt2current[x] / np.tan(theta1) + p1 / np.sin(theta1)

        else:
            pt1current[y] = 0
            pt2current[x] = p1 / np.cos(theta1)

            pt2current[y] = img.shape[0]
            pt2current[x] = - pt2current[y] / np.tan(theta1) + p1 / np.cos(theta1)

        for pos in lines:
            pos = pos[0]
            if np.all(pos == current):
                continue
            if abs(pos[0] - current[0]) < 20 and abs(pos[1] - current[1]) < np.pi * 10 / 180:
                p = pos[0]
                theta = pos[1]

                pt1, pt2 = {x: 0, y: 0}, {x: 0, y: 0}
                if np.pi * 45 / 180 < pos[1] < np.pi * 135 / 180:
                    pt1[x] = 0
                    pt1[y] = p / np.sin(theta)

                    pt2[x] = img.shape[1]
                    pt2[y] = -pt2[x] / np.tan(theta) + p / np.sin(theta)
                else:
                    pt1[y] = 0
                    pt1[x] = p / np.cos(theta)

                    pt2[y] = img.shape[0]
                    pt2[x] = -pt2[y] / np.tan(theta) + p / np.cos(theta)

                if (pt1current[x] - pt1[x]) ** 2 + (pt1current[y] - pt1[y]) ** 2 < 64 * 64 and (
                        pt2current[x] - pt2[x]) ** 2 + (pt2current[y] - pt2[y]) ** 2 < 64 ** 2:
                    current[0] = (current[0] + pos[0]) / 2
                    current[1] = (current[1] + pos[1]) / 2

                    pos[0] = 0
                    pos[1] = -100


def draw_line(line, img, rgb=(0, 0, 255)):
    if line[1] != 0:
        m = int(round(-1 / np.tan(line[1])))
        c = int(round(line[0] / np.sin(line[1])))
        cv2.line(img, (0, c), (img.shape[1], m * img.shape[0] + c), rgb, 1)
    else:
        cv2.line(img, (int(line[0]), 0), (int(line[0]), img.shape[0]), rgb, 1)


def drawLine(img, lines):
    clone = img.copy()
    lines = np.squeeze(lines)

    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(clone, (x1, y1), (x2, y2), (0, 0, 255))

    return clone


img = cv2.imread("sudoku.png", 0)

# preprocess image
outerBox = preprocess_img(img)
#cv2.imwrite("step_1.jpg", outerBox)
# finding the biggest blob
count = 0
max = -1
maxPt = (0, 0)
h, w = outerBox.shape

for y in range(h):
    row = outerBox[y]
    for x in range(w):
        if row[x] >= 128:
            area = cv2.floodFill(outerBox, None, (x, y), 64)
            if area[0] > max:
                maxPt = (x, y)
                max = area[0]

cv2.floodFill(outerBox, None, maxPt, (255, 255, 255))

for y in range(h):
    row = outerBox[y]
    for x in range(w):
        if row[x] == 64 and x != maxPt[0] and y != maxPt[1]:
            area = cv2.floodFill(outerBox, None, (x, y), (0, 0, 0))

cv2.erode(outerBox, kernel)

# ---------------------------------------------------------------------------------
# detecting lines
lines = cv2.HoughLines(outerBox, 1, np.pi / 180, 200)

# merge related lines
merge_related_lines(lines, img)
for line in lines:
    draw_line(line[0], outerBox, (255, 0, 0))

# --------------------------------------------------------------------------------
topEdge = [1000, 1000]
bottomEdge = [-1000, -1000]
leftEged = [1000, 1000]
rightEdge = [-1000, -1000]

topYIntercept, topXIntercept = 100000, 0
bottomYIntercept, bottomXIntercept = 0, 0
leftXIntercept, leftYIntercept = 100000, 0
rightXIntercept, rightYIntercept = 0, 0

for line in lines:
    current = line[0]
    r, theta = current[0], current[1]

    if r == 0 and theta == -100:
        continue

    xIntercept, yIntercept = r / np.cos(theta), r / (np.sin(theta) * np.cos(theta))

    try:
        if np.pi * 80 / 180 < theta < np.pi * 100 / 180:
            if r < topEdge[0]:
                topEdge = current

            if r > bottomEdge[0]:
                bottomEdge = current

        elif theta < np.pi * 10 / 180 or theta > np.pi * 170 / 180:
            if xIntercept > rightXIntercept:
                rightEdge = current
                rightXIntercept = xIntercept
            elif xIntercept <= leftXIntercept:
                leftEdge = current
                leftXIntercept = xIntercept
    except ZeroDivisionError:
        continue

draw_line(topEdge, img, (0, 0, 0))
draw_line(bottomEdge, img, (0, 0, 0))
draw_line(leftEdge, img, (0, 0, 0))
draw_line(rightEdge, img, (0, 0, 0))

# --------------------------------------------------------------------------------------------
# --------------------- create 4 finds surround the puzzles --------------------------------
left1, left2, right1, right2, bottom1, bottom2, top1, top2 = [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0,
                                                                                                                      0]
h, w = outerBox.shape

if leftEdge[1] != 0:
    left1[0] = 0
    left2[0] = w
    left1[1] = leftEdge[0] / np.sin(leftEdge[1])
    left2[1] = - left2[0] / np.tan(leftEdge[1]) + left1[1]
else:
    left1[1] = 0
    left1[0] = leftEdge[0] / np.cos(leftEdge[1])
    left2[1] = h
    left2[0] = left1[0] - h * np.tan(leftEdge[1])

if rightEdge[1] != 0:
    right1[0] = 0
    right1[1] = rightEdge[0] / np.sin(rightEdge[1])
    right2[0] = w
    right2[1] = - right2[0] / np.tan(rightEdge[1]) + right1[1]
else:
    right1[1] = 0
    right1[0] = rightEdge[0] / np.sin(rightEdge[1])
    right2[1] = h
    right2[0] = right1[0] - h * np.tan(rightEdge[1])

bottom1[0] = 0
bottom1[1] = bottomEdge[0] / np.sin(bottomEdge[1])
bottom2[0] = w
bottom2[1] = - bottom2[0] / np.tan(bottomEdge[1]) + bottom1[1]

top1[0] = 0
top1[1] = topEdge[0] / np.sin(topEdge[1])
top2[0] = w
top2[1] = -top2[0] / np.tan(topEdge[1]) + top1[1]

# find intersect of 4 edges
leftA = left2[1] - left1[1]
leftB = left1[0] - left2[0]
leftC = leftA * left1[0] + leftB * left1[1]

rightA = right2[1] - right1[1]
rightB = right1[0] - right2[0]
rightC = rightA * right1[0] + rightB * right1[1]

topA = top2[1] - top1[1]
topB = top1[0] - top2[0]
topC = topA * top1[0] + topB * top1[1]

bottomA = bottom2[1] - bottom1[1]
bottomB = bottom1[0] - bottom2[0]
bottomC = bottomA * bottom1[0] + bottomB * bottom1[1]

# intersect of left and top
detTopLeft = leftA * topB - topA * leftB
ptTopLeft = (int(round((topB * leftC - leftB * topC) / detTopLeft)),
             int(round((leftA * topC - topA * leftC) / detTopLeft)))

# intersect of top and right
detTopRight = rightA * topB - rightB * topA
ptTopRight = (int(round((topB * rightC - rightB * topC) / detTopRight)),
              int(round((rightA * topC - topA * rightC) / detTopRight)))

# intersect of right and bottom
detBottomRight = rightA * bottomB - rightB * bottomA
ptBottomRight = (int(round((bottomB * rightC - rightB * bottomC) / detBottomRight)),
                 int(round((rightA * bottomC - bottomA * rightC) / detBottomRight)))

# intersect of left and bottom
detBottomLeft = leftA * bottomB - leftB * bottomA
ptBottomLeft = (int(round((bottomB * leftC - leftB * bottomC) / detBottomLeft)),
                int(round((leftA * bottomC - bottomA * leftC) / detBottomLeft)))

# -----------------------
cv2.line(img, ptTopRight, ptTopRight, (0, 0, 255), 10)
cv2.line(img, ptTopLeft, ptTopLeft, (0, 0, 255), 10)
cv2.line(img, ptBottomRight, ptBottomRight, (0, 0, 255), 10)
cv2.line(img, ptBottomLeft, ptBottomLeft, (0, 0, 255), 10)

# -----------------------------------------------------------------------------------------
# find the longest edge of puzzle
maxLength = (ptBottomLeft[0] - ptBottomRight[0]) ** 2 + (ptBottomLeft[1] - ptBottomRight[1]) ** 2
temp = (ptTopRight[0] - ptBottomRight[0]) ** 2 + (ptTopRight[1] - ptBottomRight[1]) ** 2

if temp > maxLength:
    maxLength = temp

temp = (ptTopRight[0] - ptTopLeft[0]) ** 2 + (ptTopRight[1] - ptTopLeft[1]) ** 2
if temp > maxLength:
    maxLength = temp

temp = (ptBottomLeft[0] - ptTopLeft[0]) ** 2 + (ptBottomLeft[1] - ptTopLeft[1]) ** 2
if temp > maxLength:
    maxLength = temp

maxLength = int(np.sqrt(maxLength))

src = [ptTopLeft, ptTopRight, ptBottomRight, ptBottomLeft]
src = np.array([list(i) for i in src], np.float32)
dst = np.array([[0, 0], [maxLength - 1, 0], [maxLength - 1, maxLength - 1], [0, maxLength - 1]], np.float32)

transform_perspective = cv2.getPerspectiveTransform(src, dst)
undistorted = cv2.warpPerspective(img, transform_perspective, (maxLength, maxLength))

# --------------------------------------------------------------
output_img = cv2.warpPerspective(
    img, transform_perspective, (maxLength, maxLength)
)

output_img = preprocess_img(output_img)

cv2.imwrite("output.jpg", output_img)

# ---------------------------------------------------------------------------------------------
# grid = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
grid = cv2.bitwise_not(
    cv2.adaptiveThreshold(undistorted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1))
edge_h, edge_w = np.shape(grid)
celledge_h, celledge_w = edge_h // 9, edge_w // 9

temp_grid = []
for i in range(celledge_h, edge_h + 1, celledge_h):
    for j in range(celledge_w, edge_w + 1, celledge_w):
        rows = grid[i - celledge_h:i]
        temp_grid.append([rows[k][j - celledge_w: j] for k in range(len(rows))])

# creating the 9x9 grid of images
final_grid = []
for i in range(0, len(temp_grid) - 8, 9):
    final_grid.append(temp_grid[i:i + 9])

# converting all the cell images to np.array
for i in range(9):
    for j in range(9):
        final_grid[i][j] = np.array(final_grid[i][j])

for i in range(9):
    for j in range(9):
        cv2.imwrite(str("BoardCells/cell" + str(i) + str(j) + ".jpg"), final_grid[i][j])


def recognize_digit(img):
    # neu so pixel > 0 nhieu -> co so
    white = np.sum(img > 0)
    black = np.sum(img == 0)
    return white / black > 0.275


digits = []
i = 0
for images in final_grid:
    d = []
    for image in images:
        d.append('x' if recognize_digit(image) else "  ")
    digits.append(d)

# cv2.imshow("undistorted", undistorted)

f = open('output.txt', 'a')
for digit in digits:
    for d in digit:
        f.write(d)
    f.write("\n")
f.close()


# ----------------------------------------------------------------------------------
# step 4: 9x9 grid
def draw_grid(img, grid_shape, color=(225, 0, 0), thickness=1):
    h, w = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    for x in np.linspace(start=dx, stop=w - dx, num=cols):
        x = int(round(x) + x * 0.1)
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    for y in np.linspace(start=dy, stop=h - dy, num=rows):
        y = int(round(y) + y * 0.1)
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img


final_img = draw_grid(undistorted, [9, 9])
# cv2.imshow("step 4", final_img)
cv2.waitKey(0)
