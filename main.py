import cv2

# Load the image and template
img = cv2.imread('./wheres_waldo.png')
template = cv2.imread('./waldo_face.png')

# Convert the image and template to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Match the template in the image
res = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# Get the coordinates of the box around the matched template
top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

# Draw a rectangle around the matched template
cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

# Draw an arrow pointing at the box
x_center = (top_left[0] + bottom_right[0]) // 2
y_center = (top_left[1] + bottom_right[1]) // 2
cv2.arrowedLine(img, (x_center, y_center), (x_center + 50, y_center - 50), (0, 0, 255), 2)

# Write "There's Waldo!" onto the box with yellow background
text = "There's Waldo!"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2
text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
text_width, text_height = text_size

text_box = ((x_center + 50, y_center - 50), (x_center + 50 + text_width, y_center - 50 - text_height))
cv2.rectangle(img, text_box[0], text_box[1], (0, 255, 255), -1)
cv2.putText(img, text, (x_center + 50, y_center - 50), font, font_scale, (0, 0, 255), thickness)

# Display the result
cv2.imshow("There's Waldo!", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
