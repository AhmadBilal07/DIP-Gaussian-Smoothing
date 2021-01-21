import cv2 as cv

# Opening Image
img = cv.imread('sample images/lena.png')

# Applying Gaussian Filter
result = cv.GaussianBlur(img,(5,5),cv.BORDER_DEFAULT)

# Displaying both Original and Smoothened Images
cv.imshow("Orginal Image", img)
cv.imshow("Gausian Smoothen Result", result)

# Waits for a key Press
cv.waitKey(0)

# destroys the window showing image
cv.destroyAllWindows()