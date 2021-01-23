import numpy as np
from PIL import Image,ImageOps

# Convolutional Function
def convolution(image, kernel):
    m, n = kernel.shape
    if (m == n):
        x, y = image.shape
        x = x - m + 1
        y = y - m + 1
        resultImg = np.zeros((x,y))
        for i in range(x):
            for j in range(y):
                resultImg[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)
    return resultImg



if __name__ == '__main__':
    # Image Path
    IMAGE = "sample images/lena.png"

    # Opening Image using PIL
    img = Image.open(IMAGE)

    # Converting RGB to grayscale image
    grayImg = ImageOps.grayscale(img)

    # Converting grayscale img into array
    imgArr = np.asarray(grayImg, dtype="int32")

    # Padding Image with zeros
    imgArr = np.pad(imgArr, 1, mode='constant')

    '''
    3x3 Gaussian Kernal
    Kernel = 1/25 x [[1 2 1],[2,4,2],[1,2,1]
    '''

    # kernel = np.array([[0.0625,0.125,0.0625],[0.125,0.25,0.125],[0.0625,0.125,0.0625]])

    '''
    5x5 Gaussian Kernel
    Kernel = 1/273 x [[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]]
    '''
    kernel = np.array([[0.0037, 0.0147, 0.0256, 0.0147, 0.0037],
                       [0.0147, 0.0586, 0.0952, 0.0586, 0.0147],
                       [0.0256, 0.0952, 0.1501, 0.0952, 0.0256],
                       [0.0147, 0.0586, 0.0952, 0.0586, 0.0147],
                       [0.0037, 0.0147, 0.0256, 0.0147, 0.0037]])

    # Applying Convolution using our kernel Function
    dstImg = convolution(imgArr, kernel)

    # Converting array to back to image
    dstImg = Image.fromarray(dstImg)

    # Displaying both original and resultant images
    grayImg.show()
    dstImg.show()


