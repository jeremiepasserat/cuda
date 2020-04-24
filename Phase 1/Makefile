CC=gcc
CFLAGS= -g -O3 -march=native
LDLIBS=-lm -lIL

sobel: sobel.c

blur: blur.c

blurGauss: blurGauss.c

blurGaussSobel: blurGaussSobel.c

mike: mike.c

emboss: emboss.c

erode: erode.c

dilate: dilate.c

grayscale: grayscale.c

laplacian: laplacian.c


.PHONY: clean

clean:
	rm sobel blur blurGauss blurGaussSobel mike emboss erode dilate grayscale laplacian *.jpg
