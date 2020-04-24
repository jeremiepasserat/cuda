CXX=g++
CXXFLAGS= -g -G -O3 -march=native
LDLIBS=`pkg-config --libs opencv`

grayscale: grayscale.cu
	nvcc -o $@ $< $(LDLIBS)

sobel: sobel.cu
		nvcc -o $@ $< $(LDLIBS)

blur: blur.cu
	nvcc -o $@ $< $(LDLIBS)

blurGauss: blurGauss.cu
		nvcc -o $@ $< $(LDLIBS)

blurGaussSobel: blurGaussSobel.cu
			nvcc -o $@ $< $(LDLIBS)

erode: erode.cu
		nvcc -o $@ $< $(LDLIBS)

dilate: dilate.cu
		nvcc -o $@ $< $(LDLIBS)

emboss: emboss.cu
	nvcc -o $@ $< $(LDLIBS)

mike: mike.cu
	nvcc -o $@ $< $(LDLIBS)

laplacian: laplacian.cu
	nvcc -o $@ $< $(LDLIBS)


.PHONY: clean

clean:
	rm sobel blur blurGauss blurGaussSobel mike emboss erode dilate grayscale laplacian *.jpg
