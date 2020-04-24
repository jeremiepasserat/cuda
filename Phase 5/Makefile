CXX=g++
CXXFLAGS= -g -G -O3 -march=native
LDLIBS=`pkg-config --libs opencv`

grayscaleShared: grayscaleShared.cu
	nvcc -o $@ $< $(LDLIBS)

sobelShared: sobelShared.cu
		nvcc -o $@ $< $(LDLIBS)

blurShared: blurShared.cu
	nvcc -o $@ $< $(LDLIBS)

blurGaussShared: blurGaussShared.cu
		nvcc -o $@ $< $(LDLIBS)

blurSobelShared: blurSobelShared.cu
			nvcc -o $@ $< $(LDLIBS)

erodeShared: erodeShared.cu
		nvcc -o $@ $< $(LDLIBS)

dilateShared: dilateShared.cu
		nvcc -o $@ $< $(LDLIBS)

embossShared: embossShared.cu
	nvcc -o $@ $< $(LDLIBS)

mikeShared: mikeShared.cu
	nvcc -o $@ $< $(LDLIBS)

laplacianShared: laplacianShared.cu
	nvcc -o $@ $< $(LDLIBS)


.PHONY: clean

clean:
	rm sobelShared blurShared blurGaussShared blurSobelShared mikeShared embossShared erodeShared dilateShared grayscaleShared laplacianShared *.jpg
