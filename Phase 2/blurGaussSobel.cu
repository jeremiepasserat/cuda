#include <opencv2/opencv.hpp>
#include <vector>

__global__ void blurGauss ( unsigned char * data,   unsigned char * out, std::size_t cols, std::size_t rows) {

  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;


  if ( i > 1 && i < (cols - 2) && j > 1 && j < (rows - 2)) {



    for (auto c = 0; c < 3; ++c){

      auto gu =  data[((j - 2) * cols + i - 2) * 3 + c] + 4 * data[((j - 2)  * cols + i - 1) * 3 + c]
      + 7 * data[((j - 2) * cols + i) * 3 + c]
      + 4 * data[((j - 2) * cols + i + 1) * 3 + c] + data[((j - 2)  * cols + i + 2) * 3 + c]
      + 4 * data[((j - 1) * cols + i  - 2) * 3 + c] + 7 * data[((j - 1)  * cols + i - 1) * 3 + c]
      + 26 * data[((j - 1) * cols + i) * 3 + c]
      + 7 * data[((j - 1) * cols + i + 1) * 3 + c] + 4 * data[((j - 1)  * cols + i + 2) * 3 + c]
      + 7 * data[((j) * cols + i - 2) * 3 + c] + 26 * data[((j)  * cols + i - 1) * 3 + c]
      + 41 * data[((j) * cols + i) * 3 + c]
      + 26 * data[((j) * cols + i + 1) * 3 + c] + 7 * data[((j)  * cols + i + 2) * 3 + c]
      + 4 * data[((j + 1) * cols + i - 2) * 3 + c] + 16 * data[((j + 1)  * cols + i - 1) * 3 + c]
      + 26 * data[((j + 1) * cols + i) * 3 + c]
      + 16 * data[((j + 1) * cols + i + 1) * 3 + c] + 4 * data[((j + 1)  * cols + i + 2) * 3 + c]
      + data[((j + 2) * cols + i - 2) * 3 + c] + 4 * data[((j + 2)  * cols + i - 1) * 3 + c]
      + 7 * data[((j + 2) * cols + i) * 3 + c]
      + 4 * data[((j + 2) * cols + i + 1) * 3 + c] + data[((j + 2)  * cols + i + 2) * 3 + c];

      out[(j * cols + i) * 3 + c] = (gu / 273);

    }
  }
}

__global__ void sobel ( unsigned char * data,   unsigned char * out, std::size_t cols, std::size_t rows) {

  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;


  if ( i > 0 && i < (cols - 1) && j > 0 && j < (rows - 1)) {



    for (auto c = 0; c < 3; ++c){

      auto h =   data[((j - 1) * cols + i - 1) * 3 + c] -     data[((j - 1) * cols + i + 1) * 3 + c]
      + 2 * data[( j      * cols + i - 1) * 3 + c] - 2 * data[( j      * cols + i + 1) * 3 + c]
      +     data[((j + 1) * cols + i - 1) * 3 + c] -     data[((j + 1) * cols + i + 1) * 3 + c];

      auto v =   data[((j - 1) * cols + i - 1) * 3 + c] -     data[((j + 1) * cols + i - 1) * 3 + c]
      + 2 * data[((j - 1) * cols + i    ) * 3 + c] - 2 * data[((j + 1) * cols + i    ) * 3 + c]
      +     data[((j - 1) * cols + i + 1) * 3 + c] -     data[((j + 1) * cols + i + 1) * 3 + c];

      auto res = h*h + v*v;
      res = res > 255*255 ? res = 255*255 : res;

      out[(j * cols + i) * 3 + c] = sqrt((float) res);



    }
  }

}

int main()
{
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;

  std::vector< unsigned char > g( 3 * rows * cols );
  cv::Mat m_out( rows, cols, CV_8UC3, g.data() );
  unsigned char * rgb_d;
	unsigned char * g_d;
  unsigned char * out;

  cudaMalloc( &rgb_d, 3 * rows * cols);
  cudaMalloc( &g_d, 3 * rows * cols);
  cudaMalloc( &out, 3 * rows * cols );

  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );
  // dim3 t( 32, 32 );
  // dim3 be(( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );

  dim3 t( 16, 16 );
  dim3 be( 3 * 2 * (( cols - 1) / t.x + 1 ), 2 * (( rows - 1 ) / t.y + 1 ));

  // dim3 t( 4, 4 );
  // dim3 be( 3 * 8 * (( cols - 1) / t.x + 1 ), 8 * (( rows - 1 ) / t.y + 1 ));

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  cudaEventRecord( start );

  blurGauss<<< be, t >>>( rgb_d, g_d, cols, rows );
  sobel<<< be, t >>>( g_d, out, cols, rows );

  cudaMemcpy(g.data(), out, 3 * rows * cols, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  auto cudaError = cudaGetLastError();

  // Si pas d'erreur détectée dans le bordel ben on aura cudaSuccess
  if (cudaError != cudaSuccess){

    std::cout << cudaGetErrorName(cudaError) << std::endl;
    std::cout << cudaGetErrorString(cudaError) << std::endl;
  }


  else {
    std::cout << "Aucune erreur" << std::endl;

  }
  cudaEventRecord( stop );
  cudaEventSynchronize( stop );

  float duration = 0.0f;
  cudaEventElapsedTime( &duration, start, stop );

  std::cout << "Total: " << duration << "ms\n";



  cv::imwrite( "outBlurGaussSobel.jpg", m_out );
  cudaFree( rgb_d);
  cudaFree( g_d);
  cudaFree ( out);


  return 0;
}
