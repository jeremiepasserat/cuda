#include <opencv2/opencv.hpp>
#include <vector>

__global__ void grayscaleShared ( unsigned char * in,   unsigned char * out, std::size_t w, std::size_t h) {

  auto i = blockIdx.x * (blockDim.x-2) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-2) + threadIdx.y;

  auto li = threadIdx.x;
  auto lj = threadIdx.y;

  extern __shared__ unsigned char sh[];

  if( i < w && j < h ) {

    sh[ (lj * blockDim.x + li) ] = (
      307 * in[ 3 * ( j * w + i ) ]
      + 604 * in[ 3 * ( j * w + i ) + 1 ]
      + 113 * in[  3 * ( j * w + i ) + 2 ]
    ) / 1024;

    __syncthreads();

    out[(j * w + i)] = sh[(lj * blockDim.x + li)];

  }
}

int main()
{
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;

  std::vector< unsigned char > g( rows * cols );
  cv::Mat m_out( rows, cols, CV_8UC1, g.data() );
  unsigned char * rgb_d;
  unsigned char * out;

  cudaMalloc( &rgb_d, 3 * rows * cols);
  cudaMalloc( &out, rows * cols );

  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );
  dim3 t( 32, 32 );
  dim3 bu(  (( cols - 1) / (t.x-2) + 1) , ( rows - 1 ) / (t.y-2) + 1 );

  // dim3 t( 16, 16 );
  // dim3 bu(  2 * (( cols - 1) / (t.x-2) + 1) , (2 * rows - 1 ) / (t.y-2) + 1 );

  // dim3 t( 4, 4 );
  // dim3 bu(  8 *(( cols - 1) / (t.x-2) + 1) , (8 * rows - 1 ) / (t.y-2) + 1 );

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  cudaEventRecord( start );

  grayscaleShared<<< bu, t, t.x*t.y >>>( rgb_d, out, cols, rows );
  cudaMemcpy(g.data(), out, rows * cols, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  auto cudaError = cudaGetLastError();

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

  cv::imwrite( "outGrayscaleShared.jpg", m_out );
  cudaFree( rgb_d);
  cudaFree ( out);


  return 0;
}
