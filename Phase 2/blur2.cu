#include <opencv2/opencv.hpp>
#include <vector>

__global__ void blur2 ( unsigned char * data,   unsigned char * r,  unsigned char * g,  unsigned char * b, std::size_t cols, std::size_t rows) {

  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;

  if ( i > 0 && i < (cols - 1) && j > 0 && j < (rows - 1)) {

    for (auto c = 0; c < 3; ++c){

      auto gu =     data[((j - 1) * cols + i - 1) * 3 + c] +     data[((j - 1) * cols + i + 1) * 3 + c]
      +     data[( j      * cols + i - 1) * 3 + c] +     data[( j      * cols + i + 1) * 3 + c]
      +     data[((j + 1) * cols + i - 1) * 3 + c] +     data[((j + 1) * cols + i + 1) * 3 + c]
      +     data[(( j - 1) * cols + i) * 3 + c]     +     data[( j      * cols + i) * 3 + c]
      +     data[(( j + 1) * cols + i) * 3 + c];


      if (c == 2)
      b[(j * cols + i)] = (gu/9);

      else if (c == 1)

      g[(j * cols + i)] =(gu/9);

      else if (c == 0)

      r[(j * cols + i)] = (gu/9);
    }
  }
}


int main()
{
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;

  std::vector< unsigned char > r_out( rows * cols );
  cv::Mat r_mat( rows, cols, CV_8UC1, r_out.data() );
  std::vector< unsigned char > g_out( rows * cols );
  cv::Mat g_mat( rows, cols, CV_8UC1, g_out.data() );
  std::vector< unsigned char > b_out( rows * cols );
  cv::Mat b_mat( rows, cols, CV_8UC1, b_out.data() );
  std::vector<cv::Mat> toMerge;
  cv::Mat m_out;
  unsigned char * rgb_d;
  unsigned char * r;
  unsigned char * g;
  unsigned char * b;

  cudaMalloc( &rgb_d, 3 * rows * cols);
  cudaMalloc( &r, rows * cols );
  cudaMalloc( &g, rows * cols );
  cudaMalloc( &b, rows * cols );

  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );
  dim3 t( 32, 32 );
  dim3 be(( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );

  // dim3 t( 16, 16 );
  // dim3 be( 3 * 2 * (( cols - 1) / t.x + 1 ), 2 * (( rows - 1 ) / t.y + 1 ));

  // dim3 t( 1, 1 );
  // dim3 be( 3 * 32 * (( cols - 1) / t.x + 1 ), 32 * (( rows - 1 ) / t.y + 1 ));

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  cudaEventRecord( start );

  blur2<<< be, t >>>( rgb_d, r, g, b, cols, rows );

  cudaMemcpy(r_out.data(), r, rows * cols, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_out.data(), g, rows * cols, cudaMemcpyDeviceToHost);
  cudaMemcpy(b_out.data(), b, rows * cols, cudaMemcpyDeviceToHost);

  toMerge.push_back(r_mat);
  toMerge.push_back(g_mat);
  toMerge.push_back(b_mat);

  cv::merge(toMerge, m_out);

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



  cv::imwrite( "outBlur2.jpg", m_out);
  cudaFree( rgb_d);
  //cudaFree( g_d);
  cudaFree ( r);
  cudaFree ( g);
  cudaFree ( b);


  return 0;
}
