#include <opencv2/opencv.hpp>
#include <vector>

__global__ void laplacian ( unsigned char * data,   unsigned char * out, std::size_t cols, std::size_t rows) {

  //auto i = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
  //auto j = blockIdx.y * blockDim.y + threadIdx.y;

  auto i = blockIdx.x * (blockDim.x) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y) + threadIdx.y;

  if ( i > 0 && i < (cols - 1) && j > 0 && j < (rows - 1)) {

    for (auto c = 0; c < 3; ++c){

      auto gu =     data[((j - 1) * cols + i - 1) * 3 + c] * 0 +     data[((j - 1) * cols + i + 1) * 3 + c] * 0
      +     data[( j      * cols + i - 1) * 3 + c] * -1 +     data[( j      * cols + i + 1) * 3 + c] * -1
      +     data[((j + 1) * cols + i - 1) * 3 + c] * 0 +     data[((j + 1) * cols + i + 1) * 3 + c] * 0
      +     data[(( j - 1) * cols + i) * 3 + c] * -1     +     data[( j      * cols + i) * 3 + c] * 4
      +     data[(( j + 1) * cols + i) * 3 + c] * -1;

      out[(j * cols + i) * 3 + c] = (gu / 9);



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
  unsigned char * out;

  std::size_t size = 3 * m_in.cols * m_in.rows;

//  cudaHostRegister(g.data(), size, cudaHostRegisterDefault);

  cudaMalloc( &rgb_d, 3 * rows * cols);
  cudaMalloc( &out, 3 * rows * cols );

  // Streams declaration.
  cudaStream_t streams[ 2 ];

  // Creation.
  cudaStreamCreate( &streams[ 0 ] );
  cudaStreamCreate( &streams[ 1 ] );

  cudaMemcpyAsync( rgb_d, rgb, size/2, cudaMemcpyHostToDevice, streams[ 0 ] );
  cudaMemcpyAsync( rgb_d+size/2, rgb+size/2, size/2, cudaMemcpyHostToDevice, streams[ 1 ] );

  //cudaMemcpyAsync( v1_d+size/2, v1+size/2, size/2 * sizeof(int), cudaMemcpyHostToDevice, streams[ 1 ] );
  dim3 t( 32, 32 );
  dim3 be( 3 * (( cols) / ((t.x - 2) + 1) ), (( rows) / ((t.y - 2) + 1) ));
  // std::cout << "semi cols" << (m_in.cols / 2) << std::endl;
  // std::cout << "semi rows" << (m_in.rows / 2) << std::endl;
  // std::cout << "be x" << 3 * (( cols / 2 - 1) / (t.x + 1)) << std::endl;
  //
  // exit(0);

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );

  // One kernel is launched in each stream.
  laplacian<<< be, t, 0, streams[ 0 ] >>>( rgb_d, out, cols, rows / 2 + 2);
  soblaplacianel<<< be, t, 0, streams[ 1 ] >>>( rgb_d+size/2, out+size/2, cols, rows / 2);

  // Sending back the resulting vector by halves.
  cudaMemcpyAsync( g.data(), out, size/2, cudaMemcpyDeviceToHost, streams[ 0 ] );
  cudaMemcpyAsync( g.data()+size/2, out+size/2, size/2, cudaMemcpyDeviceToHost, streams[ 1 ] );


  // Synchronize everything.
  cudaDeviceSynchronize();

  // Destroy streams.
  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);

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



  cv::imwrite( "outLaplacian.jpg", m_out );
  cudaFree( rgb_d);
  //cudaFree( g_d);
  cudaFree ( out);


  return 0;
}
