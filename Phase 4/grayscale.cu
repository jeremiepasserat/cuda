#include <opencv2/opencv.hpp>
#include <vector>

__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if( i < cols && j < rows ) {
    g[ j * cols + i ] = (
      307 * rgb[ 3 * ( j * cols + i ) ]
      + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
      + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
    ) / 1024;
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

  std::size_t size = m_in.cols * m_in.rows;
  std::size_t sizeRGB = 3 * m_in.cols * m_in.rows;


//  cudaHostRegister(g.data(), size, cudaHostRegisterDefault);

  cudaMalloc( &rgb_d, 3 * rows * cols);
  cudaMalloc( &out, rows * cols );

  // Streams declaration.
  cudaStream_t streams[ 2 ];

  // Creation.
  cudaStreamCreate( &streams[ 0 ] );
  cudaStreamCreate( &streams[ 1 ] );


  cudaMemcpyAsync( rgb_d, rgb, sizeRGB/2, cudaMemcpyHostToDevice, streams[ 0 ] );
  cudaMemcpyAsync( rgb_d+sizeRGB/2, rgb+sizeRGB/2, sizeRGB/2, cudaMemcpyHostToDevice, streams[ 1 ] );


  // dim3 t( 32, 32 );
  // dim3 be( (( cols ) / ((t.x - 2) + 1) ), (( rows ) / ((t.y - 2) + 1) ));
  // dim3 t( 16, 16 );
  // dim3 be(  2 * (( cols ) / ((t.x - 2) + 1) ), (2 * ( rows ) / ((t.y - 2) + 1) ));
  dim3 t( 4, 4 );
  dim3 be( 8 * (( cols ) / ((t.x - 2) + 1) ), (8 * ( rows ) / ((t.y - 2) + 1) ));


  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );

  // One kernel is launched in each stream.
  grayscale<<< be, t, 0, streams[ 0 ] >>>( rgb_d, out, cols, rows);
  grayscale<<< be, t, 0, streams[ 1 ] >>>( rgb_d+sizeRGB/2, out+size/2, cols, rows);


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



  cv::imwrite( "outGrayscale.jpg", m_out );
  cudaFree( rgb_d);
  //cudaFree( g_d);
  cudaFree ( out);


  return 0;
}
