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


  //cudaMemcpyAsync( v1_d+size/2, v1+size/2, size/2 * sizeof(int), cudaMemcpyHostToDevice, streams[ 1 ] );
  dim3 t( 32, 32 );
  dim3 be((( cols) / ((t.x - 2) + 1) ), (( rows ) / ((t.y - 2) + 1) ));
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
  grayscaleShared<<< be, t, t.x * t.y, streams[ 0 ] >>>( rgb_d, out, cols + 2, rows);
  grayscaleShared<<< be, t, t.x * t.y, streams[ 1 ] >>>( rgb_d+sizeRGB/2, out+size/2, cols + 2, rows);


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



  cv::imwrite( "outGrayscaleShared.jpg", m_out );
  cudaFree( rgb_d);
  //cudaFree( g_d);
  cudaFree ( out);


  return 0;
}
