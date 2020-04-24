#include <opencv2/opencv.hpp>
#include <vector>

__global__ void embossShared ( unsigned char * data,   unsigned char * out, std::size_t w, std::size_t h) {

  auto i = blockIdx.x * (blockDim.x-2) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-2) + threadIdx.y;

  auto li = threadIdx.x;
  auto lj = threadIdx.y;

  extern __shared__ unsigned char sh[];

  if( i < w && j < h ) {

    // on s'occupe du rouge
    sh[3 * (lj * blockDim.x + li) ] = data[ 3 * ( j * w + i ) ];
    sh[3 * (lj * blockDim.x + li) + 1 ] = data[ 3 * ( j * w + i ) + 1];
    sh[3 * (lj * blockDim.x + li) + 2 ] = data[ 3 * ( j * w + i ) + 2 ];

    __syncthreads();

    auto ww = blockDim.x;

    if( li > 0 && li < (blockDim.x - 1) && lj > 0 && lj < (blockDim.y - 1) )
    {
      for (auto c = 0; c < 3; ++c){

        auto gu =     sh[((lj - 1) * ww + li - 1) * 3 + c] * -18 + sh[((lj - 1) * ww + li + 1) * 3 + c] * 0
        +     sh[( lj      * ww + li - 1) * 3 + c] * -9 +     sh[( lj      * ww + li + 1) * 3 + c] * 9
        +     sh[((lj + 1) * ww + li - 1) * 3 + c] * 0  +     sh[((lj + 1) * ww + li + 1) * 3 + c] * 18
        +     sh[(( lj - 1) * ww + li) * 3 + c] * -9   +  9 *   sh[( lj      * ww + li) * 3 + c]
        +     sh[(( lj + 1) * ww + li) * 3 + c] * 9;

        out[(j * w + i) * 3 + c] = (gu / 9);

      }

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

  // dim3 t( 32, 32 );
  // dim3 be( 3 * (( cols ) / ((t.x - 2) + 1) ), (( rows ) / ((t.y - 2) + 1) ));
  dim3 t( 16, 16 );
  dim3 be( 3 * 2 * (( cols ) / ((t.x - 2) + 1) ), ( 2 *  rows  / ((t.y - 2) + 1) ));

  // dim3 t( 4, 4 );
  // dim3 be( 3 * 8 * (( cols ) / ((t.x - 2) + 1) ), ( 8 *  rows  / ((t.y - 2) + 1) ));

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );

  // One kernel is launched in each stream.
  embossShared<<< be, t, 3 * t.x * t.y, streams[ 0 ] >>>( rgb_d, out, cols, rows / 2 + 2);
  embossShared<<< be, t, 3 * t.x * t.y, streams[ 1 ] >>>( rgb_d+size/2, out+size/2, cols, rows / 2);

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



  cv::imwrite( "outEmbossShared.jpg", m_out );
  cudaFree( rgb_d);
  //cudaFree( g_d);
  cudaFree ( out);


  return 0;
}
