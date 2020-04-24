#include <opencv2/opencv.hpp>
#include <vector>

__global__ void sobelShared ( unsigned char * data,   unsigned char * out, std::size_t w, std::size_t h) {

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

        auto hh = sh[ ((lj-1)*ww + li - 1)* 3 + c ] - sh[ ((lj-1)*ww + li + 1) * 3 + c ]
        + 2 * sh[ (lj*ww + li - 1) * 3 + c ] - 2* sh[ (lj*ww+li+1) * 3 + c]
        + sh[ ((lj+1)*ww + li -1) * 3 + c] - sh[ ((lj+1)*ww +li + 1) * 3 + c];
        auto vv = sh[ ((lj-1)*ww + li - 1) * 3 + c ] - sh[ ((lj+1)*ww + li - 1) * 3 + c ]
        + 2 * sh[ ((lj-1)*ww + li) * 3 + c ] - 2* sh[ ((lj+1)*ww+li) * 3 + c ]
        + sh[ ((lj-1)*ww + li +1) * 3 + c] - sh[ ((lj+1)*ww +li + 1) * 3 + c];

        auto res = hh * hh + vv * vv;
        res = res > 255*255 ? res = 255*255 : res;
        out[ (j * w + i) * 3 + c ] = sqrt( (float)res );

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
  cudaStream_t streams[ 4 ];

  // Creation.
  cudaStreamCreate( &streams[ 0 ] );
  cudaStreamCreate( &streams[ 1 ] );
  cudaStreamCreate( &streams[ 2 ] );
  cudaStreamCreate( &streams[ 3 ] );


  cudaMemcpyAsync( rgb_d, rgb, size/4, cudaMemcpyHostToDevice, streams[ 0 ] );
  cudaMemcpyAsync( rgb_d+size/4, rgb+size/4, size/4, cudaMemcpyHostToDevice, streams[ 1 ] );
  cudaMemcpyAsync( rgb_d+size/2, rgb+size/2, size/4, cudaMemcpyHostToDevice, streams[ 1 ] );
  cudaMemcpyAsync( rgb_d+3*size/4, rgb+3*size/4, size/4, cudaMemcpyHostToDevice, streams[ 1 ] );

  //cudaMemcpyAsync( v1_d+size/2, v1+size/2, size/2 * sizeof(int), cudaMemcpyHostToDevice, streams[ 1 ] );
  dim3 t( 32, 32 );
  dim3 be( 3 * (( cols / 2) / (t.x + 1) ), (( rows / 2) / (t.y + 1) ));
  dim3 bu( 3 * (( cols - 1) / (t.x-2) + 1) , ( rows - 1 ) / (t.y-2) + 1 );
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
  sobelShared<<< bu, t, 3 * t.x * t.y, streams[ 0 ] >>>( rgb_d, out, cols, rows/4 + 2);
  sobelShared<<< bu, t, 3 * t.x * t.y, streams[ 1 ] >>>( rgb_d+size/4, out+size/4, cols, rows/4 + 4);
  sobelShared<<< bu, t, 3 * t.x * t.y, streams[ 2 ] >>>( rgb_d+size/2, out+size/2, cols, rows/4 + 2);
  sobelShared<<< bu, t, 3 * t.x * t.y, streams[ 3 ] >>>( rgb_d+3*size/4, out+3*size/4, cols, rows/4 );

  // Sending back the resulting vector by halves.
  cudaMemcpyAsync( g.data(), out, size/4, cudaMemcpyDeviceToHost, streams[ 0 ] );
  cudaMemcpyAsync( g.data()+size/4, out+size/4, size/4, cudaMemcpyDeviceToHost, streams[ 1 ] );
  cudaMemcpyAsync( g.data()+size/2, out+size/2, size/4, cudaMemcpyDeviceToHost, streams[ 2 ] );
  cudaMemcpyAsync( g.data()+3*size/4, out+3*size/4, size/4, cudaMemcpyDeviceToHost, streams[ 3 ] );

  // Synchronize everything.
  cudaDeviceSynchronize();

  // Destroy streams.
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



  cv::imwrite( "outSobelShared.jpg", m_out );
  cudaFree( rgb_d);
  //cudaFree( g_d);
  cudaFree ( out);


  return 0;
}
