#include <opencv2/opencv.hpp>
#include <vector>

__global__ void blurSobelShared ( unsigned char * data,   unsigned char * out, std::size_t w, std::size_t h) {

  auto i = blockIdx.x * (blockDim.x-4) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-4) + threadIdx.y;

  auto li = threadIdx.x;
  auto lj = threadIdx.y;


  extern __shared__ unsigned char sh[];

  if( i < w && j < h ) {

    // on s'occupe du rouge
    //  sh[3 * (lj * blockDim.x + li) ] = data[ 3 * ( j * w + i ) ];
    //  sh[3 * (lj * blockDim.x + li) + 1 ] = data[ 3 * ( j * w + i ) + 1];
    //  sh[3 * (lj * blockDim.x + li) + 2 ] = data[ 3 * ( j * w + i ) + 2 ];


    auto ww = blockDim.x;

    if( li > 0 && li < (blockDim.x - 1) && lj > 0 && lj < (blockDim.y - 1) )
    {
      for (auto c = 0; c < 3; ++c){

        auto gu =     data[((j - 1) * w + i - 1) * 3 + c] +     data[((j - 1) * w + i + 1) * 3 + c]
        +     data[( j      * w + i - 1) * 3 + c] +     data[( j      * w + i + 1) * 3 + c]
        +     data[((j + 1) * w + i - 1) * 3 + c] +     data[((j + 1) * w + i + 1) * 3 + c]
        +     data[(( j - 1) * w + i) * 3 + c]     +     data[( j      * w + i) * 3 + c]
        +     data[(( j + 1) * w + i) * 3 + c];


        sh[(lj * blockDim.x + li) * 3 + c] = (gu / 9);
        //out[ (j * w + i) * 3 + c ] = (gu / 9);

      }
    }
    __syncthreads();

    if( li > 1 && li < (blockDim.x - 2) && lj > 1 && lj < (blockDim.y - 2) )
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

  cudaMalloc( &rgb_d, 3 * rows * cols);
  cudaMalloc( &out, 3 * rows * cols );

  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );
  // dim3 t( 32, 32 );
  // dim3 bu( 3 * (( cols - 1) / (t.x-4) + 1) , ( rows - 1 ) / (t.y-4) + 1 );

  dim3 t( 16, 16 );
  dim3 bu( 3 * 2 *( cols - 1) / (t.x-4 + 1) , (2 * rows - 1 ) / (t.y-4 + 1 ));

  // dim3 t( 4, 4 );
  // dim3 bu( 3 * 8 *( cols - 1) / (t.x-4 + 1) , (8 * rows - 1 ) / (t.y-4 + 1 ));

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  cudaEventRecord( start );

  blurSobelShared<<< bu, t, 3*t.x*t.y >>>( rgb_d, out, cols, rows );
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

  cv::imwrite( "outBlurSobelShared.jpg", m_out );
  cudaFree( rgb_d);
  cudaFree ( out);


  return 0;
}
