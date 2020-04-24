#include <opencv2/opencv.hpp>
#include <vector>

__global__ void blurGaussShared ( unsigned char * data,   unsigned char * out, std::size_t w, std::size_t h) {

  auto i = blockIdx.x * (blockDim.x-4) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-4) + threadIdx.y;

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

    if( li > 1 && li < (blockDim.x - 2) && lj > 1 && lj < (blockDim.y - 2) )
    {
      for (auto c = 0; c < 3; ++c){

          /*
            1 : 1 - 4 - 7 - 4 - 1
            2 : 4 - 16 - 26 - 16 - 4
            3 : 7 - 26 - 41 - 26 - 7
            4 : 4 - 16 - 26 - 16 - 4
            5 : 1 - 4 - 7 - 4 - 1

          */

          auto gu = sh[(( lj - 2) * ww + li - 2) * 3 + c] + sh[(( lj - 2) * ww + li + 2) * 3 + c]
          +     4 * sh[(( lj - 2) * ww + li - 1) * 3 + c] + 4 * sh[(( lj - 2) * ww + li + 1) * 3 + c]
          +     7 *sh[(( lj - 2) * ww + li) * 3 + c]
          +     4 *sh[(( lj - 1) * ww + li - 2) * 3 + c] + 4 * sh[(( lj - 1) * ww + li + 2) * 3 + c]
          +     16 * sh[(( lj - 1) * ww + li - 1) * 3 + c] + 16 * sh[(( lj - 1) * ww + li + 1) * 3 + c]
          +     26 * sh[(( lj - 1) * ww + li) * 3 + c]
          +     7 * sh[(( lj ) * ww + li - 1) * 3 + c] + 7 * sh[(( lj ) * ww + li + 1) * 3 + c]
          +     26 * sh[(( lj ) * ww + li - 2) * 3 + c] + 26 * sh[(( lj ) * ww + li + 2) * 3 + c]
          +     41 * sh[(( lj ) * ww + li) * 3 + c]
          +     4 * sh[(( lj + 1) * ww + li - 1) * 3 + c] + 4 * sh[(( lj + 1) * ww + li + 1) * 3 + c]
          +     16 * sh[(( lj + 1) * ww + li - 2) * 3 + c] + 16 * sh[(( lj + 1) * ww + li + 2) * 3 + c]
          +     26 * sh[(( lj + 1) * ww + li) * 3 + c]
          +     sh[(( lj + 2) * ww + li - 1) * 3 + c] + sh[(( lj + 2) * ww + li + 1) * 3 + c]
          +     4 * sh[(( lj + 2) * ww + li - 2) * 3 + c] + 4 * sh[(( lj + 2) * ww + li + 2) * 3 + c]
          +     7 * sh[(( lj + 2) * ww + li) * 3 + c];

          out[(j * w + i) * 3 + c] = (gu / 273);

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
  dim3 t( 32, 32 );
  dim3 be( 3 * (( cols - 1) / t.x + 1 ), (( rows - 1 ) / t.y + 1 ));
  dim3 bu( 3 * (( cols - 1) / (t.x-4) + 1) , ( rows - 1 ) / (t.y-4) + 1 );

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  cudaEventRecord( start );

  blurGaussShared<<< bu, t, 3*t.x*t.y >>>( rgb_d, out, cols, rows );
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

  cv::imwrite( "outBlurGaussShared.jpg", m_out );
  cudaFree( rgb_d);
  cudaFree ( out);


  return 0;
}
