#include <opencv2/opencv.hpp>
#include <vector>

__global__ void erodeShared ( unsigned char * data,   unsigned char * out, std::size_t w, std::size_t h) {

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

        auto hu = umax(sh[((lj - 1) * ww + li - 1) * 3 + c], sh[((lj - 1) * ww + li + 1) * 3 + c]);
        hu = umax(hu, sh[(( lj - 1) * ww + li) * 3 + c]);
        auto mu = umax(sh[( lj * ww + li - 1) * 3 + c],  sh[( lj * ww + li + 1) * 3 + c]);
        mu = umax(mu, sh[( lj * ww + li) * 3 + c]);
        auto bu = umax(sh[((lj + 1) * ww + li - 1) * 3 + c], sh[((lj + 1) * ww + li + 1) * 3 + c]);
        bu = umax(bu, sh[((lj + 1) * ww + li) * 3 + c]);

        auto res = umax (hu, mu);
        res = umax (res, bu);

        out[(j * w + i) * 3 + c] = res;
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
  dim3 bu( 3 * (( cols - 1) / (t.x-2) + 1) , ( rows - 1 ) / (t.y-2) + 1 );

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  cudaEventRecord( start );

  erodeShared<<< bu, t, 3*t.x*t.y >>>( rgb_d, out, cols, rows );
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

  cv::imwrite( "outErodeShared.jpg", m_out );
  cudaFree( rgb_d);
  cudaFree ( out);


  return 0;
}
