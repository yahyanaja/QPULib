#include "QPULib.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>
#include <chrono>

using namespace std;

// Define function that runs on the GPU.

// out 1132 vec 5000 main 633
std::vector<float> vec = {1, 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15};
std::vector<float> main_filter = {5, 6, 7, 8, 9, 10};
static const int out_siz = vec.size() + main_filter.size() - 1;
SharedArray<float> out(out_siz);

static int const main_siz = main_filter.size();
static int const vec_siz = vec.size();


 inline void multi_vec_elem(float elem, int it) {
     for(int i = 0; i < main_siz ; i++){
        out[it] += (float) main_filter[i] * elem;
        it++;
    }

}

void conv_p() {
    printf("DP: Conv started!\n");
    auto start = std::chrono::high_resolution_clock::now();
    // int const ng = g.size();
    // int const n  = nf + ng - 1;
    // std::vector<T> out(n, T());
    // int out_beg = 0; // out.begin();
    for(auto i(0); i < vec_siz; ++i) {
        multi_vec_elem(vec[i], i );

    }
    auto finish = std::chrono::high_resolution_clock::now();

    printf("DP: Conv ended. Took: ");
    printf("%lld", std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count());
    printf("ns\n");

    }


// void hello(Ptr<Int> p)
// {
//   *p = 1;
// }
// SharedArray<int> vec(4);
// SharedArray<int> main_filter(3);

int main()
{
  for (int i = 0; i < out_siz; i++)
    out[i] = 0.0;

  // Construct kernel
  auto k = compile(conv_p);

  // Allocate and initialise array shared between ARM and GPU
  // SharedArray<int> array(16);

  //   int value = 5;
  // for( int j = 0; j < 3; j++)
  //   main_filter[j] = value++;

  // Invoke the kernel and display the result
  k();

  for (int i = 0; i < out_siz; i++) {
    printf("%i: %f\n", i, out[i]);
  }

  return 0;
}
