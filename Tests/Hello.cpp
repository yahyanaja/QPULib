#include "QPULib.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>
#include <chrono>

using namespace std;

// Define function that runs on the GPU.

SharedArray<double> out(6);
std::vector<double> vec = {1, 2, 3, 4};
std::vector<double> main_filter = {5, 6, 7};
static int const main_siz = main_filter.size();
static int const vec_siz = vec.size();


 inline void multi_vec_elem(double elem, int it) {
     for(int i = 0; i < main_siz ; i++){
        out[it] += main_filter[i] * elem;
        it++;
    }

}

void conv_p() {
    printf("DP: Conv started!\n");
    auto t = std::chrono::system_clock::now();
    // int const ng = g.size();
    // int const n  = nf + ng - 1;
    // std::vector<T> out(n, T());
    // int out_beg = 0; // out.begin();
    for(auto i(0); i < vec_siz; ++i) {
        multi_vec_elem(main_filter, vec[i], i );

    }
    std::chrono::duration<double> dif_loc = std::chrono::system_clock::now() - t;
    printf("DP: Conv ended. Took: %fs\n", dif_loc.count());
    }


// void hello(Ptr<Int> p)
// {
//   *p = 1;
// }
// SharedArray<int> vec(4);
// SharedArray<int> main_filter(3);

int main()
{
  for (int i = 0; i < 6; i++)
    out[i] = 0;

  // Construct kernel
  auto k = compile(conv_p);

  // Allocate and initialise array shared between ARM and GPU
  // SharedArray<int> array(16);

  //   int value = 5;
  // for( int j = 0; j < 3; j++)
  //   main_filter[j] = value++;

  // Invoke the kernel and display the result
  k();

  for (int i = 0; i < 6; i++) {
    printf("%i: %f\n", i, out[i]);
  }

  return 0;
}
