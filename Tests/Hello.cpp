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

 inline void multi_vec_elem(std::vector<double> const &vec, double elem, int it) {
     static int const vec_siz = vec.size();
     for(int i = 0; i < vec_siz ; i++){
        out[it] += vec[i] * elem;
        it++;
    }

}

void conv_p() {
    printf("DP: Conv started!\n");
    auto t = std::chrono::system_clock::now();
    int const nf = vec.size();
    // int const ng = g.size();
    // int const n  = nf + ng - 1;
    // std::vector<T> out(n, T());
    // int out_beg = 0; // out.begin();
    for(auto i(0); i < nf; ++i) {
        multi_vec_elem(main_filter, vec[i], i );

    }
    std::chrono::duration<double> dif_loc = std::chrono::system_clock::now() - t;
    printf("DP: Conv ended. Took: %fs\n", dif_loc.count());
    }


void hello(Ptr<Int> p)
{
  *p = 1;
}
// SharedArray<int> vec(4);
// SharedArray<int> main_filter(3);

int main()
{
  // Construct kernel
  auto k = compile(conv_p);

  // Allocate and initialise array shared between ARM and GPU
  // SharedArray<int> array(16);
  for (int i = 0; i < 6; i++)
    out[i] = 0;

  //   int value = 5;
  // for( int j = 0; j < 3; j++)
  //   main_filter[j] = value++;

  // Invoke the kernel and display the result
  k();

  for (int i = 0; i < 6; i++) {
    printf("%i: %i\n", i, out[i]);
  }

  return 0;
}
