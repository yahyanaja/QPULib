#include "QPULib.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>
#include <chrono>
#include <math.h>       /* ceil */

using namespace std;

// Define function that runs on the GPU.

// out 1132 vec 5000 main 633
std::vector<double> vec_v =          {378.5000000000000000, 376.9233274231110045, 375.2955236053464887, 371.1398649746128058, 367.3476591388542261, 361.0878024687539778, 355.5915564636507611, 348.5518813468847839, 340.3937603406353105, 330.0817273248573542, 319.6766768800011391, 308.8258974432939112, 298.1188003414918057, 284.7249569551154309, 271.9632436777842486, 258.2735846954043382, 244.0965983155249432, 229.2797459089320000, 214.1501418971103874,
                                      197.8764081924703362, 182.2882756518549172, 166.1476595700851533, 149.7151558524639654, 133.2411582905070588, 117.0100197815578440, 100.5541299696064357, 84.1213857784912875, 67.4818630881904937, 51.1493884537923975, 35.1096213341426022, 51.1493884537923975, 35.1096213341426022 };
                                      // 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      // 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

                                      // 378.5000000000000000, 376.9233274231110045, 375.2955236053464887, 371.1398649746128058, 367.3476591388542261, 361.0878024687539778, 355.5915564636507611, 348.5518813468847839, 340.3937603406353105, 330.0817273248573542, 319.6766768800011391, 308.8258974432939112, 298.1188003414918057, 284.7249569551154309, 271.9632436777842486, 258.2735846954043382, 244.0965983155249432, 229.2797459089320000, 214.1501418971103874,
                                      //                                       197.8764081924703362, 182.2882756518549172, 166.1476595700851533, 149.7151558524639654, 133.2411582905070588, 117.0100197815578440, 100.5541299696064357, 84.1213857784912875, 67.4818630881904937, 51.1493884537923975, 35.1096213341426022, 19.3959914826039217, 3.8401817883525413, 19.3959914826039217, 3.8401817883525413};
                                     std::vector<double> main_filter_v = {-1.3297364530947558e-04, -1.8619813326340053e-05, -1.9927583273486273e-05, -2.1230104229368908e-05, -2.2634103062365175e-05, -2.4041162317835777e-05, -2.5557767918874349e-05, -2.7080406648550287e-05, -2.8710722180127414e-05, -3.0337532931360952e-05, -3.2062154932255169e-05, -3.3774152468525441e-05, -3.5591572552427255e-05, -3.7415817130812727e-05, -3.9382927173854860e-05, -4.1369340531321311e-05 };// ,
                                                                          // -1.3297364530947558e-04, -1.8619813326340053e-05, -1.9927583273486273e-05, -2.1230104229368908e-05, -2.2634103062365175e-05, -2.4041162317835777e-05, -2.5557767918874349e-05, -2.7080406648550287e-05, -2.8710722180127414e-05, -3.0337532931360952e-05, -3.2062154932255169e-05, -3.3774152468525441e-05, -3.5591572552427255e-05, -3.7415817130812727e-05, -3.9382927173854860e-05, -4.1369340531321311e-05 ,
                                                                          // -1.3297364530947558e-04, -1.8619813326340053e-05, -1.9927583273486273e-05, -2.1230104229368908e-05, -2.2634103062365175e-05, -2.4041162317835777e-05, -2.5557767918874349e-05, -2.7080406648550287e-05, -2.8710722180127414e-05, -3.0337532931360952e-05, -3.2062154932255169e-05, -3.3774152468525441e-05, -3.5591572552427255e-05, -3.7415817130812727e-05, -3.9382927173854860e-05, -4.1369340531321311e-05 ,
                                                                          // -1.3297364530947558e-04, -1.8619813326340053e-05, -1.9927583273486273e-05, -2.1230104229368908e-05, -2.2634103062365175e-05, -2.4041162317835777e-05, -2.5557767918874349e-05, -2.7080406648550287e-05, -2.8710722180127414e-05, -3.0337532931360952e-05, -3.2062154932255169e-05, -3.3774152468525441e-05, -3.5591572552427255e-05, -3.7415817130812727e-05, -3.9382927173854860e-05, -4.1369340531321311e-05 ,
                                                                          // 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                                          // 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
   // ceil(3750/16)*16
static int const main_siz = main_filter_v.size();
static int const vec_siz = vec_v.size() * 16;
static int const out_siz = vec_siz/16 + main_siz ;// - 1 ; // omitted -1 inorder to be multiple of 16

// std::vector<double>         out_v (out_siz);


// void hello(Ptr<float> p)
// {
//   p = p + (me() << 4);
//   *p = me();
// }

// inline Float multi_vec_elem(Float m_ptr, Float prev_out, Float v_ptr) {
// // Float elem_Float(elem);
// // Int it_Int = it;
//
// // o_ptr = o_ptr + it_Int;
// // gather(o_ptr + it_Int + index());
// // gather(m_ptr+index());
// //
// // Float o_ptr_Float;
// // Float m_ptr_Float;
// //
// // receive(o_ptr_Float);
// // receive(m_ptr_Float);
//
//     // For(Int i = 0, i < main_siz , i = i + 16)
//
//        return  prev_out + m_ptr * v_ptr;
//
//        // m_ptr = m_ptr + 16;
//        // o_ptr = o_ptr + 16;
//    // End
//
// }


void conv(Int m_ptr_siz, Int o_ptr_siz, Int vec_ptr_siz, Ptr<Float> m_ptr, Ptr<Float> o_ptr, Ptr<Float> vec_ptr)
{
  Int inc = 16;
  Ptr<Float> m = m_ptr ; // + index();
  Ptr<Float> o = o_ptr ; // + index();
  Ptr<Float> v = vec_ptr; //  + index();
  gather(m);

  // SharedArray<float> out(o_ptr_siz);
  Float mOld, oOld, vOld;
  receive(mOld);
  For (Int i = 0, i < vec_ptr_siz, i = i+inc)

    store(*o + *m * *(v+i), o);
                  o = o+1; // m = m+inc;
  End
}

// void conv(Int m_ptr_siz, Int o_ptr_siz, Int vec_ptr_siz, Ptr<Float> m_ptr, Ptr<Float> o_ptr, Ptr<Float> vec_ptr)
// {
//   Int inc = 1;
//   Ptr<Float> m = m_ptr ; // + index();
//   Ptr<Float> o = o_ptr ; // + index();
//   Ptr<Float> v = vec_ptr; //  + index();
//   // SharedArray<float> out(o_ptr_siz);
//
//   For (Int i = 0, i < vec_ptr_siz, i = i+inc)
//     store(*o + *m /* * v[0] */, o);
//                   o = o+inc; v = v+inc;
//   End
//
// }


// void conv_p(Ptr<Float> m_ptr, Ptr<Float> o_ptr, Ptr<Float> vec_ptr) {
//     printf("DP: Conv started!\n");
//     auto start = std::chrono::high_resolution_clock::now();
//     // float section =  (float) vec_siz / 1; // numQPUs().expr->intLit
//     int i_at_start = 0;       // (int) (section * (float)  0    /* me().expr->intLit */) ;
//     int i_at_end =   vec_siz; // (int) (section * (float) (1 + 0/* me().expr->intLit */ ));
//     // printf("QPU (%d/%d), section: %f, i_start: %d, i_end: %d\n", me().expr->intLit,
//                             // numQPUs().expr->intLit, section, i_at_start, i_at_end);
//     for(int i = i_at_start; i < i_at_end; i++) {
//       // if( i >= out_siz)
//       //   printf("i >= out_siz ( %d >= %d )\n", i, out_siz);
//       //   if( i >= vec_siz)
//       //   printf("i >= vec_siz ( %d >= %d )\n", i, vec_siz);
//       Ptr<Float> m_ptr_loc = m_ptr;
//       Ptr<Float> o_ptr_loc = o_ptr + i;
//       const int inc = 16;
//       Float c(vec_ptr[i]);
//       for( int j = 0; j < main_siz; j += inc ){
//
//         Float b, c(vec_ptr[i]);
//       //
//       // gather(m_ptr_loc);
//       // receive(a);
//       //
//       gather(o_ptr_loc);
//       receive(b);
//       Float res = b + *m_ptr_loc * vec_ptr[i];
//             // store(1.0, o_ptr_loc);
//             if( j + inc < main_siz )
//             {
//               m_ptr_loc = m_ptr_loc + inc;
//               o_ptr_loc = o_ptr_loc + inc;
//             }
//           }
//
//     }
//
//     auto finish = std::chrono::high_resolution_clock::now();
//
//     printf("DP: Conv ended. Took: ");
//     printf("%lld", std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count());
//     printf("ns\n");
//
//     }

int main()
{
//   if(out_siz % 16 != 0 || main_siz % 16 != 0 || out_siz % 16 != 0)
// {
//       printf("Error - sizes expected to be mutliple of 16.\n");
// }
      printf("vec_siz: %d, main_siz: %d, out_siz: %d\n", vec_siz, main_siz, out_siz);

      SharedArray<float>          out(out_siz);
      SharedArray<float>          vec(vec_siz);
      SharedArray<float>  main_filter(main_siz);

  for(int i = 0; i < main_siz; i++){
      main_filter[i] = main_filter_v[i];
  }
  int ind = 0;
  for(int i = 0; i < vec_siz; i+=16){
    for(int j = 0; j < 16; j++)
      vec[i + j] = vec_v[ind];
      ind++;
  }
  for (int i = 0; i < out_siz; i++){
    out[i] = 0.0;
  }

  // Construct kernel
  auto k = compile(conv);
  const int NQPUS  = 1;
  k.setNumQPUs(NQPUS);
  // if(numQPUs().expr->intLit != NQPUS )
  // {
  //   printf("Expected numQPUs() to be eq to NQPUS = %d\n", NQPUS);
  //   return 1;
  // }
  // else
  //   printf("Equal: numQPUs().expr->intLit: %d == NQPUS: %d\n", numQPUs().expr->intLit, NQPUS);


  // Allocate and initialise array shared between ARM and GPU
  // SharedArray<int> array(16);

  //   int value = 5;
  // for( int j = 0; j < 3; j++)
  //   main_filter[j] = value++;

  // void conv(Int m_ptr_siz, Int o_ptr_siz, Int vec_ptr_siz, Ptr<Float> m_ptr, Ptr<Float> o_ptr, Ptr<Float> vec_ptr)
  // Invoke the kernel and display the result
  k(main_siz, out_siz, vec_siz, &main_filter, &out, &vec);

  for (int i = 0; i < out_siz; i++) {
    printf("%i: %f\n", i, out[i]);
  }

  return 0;
}
