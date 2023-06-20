#include "linux-perf-events.h"
#include <x86intrin.h>
#include <iostream>
#include <random>
#include <vector>
#include <immintrin.h>


 __attribute__((optimize("no-tree-vectorize")))
size_t pure_scalar_utf8_length(const uint8_t *c, size_t len) {
  size_t answer = 0;
  for (size_t i = 0; i < len; i++) {
    if ((c[i] >> 7)) {
      answer++;
    }
  }
  return answer + len;
}

size_t scalar_utf8_length(const uint8_t *c, size_t len) {
  size_t answer = 0;
  for (size_t i = 0; i < len; i++) {
    if ((c[i] >> 7)) {
      answer++;
    }
  }
  return answer + len;
}

size_t avx2_utf8_length_basic(const uint8_t *str, size_t len) {
  size_t answer = len / sizeof(__m256i) * sizeof(__m256i);
  size_t i;
  for (i = 0; i + sizeof(__m256i) <= len; i += 32) {
    __m256i input = _mm256_loadu_si256((const __m256i *)(str + i));
    answer += __builtin_popcount(_mm256_movemask_epi8(input));
  }
  return answer + scalar_utf8_length(str + i, len - i);
}

/**
 * Similar to :
 * Wojciech Muła, Nathan Kurz, Daniel Lemire
 * Faster Population Counts Using AVX2 Instructions
 * Computer Journal 61 (1), 2018
 **/
size_t avx2_utf8_length_mkl(const uint8_t *str, size_t len) {
  size_t answer = len / sizeof(__m256i) * sizeof(__m256i); //gets the number of m256 that will fit into the length
  size_t i = 0; // i is the number of bytes we've done so far
  __m256i four_64bits = _mm256_setzero_si256(); //reserve memory for our 4 x 64 bits
  while (i + sizeof(__m256i) <= len) { //check if we have enough memory left to load 256 bits. 
    //if we do:
    __m256i runner = _mm256_setzero_si256(); //reserve memory to store intermediate values
    
    // the current (remaining?) number of iterations as measured in m256i
    size_t iterations = (len - i) / sizeof(__m256i); // remember len is the number of utf8 since str is a utf8 string
    if (iterations > 255) {// We can do up to 255 loops without overflow. That is because the runner is divided into 8 bits elements,
                           // which can hold a max of 255
      iterations = 255;
    }
    // the max number of iterations is: 
    size_t max_i = i + //i = number of iterations done or how far along are we in progress... probably a safety valve?
                    iterations * sizeof(__m256i)  // how many bytes are remaining
                    - sizeof(__m256i); // minus the current m256i that we are processing


    // Count how many 8-bit elements have ones. Go as far as you can in 256 bits increments
    for (; i <= max_i; i += sizeof(__m256i)) { //advance by 256 bytes each time.
      __m256i input = _mm256_loadu_si256((const __m256i *)(str + i)); 
        runner = _mm256_sub_epi8( // again we subdivide in 8-bit elements, add the result of the inner operation to the runner element-wise... 
                                  // It is to find out whether or not each element have ones. 
                                  runner, 
                                  _mm256_cmpgt_epi8( // take a zero vector and the 256 bit input, divide them into 8 bits, and compare the 8-bits elements with eachother 
                                                     // If the input byte is positive, return 0xFF (-1 in a signed byte context/using two's complement)
                                                     // otherwise it's set to 0. 
                                                    _mm256_setzero_si256(), 
                                                    input));
    }

    // once we processed as many m256i as we could:
    // we divide things by 64 bits, then add the 4 counts gotten by SAD in the runner together to the running count. 
    four_64bits = _mm256_add_epi64( 
                                    four_64bits, 
                                    //we are trying here to sum up the 4x numbers in the runner. 
                                    _mm256_sad_epu8( // subtracts absolute difference between the runner and zero vector by 8-bit elements
                                                     // then divides those elements into blocks of 4, and sums them. 
                                                     // Those 4 x 8 differences are stored at specific locations in the destination vector. Remaining bits in the destination vector are set to zero.
                                                    runner,
                                                    _mm256_setzero_si256()));
  }

  //we then add the results in the separate 4 x 64 bits together
  answer += _mm256_extract_epi64(four_64bits, 0) +
            _mm256_extract_epi64(four_64bits, 1) +
            _mm256_extract_epi64(four_64bits, 2) +
            _mm256_extract_epi64(four_64bits, 3);
  return answer + scalar_utf8_length(str + i, len - i); //and use scalar to return the rest
}

size_t avx512_utf8_length_mkl(const uint8_t *str, size_t len) {
  size_t answer = len / sizeof(__m512i) * sizeof(__m512i);
  // __m512i negative_ones = _mm512_set1_epi8(-1);
  size_t i = 0;
  __m512i eight_64bits = _mm512_setzero_si512();
  while (i + sizeof(__m512i) <= len) {
    __m512i runner = _mm512_setzero_si512();
    // We can do up to 255 loops without overflow.
    size_t iterations = (len - i) / sizeof(__m512i);
    if (iterations > 255) {
      iterations = 255;
    }
    size_t max_i = i + iterations * sizeof(__m512i) - sizeof(__m512i);
    for (; i <= max_i; i += sizeof(__m512i)) {
      __m512i input = _mm512_loadu_si512((const __m512i *)(str + i));

      __mmask64 mask = _mm512_cmpgt_epi8_mask(//  Pack in 64-bit unit. 
                                              // If the input byte is positive(ASCII), return 1
                                              //  else if non-ASCII return 0.
                                              _mm512_setzero_si512(),
                                              input //input is negative (leading bit is 1 under two compleements) => not an ASCII
                                              );
/*       __m512i blended = _mm512_mask_blend_epi8( // process by chunks of 8 bits, retain only the chunks of input that are masked (not ASCII)
                                                mask, 
                                                _mm512_setzero_si512(),
                                                input); */
     // __m512i not_ascii = _mm512_mask_blend_epi8(mask, _mm512_setzero_si512(), negative_ones);
     __m512i not_ascii = _mm512_mask_set1_epi8(_mm512_setzero_si512(), mask, 0xFF);


      runner = _mm512_sub_epi8(runner, not_ascii);
      // runner = _mm512_sub_epi8(runner, blended); // we add the number of non-ASCII in blended to the runner
    }
    eight_64bits = _mm512_add_epi64(
                                    eight_64bits, 
                                    _mm512_sad_epu8(
                                                    runner,
                                                    _mm512_setzero_si512()));
  }
  __m256i first_half = _mm512_extracti64x4_epi64(eight_64bits, 0);
  __m256i second_half = _mm512_extracti64x4_epi64(eight_64bits, 1);
  answer += (size_t)_mm256_extract_epi64(first_half, 0) +
            (size_t)_mm256_extract_epi64(first_half, 1) +
            (size_t)_mm256_extract_epi64(first_half, 2) +
            (size_t)_mm256_extract_epi64(first_half, 3) +
            (size_t)_mm256_extract_epi64(second_half, 0) +
            (size_t)_mm256_extract_epi64(second_half, 1) +
            (size_t)_mm256_extract_epi64(second_half, 2) +
            (size_t)_mm256_extract_epi64(second_half, 3);
  return answer + scalar_utf8_length(str + i, len - i);
}

size_t avx2_utf8_length_mkl2(const uint8_t *str, size_t len) {
  size_t answer = len / sizeof(__m256i) * sizeof(__m256i); //get how many m256i fits into the len
  size_t i = 0; // how many bytes we've read thus far
  __m256i four_64bits = _mm256_setzero_si256(); 
  while (i + sizeof(__m256i) <= len) { //can we read another 256 bits?
    __m256i runner = _mm256_setzero_si256(); //running count
    // We can do up to 255 loops without overflow.
    size_t iterations = (len - i) / sizeof(__m256i); 
    if (iterations > 255) {
      iterations = 255;
    }
    size_t max_i = i + iterations * sizeof(__m256i) - sizeof(__m256i);

    for (; i + 4*sizeof(__m256i) <= max_i; i += 4*sizeof(__m256i)) { //the difference here is that we process 4 m256i at a time
      __m256i input1 = _mm256_loadu_si256((const __m256i *)(str + i));
      __m256i input2 = _mm256_loadu_si256((const __m256i *)(str + i + sizeof(__m256i)));
      __m256i input3 = _mm256_loadu_si256((const __m256i *)(str + i + 2*sizeof(__m256i)));
      __m256i input4 = _mm256_loadu_si256((const __m256i *)(str + i + 3*sizeof(__m256i)));
      __m256i input12 = _mm256_add_epi8( // add up the presence of non-null bytes in input 1 & 2 per byte element
                                        _mm256_cmpgt_epi8( //check whether each byte is non-null
                                                          _mm256_setzero_si256(), 
                                                          input1),
                                        _mm256_cmpgt_epi8( //do that for input 2 as well
                                                          _mm256_setzero_si256(),
                                                           input2));

      //do the same but this time for input 3&4
      __m256i input23 = _mm256_add_epi8(
                                        _mm256_cmpgt_epi8(
                                                          _mm256_setzero_si256(),
                                                          input3),
                                        _mm256_cmpgt_epi8(
                                                          _mm256_setzero_si256(),
                                                          input4));
      // add them up together
      __m256i input1234 = _mm256_add_epi8(input12, input23);
      runner = _mm256_sub_epi8(runner, input1234); // you have your runner
    }

    // now do the same as last one
    for (; i <= max_i; i += sizeof(__m256i)) {
      __m256i input = _mm256_loadu_si256((const __m256i *)(str + i));
      runner = _mm256_sub_epi8(
          runner, _mm256_cmpgt_epi8(_mm256_setzero_si256(), input));
    }
    four_64bits = _mm256_add_epi64(
        four_64bits, _mm256_sad_epu8(runner, _mm256_setzero_si256()));
  }
  answer += _mm256_extract_epi64(four_64bits, 0) +
            _mm256_extract_epi64(four_64bits, 1) +
            _mm256_extract_epi64(four_64bits, 2) +
            _mm256_extract_epi64(four_64bits, 3);
  return answer + scalar_utf8_length(str + i, len - i);
}

#include <immintrin.h>

size_t avx512_utf8_length_mkl2(const uint8_t *str, size_t len) {
  size_t answer = len / sizeof(__m512i) * sizeof(__m512i);
  size_t i = 0;
  __m512i eight_64bits = _mm512_setzero_si512();
  while (i + sizeof(__m512i) <= len) {
    __m512i runner = _mm512_setzero_si512();
    // We can do up to 511 loops without overflow.
    size_t iterations = (len - i) / sizeof(__m512i);
    if (iterations > 511) {
      iterations = 511;
    }
    //bytes we've read + how many  - current 512 bits block we're reading
    size_t max_i = i + iterations * sizeof(__m512i) - sizeof(__m512i);
    for (; i + 4*sizeof(__m512i) <= max_i; i += 4*sizeof(__m512i)) {
      __m512i input1 = _mm512_loadu_si512((const __m512i *)(str + i));
      __m512i input2 = _mm512_loadu_si512((const __m512i *)(str + i + sizeof(__m512i)));
      __m512i input3 = _mm512_loadu_si512((const __m512i *)(str + i + 2*sizeof(__m512i)));
      __m512i input4 = _mm512_loadu_si512((const __m512i *)(str + i + 3*sizeof(__m512i)));
      __m512i input12 = _mm512_add_epi8(_mm512_movm_epi8(
                                                        _mm512_cmpgt_epi8_mask(
                                                                                input1,
                                                                                _mm512_setzero_si512())),
                                        _mm512_movm_epi8(
                                                          _mm512_cmpgt_epi8_mask(
                                                                                input2, 
                                                                                _mm512_setzero_si512())));
      __m512i input23 = _mm512_add_epi8(_mm512_movm_epi8(_mm512_cmpgt_epi8_mask(input3, _mm512_setzero_si512())),
                                        _mm512_movm_epi8(_mm512_cmpgt_epi8_mask(input4, _mm512_setzero_si512())));
      __m512i input1234 = _mm512_add_epi8(input12, input23);
      runner = _mm512_sub_epi8(runner, input1234);
    }
    for (; i <= max_i; i += sizeof(__m512i)) {
      __m512i input = _mm512_loadu_si512((const __m512i *)(str + i));
      runner = _mm512_sub_epi8(runner, _mm512_movm_epi8(_mm512_cmpgt_epi8_mask(input, _mm512_setzero_si512())));
    }
    eight_64bits = _mm512_add_epi64(eight_64bits, _mm512_sad_epu8(runner, _mm512_setzero_si512()));
  }
  answer += _mm512_reduce_add_epi64(eight_64bits);
  return answer + scalar_utf8_length(str + i, len - i);
}



int main() {
  size_t trials = 3;
  size_t warm_trials = 20;

  size_t N = 8000;
  uint8_t *input = new uint8_t[N];
  for (size_t i = 0; i < N; i++) {
    input[i] = rand();
  }
  size_t expected = scalar_utf8_length(input, N);
  LinuxEvents<PERF_TYPE_HARDWARE> linux_events(
      std::vector<int>{ PERF_COUNT_HW_CPU_CYCLES,
                        PERF_COUNT_HW_INSTRUCTIONS, });
  volatile size_t len{ 0 };

  std::cout << "scalar (no autovec)" << std::endl;
  std::vector<unsigned long long> results(2);
  for (size_t t = 0; t < trials + warm_trials; t++) {
    linux_events.start();
    len = pure_scalar_utf8_length(input, N);
    linux_events.end(results);
    if (t >= warm_trials) {

      std::cout << "cycles/bytes " << double(results[0]) / (len) << " ";
      std::cout << "instructions/bytes " << double(results[1]) / (len) << " ";
      std::cout << "instructions/cycle " << double(results[1]) / results[0]
                << std::endl;
    }
  }
  std::cout << std::endl;

  if(len != expected) { abort(); }

  std::cout << "scalar" << std::endl;
  for (size_t t = 0; t < trials + warm_trials; t++) {
    linux_events.start();
    len = scalar_utf8_length(input, N);
    linux_events.end(results);
    if (t >= warm_trials) {

      std::cout << "cycles/bytes " << double(results[0]) / (len) << " ";
      std::cout << "instructions/bytes " << double(results[1]) / (len) << " ";
      std::cout << "instructions/cycle " << double(results[1]) / results[0]
                << std::endl;
    }
  }
  if(len != expected) { abort(); }

  std::cout << std::endl;

  std::cout << "avx2 (basic)" << std::endl;
  for (size_t t = 0; t < trials + warm_trials; t++) {

    linux_events.start();
    len = avx2_utf8_length_basic(input, N);
    linux_events.end(results);
    if (t >= warm_trials) {
      std::cout << "cycles/bytes " << double(results[0]) / (len) << " ";
      std::cout << "instructions/bytes " << double(results[1]) / (len) << " ";
      std::cout << "instructions/cycle " << double(results[1]) / results[0]
                << std::endl;
    }
  }
  if(len != expected) { abort(); }

  std::cout << std::endl;

  std::cout << "avx2 (mkl)" << std::endl;
  for (size_t t = 0; t < trials + warm_trials; t++) {

    linux_events.start();
    len = avx2_utf8_length_mkl(input, N);
    linux_events.end(results);
    if (t >= warm_trials) {

      std::cout << "cycles/bytes " << double(results[0]) / (len) << " ";
      std::cout << "instructions/bytes " << double(results[1]) / (len) << " ";
      std::cout << "instructions/cycle " << double(results[1]) / results[0]
                << std::endl;
    }
  }
  if(len != expected) { abort(); }

  std::cout << std::endl;

  std::cout << "avx2 (mkl 2)" << std::endl;
  for (size_t t = 0; t < trials + warm_trials; t++) {

    linux_events.start();
    len = avx2_utf8_length_mkl2(input, N);
    linux_events.end(results);
    if (t >= warm_trials) {

      std::cout << "cycles/bytes " << double(results[0]) / (len) << " ";
      std::cout << "instructions/bytes " << double(results[1]) / (len) << " ";
      std::cout << "instructions/cycle " << double(results[1]) / results[0]
                << std::endl;
    }
  }
  if(len != expected) { abort(); }

  #ifdef __AVX512F__

  std::cout << std::endl;
  std::cout << "avx512 (mkl 1)" << std::endl;
  for (size_t t = 0; t < trials + warm_trials; t++) {
    linux_events.start();
    len = avx512_utf8_length_mkl(input, N);
    linux_events.end(results);
    if (t >= warm_trials) {
      std::cout << "cycles/bytes " << double(results[0]) / (len) << " ";
      std::cout << "instructions/bytes " << double(results[1]) / (len) << " ";
      std::cout << "instructions/cycle " << double(results[1]) / results[0]
                << std::endl;
    }
  }
  if(len != expected) { 
    std::cout << "Problem!" << std::endl;
    std::cout << "len: " << len << " Expected: " << expected << std::endl;
    // abort();
     }

  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "avx512 (mkl 2)" << std::endl;
  for (size_t t = 0; t < trials + warm_trials; t++) {
    linux_events.start();
    len = avx512_utf8_length_mkl2(input, N);
    linux_events.end(results);
    if (t >= warm_trials) {
      std::cout << "cycles/bytes " << double(results[0]) / (len) << " ";
      std::cout << "instructions/bytes " << double(results[1]) / (len) << " ";
      std::cout << "instructions/cycle " << double(results[1]) / results[0]
                << std::endl;
    }
  }
  if(len != expected) { 
    std::cout << "Problem!" << std::endl;
    std::cout << "len: " << len << " Expected: " << expected << std::endl;
    // abort(); 
    }

  std::cout << std::endl;
  #endif

  std::cout << std::endl;
  return EXIT_SUCCESS;
}
