size_t avx512_utf8_length_mkl2(const uint8_t *str, size_t len) {
  size_t answer = len / sizeof(__m512i) * sizeof(__m512i);
  size_t i = 0;
  __m512i eight_64bits = _mm512_setzero_si512();
  __m512i v_01 = _mm512_set1_epi8(0x01);
  while (i + sizeof(__m512i) <= len) {
    __m512i runner = _mm512_setzero_si512();
    size_t iterations = (len - i) / sizeof(__m512i);
    if (iterations > 255) {
      iterations = 255;
    }
    size_t max_i = i + iterations * sizeof(__m512i) - sizeof(__m512i);
    // for (; i + 8*sizeof(__m512i) <= max_i; i += 8*sizeof(__m512i)) {
    for (; i + 4*sizeof(__m512i) <= max_i; i += 4*sizeof(__m512i)) {
            // Load four __m512i vectors
            __m512i input1 = _mm512_loadu_si512((const __m512i *)(str + i));
            __m512i input2 = _mm512_loadu_si512((const __m512i *)(str + i + sizeof(__m512i)));
            __m512i input3 = _mm512_loadu_si512((const __m512i *)(str + i + 2*sizeof(__m512i)));
            __m512i input4 = _mm512_loadu_si512((const __m512i *)(str + i + 3*sizeof(__m512i)));

/*             // Generate four masks
            __mmask64 mask1 = _mm512_cmpgt_epi8_mask(_mm512_setzero_si512(), input1);
            __mmask64 mask2 = _mm512_cmpgt_epi8_mask(_mm512_setzero_si512(), input2);
            __mmask64 mask3 = _mm512_cmpgt_epi8_mask(_mm512_setzero_si512(), input3);
            __mmask64 mask4 = _mm512_cmpgt_epi8_mask(_mm512_setzero_si512(), input4);
            // Apply the masks and subtract from the runner
            __m512i not_ascii1 = _mm512_mask_set1_epi8(_mm512_setzero_si512(), mask1, 0xFF);
            __m512i not_ascii2 = _mm512_mask_set1_epi8(_mm512_setzero_si512(), mask2, 0xFF);
            __m512i not_ascii3 = _mm512_mask_set1_epi8(_mm512_setzero_si512(), mask3, 0xFF);
            __m512i not_ascii4 = _mm512_mask_set1_epi8(_mm512_setzero_si512(), mask4, 0xFF);

            runner = _mm512_sub_epi8(runner, not_ascii1);
            runner = _mm512_sub_epi8(runner, not_ascii2);
            runner = _mm512_sub_epi8(runner, not_ascii3);
            runner = _mm512_sub_epi8(runner, not_ascii4); */

            // Shift each input right by 7 to isolate the leading bit
__m512i masked_input1 = _mm512_srli_epi16(input1, 7);
__m512i masked_input2 = _mm512_srli_epi16(input2, 7);
__m512i masked_input3 = _mm512_srli_epi16(input3, 7);
__m512i masked_input4 = _mm512_srli_epi16(input4, 7);

// And each shifted input with 0x01 to keep only the leading bit
__m512i lead_bit1 = _mm512_and_si512(masked_input1, v_01);
__m512i lead_bit2 = _mm512_and_si512(masked_input2, v_01);
__m512i lead_bit3 = _mm512_and_si512(masked_input3, v_01);
__m512i lead_bit4 = _mm512_and_si512(masked_input4, v_01);

// Add the leading bits to the runner
runner = _mm512_add_epi8(runner, lead_bit1);
runner = _mm512_add_epi8(runner, lead_bit2);
runner = _mm512_add_epi8(runner, lead_bit3);
runner = _mm512_add_epi8(runner, lead_bit4);


    }

    for (; i <= max_i; i += sizeof(__m512i)) {
      __m512i input = _mm512_loadu_si512((const __m512i *)(str + i));

      __mmask64 mask = _mm512_cmpgt_epi8_mask(_mm512_setzero_si512(), input);
      __m512i not_ascii = _mm512_mask_set1_epi8(_mm512_setzero_si512(), mask, 0xFF);
      runner = _mm512_sub_epi8(runner, not_ascii);
    }

    eight_64bits = _mm512_add_epi64(eight_64bits, _mm512_sad_epu8(runner, _mm512_setzero_si512()));
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