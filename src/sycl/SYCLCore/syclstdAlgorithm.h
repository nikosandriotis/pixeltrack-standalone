#ifndef HeterogeneousCore_SYCLUtilities_syclstdAlgorithm_h
#define HeterogeneousCore_SYCLUtilities_syclstdAlgorithm_h

#include <CL/sycl.hpp>
#include <utility>

// reimplementation of std algorithms able to compile with SYCL and run on GPUs,
// mostly by declaringthem constexpr

namespace sycl_std {

  template <typename T = void>
  struct less {
    constexpr bool operator()(const T &lhs, const T &rhs) const { return lhs < rhs; }
  };

  template <>
  struct less<void> {
    template <typename T, typename U>
    constexpr bool operator()(const T &lhs, const U &rhs) const {
      return lhs < rhs;
    }
  };

  template <typename RandomIt, typename T, typename Compare = less<T>>
  constexpr RandomIt lower_bound(RandomIt first, RandomIt last, const T &value, Compare comp = {}) {
    auto count = last - first;

    while (count > 0) {
      auto it = first;
      auto step = count / 2;
      it += step;
      if (comp(*it, value)) {
        first = ++it;
        count -= step + 1;
      } else {
        count = step;
      }
    }
    return first;
  }

  template <typename RandomIt, typename T, typename Compare = less<T>>
  constexpr RandomIt upper_bound(RandomIt first, RandomIt last, const T &value, Compare comp = {}) {
    auto count = last - first;

    while (count > 0) {
      auto it = first;
      auto step = count / 2;
      it += step;
      if (!comp(value, *it)) {
        first = ++it;
        count -= step + 1;
      } else {
        count = step;
      }
    }
    return first;
  }

  template <typename RandomIt, typename T, typename Compare = sycl_std::less<T>>
  constexpr RandomIt binary_find(RandomIt first, RandomIt last, const T &value, Compare comp = {}) {
    first = sycl_std::lower_bound(first, last, value, comp);
    return first != last && !comp(value, *first) ? first : last;
  }

}  // namespace sycl_std

#endif  // HeterogeneousCore_SYCLUtilities_syclstdAlgorithm_h