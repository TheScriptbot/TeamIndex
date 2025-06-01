#pragma once

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;


namespace TeamIndex {

class TableQuantizer
{
private:
    static inline uint8_t find_bin(double value, const double* quantiles, size_t num_quantiles) {
        size_t low = 0, high = num_quantiles;
        while (low < high) {
            size_t mid = (low + high) / 2;
            if (value < quantiles[mid])
                high = mid;
            else
                low = mid + 1;
        }
        return static_cast<uint8_t>(low);  // Bin index ∈ [0, cell_count - 1]
    }
public:
    ~TableQuantizer() = default;
    TableQuantizer() = default;
    using TableType = double; // input type of the quantization - all attributes of the input table have to have this type
    using QuantizedType = uint8_t; // result type of the quantization

    static py::array_t<QuantizedType> quantize_table(
        py::array_t<TableType, py::array::f_style | py::array::forcecast> table,
        py::list quantile_list,
        std::vector<size_t> cell_counts);

};

}

