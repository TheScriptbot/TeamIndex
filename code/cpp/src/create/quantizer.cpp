#include "create/quantizer.hpp"

namespace TeamIndex {

    py::array_t<TableQuantizer::QuantizedType> TableQuantizer::quantize_table(
        py::array_t<TableType, py::array::f_style | py::array::forcecast> table,
        py::list quantile_list,
        std::vector<size_t> cell_counts)
    {
        py::buffer_info info = table.request();
        const TableType* data_ptr = static_cast<TableType*>(info.ptr);
        const size_t n_rows = info.shape[0];
        const size_t n_cols = info.shape[1];
    
        if (quantile_list.size() != n_cols || cell_counts.size() != n_cols)
            throw std::runtime_error("Quantile list and cell count must match column count.");
        
        // Prepare quantile pointers
        std::vector<const TableType*> quantile_ptrs(n_cols);
        std::vector<size_t> quantile_sizes(n_cols);
        for (size_t j = 0; j < n_cols; ++j) {
            auto arr = py::array_t<TableType>(quantile_list[j]);
            py::buffer_info qinfo = arr.request();
            quantile_ptrs[j] = static_cast<const TableType*>(qinfo.ptr);
            quantile_sizes[j] = qinfo.shape[0];
            if (quantile_sizes[j] + 1 != cell_counts[j])
                throw std::runtime_error("cell_counts[j] must be quantiles[j].size() + 1");
        }

        // Allocate output array
        // Strides for an F-contiguous array: 
        //   stride for row dimension = sizeof(QuantizedType)
        //   stride for col dimension = n_rows * sizeof(QuantizedType)

        std::vector<ssize_t> shape    = {(ssize_t)n_rows, (ssize_t)n_cols};
        std::vector<ssize_t> strides  = {sizeof(QuantizedType), (ssize_t)(n_rows * sizeof(QuantizedType))};

        py::array_t<QuantizedType> result(
            py::buffer_info(
                nullptr,                      // let Pybind11 allocate
                sizeof(QuantizedType),
                py::format_descriptor<QuantizedType>::format(),
                2,                            // rank
                shape,
                strides
            )
        );
        QuantizedType* out_ptr = static_cast<QuantizedType*>(result.request().ptr);

        // Quantize
        for (size_t i = 0; i < n_rows; ++i) {
            for (size_t j = 0; j < n_cols; ++j) {
                TableType value = data_ptr[j * n_rows + i];  // column-major access
                auto bin = find_bin(value, quantile_ptrs[j], quantile_sizes[j]);
                out_ptr[j * n_rows + i] = bin;
            }
        }

        return result;
    }
}