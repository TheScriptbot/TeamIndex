#pragma once

#include "ti_codecs.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace TeamIndex {

class TableInverter {
private:        
    std::vector<char> output_blob;
    IDType* cardinality_ptr;
    std::size_t* compressed_size_ptr;
    CodecID* used_codec_ptr;
    IDType* page_offset_ptr;
    std::vector<size_t> multipliers;
    size_t last_written_index = 0;
public:
    const std::size_t CHUNK_BYTES  = 32ull << 20;     // grow output_blob in 32 MiB steps
    const std::size_t MAX_BLOB_GB  = 15;              // abort if blob > 15 GB

    TableInverter() = default;
    ~TableInverter() = default;

    void process_cell(const std::vector<uint8_t>& key, const std::vector<IDType>& row_ids, CodecID codec_id);
    
    
    py::tuple invert_quantized_table(
        py::array_t<uint8_t, py::array::f_style | py::array::forcecast> table,
        py::array_t<uint64_t, py::array::c_style | py::array::forcecast> row_ids,
        py::list cell_counts,
        CodecID codec_id);
};
}
