#include "create/inverter.hpp"
#include "common_types.hpp"
#include <iostream>


// ---------------------------------------------------------------------
// Constants that govern page alignment and blob growth
// ---------------------------------------------------------------------
namespace TeamIndex
{

// ---------------------------------------------------------------------
// TableInverter  — header already included in the canvas
// ---------------------------------------------------------------------
void TableInverter::process_cell(const std::vector<uint8_t>& key,
                                 const std::vector<IDType>& row_ids_input,
                                 CodecID codec_id)
{
    if (row_ids_input.empty()) return;                 // nothing to write

    //------------------------------------------------------------------
    // 1) Compute linear index of this key
    //------------------------------------------------------------------
    size_t linear_index = 0;
    for (size_t d = 0; d < key.size(); ++d) {
        linear_index += static_cast<size_t>(key[d]) * multipliers[d];
    }

    //------------------------------------------------------------------
    // 2) Helper to grow output_blob in coarse 32‑MiB chunks
    //------------------------------------------------------------------
    auto reserve_extra = [this](std::size_t extra)
    {
        if (output_blob.capacity() - output_blob.size() < extra) {
            std::size_t need     = output_blob.size() + extra;
            std::size_t new_cap  = ((need + TableInverter::CHUNK_BYTES - 1)
                                   / TableInverter::CHUNK_BYTES) * TableInverter::CHUNK_BYTES;
            output_blob.reserve(new_cap);
        }
    };

    //------------------------------------------------------------------
    // 3) Page‑align: every list starts on a fresh 4 KiB page
    //------------------------------------------------------------------
    std::size_t pad = (TeamIndex::PAGESIZE - (output_blob.size() % TeamIndex::PAGESIZE)) % TeamIndex::PAGESIZE;
    reserve_extra(pad);
    output_blob.insert(output_blob.end(), pad, 0);

    //------------------------------------------------------------------
    // 4) Fill intermediate page offsets for empty bins
    //------------------------------------------------------------------
    for (size_t i = last_written_index + 1; i < linear_index; ++i) {
        page_offset_ptr[i] = page_offset_ptr[last_written_index];
    }
    last_written_index = linear_index;
    page_offset_ptr[linear_index] = static_cast<IDType>(output_blob.size() / TeamIndex::PAGESIZE);

    //------------------------------------------------------------------
    // 5) Compress the list (COPY codec or real codec via encode)
    //------------------------------------------------------------------
    std::size_t raw_bytes    = row_ids_input.size() * sizeof(IDType);
    std::size_t max_out = static_cast<std::size_t>(std::ceil(raw_bytes * SIZE_OVERHEAD_TOLERANCE)) + TeamIndex::PAGESIZE;
    reserve_extra(max_out);
    std::size_t insert_pos = output_blob.size();
    output_blob.resize(insert_pos + max_out);

    std::span<char> out_span(output_blob.data() + insert_pos, max_out);

    auto [comp_size, used_codec] = encode(codec_id, row_ids_input, out_span, row_ids_input.size());

    output_blob.resize(insert_pos + comp_size);        // trim to real size

    //------------------------------------------------------------------
    // 6) Enforce 10 GB hard cap
    //------------------------------------------------------------------
    if (output_blob.size() > TableInverter::MAX_BLOB_GB * (1ull << 30)) {
        throw std::runtime_error("output_blob exceeded "
                                 + std::to_string(TableInverter::MAX_BLOB_GB)
                                 + " GB — aborting");
    }

    //------------------------------------------------------------------
    // 7) Write metadata
    //------------------------------------------------------------------
    cardinality_ptr[linear_index]      = static_cast<IDType>(row_ids_input.size());
    compressed_size_ptr[linear_index]  = comp_size;
    used_codec_ptr[linear_index]       = used_codec;
}

// ---------------------------------------------------------------------
// invert_quantized_table 
// ---------------------------------------------------------------------
py::tuple TableInverter::invert_quantized_table(
    py::array_t<uint8_t,  py::array::f_style | py::array::forcecast>  table,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast>  row_ids,
    py::list                                                    cell_counts,
    CodecID                                                     codec_id)
{
    py::buffer_info info = table.request();
    const uint8_t* data_ptr = static_cast<uint8_t*>(info.ptr);
    const size_t n_rows = info.shape[0];
    const size_t n_dims = info.shape[1];

    std::cout << "Table shape: " << info.shape[0] << " x " << info.shape[1] << std::endl;

    py::buffer_info row_info = row_ids.request();
    const uint64_t* row_id_ptr = static_cast<uint64_t*>(row_info.ptr);
    if (row_info.shape[0] != n_rows) {
        throw std::runtime_error("Row ID array must match the number of table rows.");
    }

    // Extract cell counts from provided list
    std::vector<size_t> counts;
    for (auto item : cell_counts) {
        counts.push_back(py::cast<size_t>(item));
    }
    if (counts.size() != n_dims) {
        throw std::runtime_error("bin_counts length must match number of dimensions");
    }

    // Compute total number of cells and cell index multipliers
    size_t total_cells = 1;
    multipliers.resize(n_dims);
    for (size_t d = 0; d < n_dims; ++d) {
        multipliers[d] = (d == 0) ? 1 : multipliers[d - 1] * counts[d - 1];
        total_cells *= counts[d];
    }

    // Consistency check for overflow
    if (total_cells == 0) {
        throw std::runtime_error("Total number of bins is zero. Check bin counts.");
    }

    std::cout << "Total bins: " << total_cells << std::endl;
    std::cout << "Bin counts: ";
    for (size_t i = 0; i < n_dims; ++i) {
        std::cout << counts[i] << " ";
    }
    std::cout << std::endl;
    std::size_t worst_case_bytes =
    std::min<std::size_t>(n_rows, total_cells) * TeamIndex::PAGESIZE;
    
    if (worst_case_bytes > TableInverter::MAX_BLOB_GB * (1ull << 30)) {
        std::cerr << "[WARN] worst-case blob may exceed "
                  << TableInverter::MAX_BLOB_GB << " GB ("
                  << worst_case_bytes / (1ull<<30) << " GB)\n";
    }
    

    // Use a dynamic byte buffer to accumulate output
    output_blob.clear();
          
    py::array_t<IDType> cardinalities(total_cells);
    py::array_t<std::size_t> compressed_sizes(total_cells);
    py::array_t<CodecID> used_codecs(total_cells);
    py::array_t<IDType> page_offsets(total_cells + 1); // extra entry for total page count

    auto card_info = cardinalities.request();
    auto size_info = compressed_sizes.request();
    auto codec_info = used_codecs.request();
    auto offset_info = page_offsets.request();

    cardinality_ptr = static_cast<IDType*>(card_info.ptr);
    compressed_size_ptr = static_cast<std::size_t*>(size_info.ptr);
    used_codec_ptr = static_cast<CodecID*>(codec_info.ptr);
    page_offset_ptr = static_cast<IDType*>(offset_info.ptr);

    std::fill_n(cardinality_ptr, total_cells, 0);
    std::fill_n(compressed_size_ptr, total_cells, 0);
    std::fill_n(used_codec_ptr, total_cells, CodecID::UNKNOWN);

    
    // print out how large (in MB) the arrays are we will return
    auto print_percentage = 10; // Progress indicator: print every x% of the rows
    bool print_progress = true;
    // do not print if n_rows < 1000
    if (n_rows < 1000) {
        print_progress = false;
    }
    else if (n_rows > 1000000) {
        print_percentage = 1; // 1% for large tables
    }
    std::size_t step = n_rows * print_percentage / 100;   // e.g. 10 % → every  n_rows/10  iterations
    if (step == 0) step = 1;                              // protect against tiny loops
    std::cout << "Starting inversion process..." << std::endl;
    auto total_size = (total_cells * (sizeof(IDType) + sizeof(std::size_t) + sizeof(CodecID) + sizeof(IDType))) / (1024 * 1024);
    std::cout << "Output size (metadata): " << total_size << " MB" << std::endl;
    
    
    // --- main loop ----------------------------------------------------
    // Iterate over sorted table and detect new cell boundaries
    std::vector<IDType> current_row_ids;
    std::vector<uint8_t> current_key(n_dims);
    std::vector<uint8_t> prev_key(n_dims, 255);   // sentinel

    for (size_t i = 0; i < n_rows; ++i) {
        bool is_new_cell = false;
        for (size_t rev_j = 0; rev_j < n_dims; ++rev_j) {
            size_t j = (n_dims - 1) - rev_j;
            current_key[j] = data_ptr[j * n_rows + i];
            if (!is_new_cell && current_key[j] != prev_key[j]) {
                is_new_cell = true;
            }
        }

        if (is_new_cell) {
            if (!current_row_ids.empty()) {
                process_cell(prev_key, current_row_ids, codec_id);
                current_row_ids.clear();
            }
            prev_key = current_key;
        }
        current_row_ids.push_back(row_id_ptr[i]);

        // print progress every print_percentage% of the rows
        if (print_progress && (i % step == 0 || i + 1 == n_rows)) {
            std::cout << "Progress: " << (i + 1) * 100 / n_rows << "%\r";
            std::cout.flush();
        }
    }

    if (!current_row_ids.empty())
        process_cell(current_key, current_row_ids, codec_id);

    //------------------------------------------------------------------
    // Final page padding + fill trailing page offsets for empty cells
    //------------------------------------------------------------------
    std::size_t final_pad = (TeamIndex::PAGESIZE - (output_blob.size() % TeamIndex::PAGESIZE)) % TeamIndex::PAGESIZE;
    output_blob.insert(output_blob.end(), final_pad, 0);

    for (size_t i = last_written_index + 1; i < total_cells; ++i)
        page_offset_ptr[i] = page_offset_ptr[last_written_index];

    page_offset_ptr[total_cells] = static_cast<IDType>(output_blob.size() / TeamIndex::PAGESIZE);

    std::cout << "Output size (compressed): " << output_blob.size() / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Final page count: " << page_offset_ptr[total_cells] << std::endl;
    
    for (size_t i = 0; i < total_cells; ++i) {
        if (cardinality_ptr[i] == 0) {
            compressed_size_ptr[i] = 0;
            used_codec_ptr[i]      = CodecID::UNKNOWN;   // or 0
        }
    }
    
    std::cout << "Inversion process completed." << std::endl;
    //------------------------------------------------------------------
    // Return tuple
    //------------------------------------------------------------------
    py::bytes blob_py(reinterpret_cast<const char*>(output_blob.data()), output_blob.size());
    return py::make_tuple(blob_py, cardinalities, compressed_sizes, used_codecs, page_offsets);
}
}