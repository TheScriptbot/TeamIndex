
#include "create/create.hpp"
#include "create/quantizer.hpp"
#include "create/inverter.hpp"

namespace py = pybind11;

void define_creation_interface(py::module &m) {
    ////////////////////// CONVERTER //////////////////////
    using Converter = TeamIndex::BatchConverter;
    py::class_<Converter, std::unique_ptr<Converter>> converter(m, "BatchConverter");

    converter.def(py::init([](const py::buffer &quantile_array) {
        py::buffer_info info = quantile_array.request(false);
        if (info.format != py::format_descriptor<Converter::data_type>::format()) {
            throw std::runtime_error("Incompatible format of quantile array!");
        }

        /* Some sanity checks ... */
        if (info.ndim != 2) {
            throw std::runtime_error("Incompatible quantile array dimension!");
        }
        if (info.shape[0] <= 0u or info.shape[1] <= 0u) {
            throw std::runtime_error("Incompatible shape of quantile array!");
        }
        /// Fetch quantile values from buffer.
        ///
        /// Assumed to be a COLUMN-STORE/Fortran storage order.
        ///
        /// We assume they are sorted in ascending order!
        /// the dimensions of the quantile array are  `\forall i: max(b_i) \times |Team|`
        auto quantile_cnt = static_cast<size_t>(info.shape[0]);
        auto cols = static_cast<size_t>(info.shape[1]);
        return std::make_unique<Converter>((Converter::data_type*) info.ptr, quantile_cnt, cols);
    }));

    converter.def("process_batch", &Converter::process_batch);

    converter.def("intermediate_result_count",
                  &Converter::intermediate_result_count);

    converter.def("get_result", // Method destroys index data within the class and moves it to python... yes?
                  &Converter::get_result,
                  py::arg("codec_name"),
                  py::arg("batch_id") = 0u
            ,py::return_value_policy::move
    );

    converter.def("dump_index", // Method destroys index data within the class, dumps data in binary files
                  &Converter::dump_index);
    ////////////////////// Quantizer (alternative creation method) //////////////////////
    
    using TableQuantizer = TeamIndex::TableQuantizer;
    py::class_<TableQuantizer, std::unique_ptr<TableQuantizer>> quantizer(m, "TableQuantizer");
    
    quantizer.def_static("quantize_table",
                  &TableQuantizer::quantize_table,
                  py::arg("table"),
                  py::arg("quantile_list"),
                  py::arg("cell_counts"),
                  py::return_value_policy::move,
                  R"pbdoc(
                      Quantizes a table using the provided quantiles and cell counts.
                      Args:
                          table: A 2D numpy array to be quantized.
                          quantile_list: A list of quantiles for each column.
                          cell_counts: A list of cell counts for each column.
                      Returns:
                          A 2D numpy array with the quantized values.
                  )pbdoc"
    );

    using TableInverter = TeamIndex::TableInverter;
    py::class_<TableInverter, std::unique_ptr<TableInverter>> inverter(m, "TableInverter");
    inverter.def(py::init<>());
    inverter.def("invert_quantized_table",
                  &TableInverter::invert_quantized_table,
                  py::arg("table"),
                  py::arg("row_ids"),
                  py::arg("cell_counts"),
                  py::arg("codec_name"),
                  R"pbdoc(
                      Inverts a quantized table into inverted lists.
                      Args:
                          table: A 2D numpy array to be inverted. Fotran order!
                          row_ids: A 1D numpy array of row IDs.
                          cell_counts: A list of cell counts for each column.
                          codec_name: The codec to use for compression.
                      Returns:
                          A tuple containing the inverted lists as a binary blob, as well as 3 matrices with metadata (cardinality, compressed size, compression).
                  )pbdoc"
    );

    ////////////////////// OTHER //////////////////////

    py::enum_<TeamIndex::CodecID> codec_enum(m,"CodecID");
    codec_enum.value(TeamIndex::to_string(TeamIndex::CodecID::UNKNOWN).c_str(),TeamIndex::CodecID::UNKNOWN, "Unkown encoding, likely represents an empty list.");
    codec_enum.value(TeamIndex::to_string(TeamIndex::CodecID::COPY).c_str(),TeamIndex::CodecID::COPY, "Simple copy, no encoding.");
    codec_enum.value(TeamIndex::to_string(TeamIndex::CodecID::ZSTD).c_str(),TeamIndex::CodecID::ZSTD, "General Purpose zstd compression applied on Deltas.");
    codec_enum.value(TeamIndex::to_string(TeamIndex::CodecID::ZSTDMORE).c_str(),TeamIndex::CodecID::ZSTDMORE, "General Purpose zstd compression applied on Deltas. Higher compression ratio.");
    codec_enum.value(TeamIndex::to_string(TeamIndex::CodecID::ZSTDFASTER).c_str(),TeamIndex::CodecID::ZSTDFASTER, "General Purpose zstd compression applied on Deltas. Higher decompression speed.");
    codec_enum.value(TeamIndex::to_string(TeamIndex::CodecID::ROARING).c_str(),TeamIndex::CodecID::ROARING, "Roaring bitmaps");
#ifdef ENABLE_FASTPFOR
    codec_enum.value(TeamIndex::to_string(TeamIndex::CodecID::VARINT).c_str(),TeamIndex::CodecID::VARINT, "Variable-length integer encoding with Delta encoding.");
    codec_enum.value(TeamIndex::to_string(TeamIndex::CodecID::VARINTZSTD).c_str(),TeamIndex::CodecID::VARINTZSTD, "Variable-length and also zstd on top.");
    codec_enum.value(TeamIndex::to_string(TeamIndex::CodecID::SIMDOPTPFOR).c_str(),TeamIndex::CodecID::SIMDOPTPFOR);
    codec_enum.value(TeamIndex::to_string(TeamIndex::CodecID::SIMDSIMPLEPFOR).c_str(),TeamIndex::CodecID::SIMDSIMPLEPFOR);
    codec_enum.value(TeamIndex::to_string(TeamIndex::CodecID::SIMDFASTPFOR256).c_str(),TeamIndex::CodecID::SIMDFASTPFOR256);
#endif
#ifdef ENABLE_MORE_COMPRESSIONS
    codec_enum.value(TeamIndex::to_string(TeamIndex::CodecID::WAH).c_str(),TeamIndex::CodecID::WAH, "WAH bitmap compression from the FastBit-project.");
    codec_enum.value(TeamIndex::to_string(TeamIndex::CodecID::TEB).c_str(),TeamIndex::CodecID::TEB, "Tree Encoded Bitmaps");
#endif

    m.def("string_to_codec_id", &TeamIndex::string_to_codec);
    m.def("codec_id_to_string", py::overload_cast<TeamIndex::CodecID>(&TeamIndex::to_string));
}
