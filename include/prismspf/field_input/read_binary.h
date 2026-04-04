// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <prismspf/field_input/read_structured_base.h>

#include <bit>
#include <filesystem>
#include <fstream>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, typename number>
class ReadBinary : public ReadStructuredGridBase<dim, number>
{
public:
  ReadBinary(const InitialConditionFile       &_ic_file,
             const SpatialDiscretization<dim> &_spatial_discretization);

  static void
  write_file(const std::vector<number> &data, const InitialConditionFile &ic_file);

private:
  void
  check_file_size();
};

template <unsigned int dim, typename number>
inline ReadBinary<dim, number>::ReadBinary(
  const InitialConditionFile       &_ic_file,
  const SpatialDiscretization<dim> &_spatial_discretization)
  : ReadStructuredGridBase<dim, number>(_ic_file, _spatial_discretization)
{
  // Make sure the dataset format is correct
  AssertThrow(this->ic_file.dataset_format == DataFormatType::FlatBinary,
              dealii::ExcMessage("Dataset format must be FlatBinary"));

  // Check that only one field is being read in
  AssertThrow(this->ic_file.file_variable_names.size() == 1 &&
                this->ic_file.simulation_variable_names.size() == 1,
              dealii::ExcMessage("Only one field can be read in from a binary file"));

  // Make sure we have a rectangular domain
  AssertThrow(_spatial_discretization.type == TriangulationType::Rectangular,
              dealii::ExcMessage(
                "Only rectangular domains are supported for binary input files"));

  // Compute the total number of points in the binary file
  for (unsigned int d : std::views::iota(0U, dim))
    this->n_points *= this->ic_file.n_data_points[d];

  // Check that the binary matches an expected size
  check_file_size();

  // Read in the binary file
  std::ifstream data_file(this->ic_file.filename, std::ios::binary);
  AssertThrow(data_file,
              dealii::ExcMessage("Could not open binary file: " +
                                 this->ic_file.filename));

  // Reserve space in the data vectors
  this->data.reserve(this->n_values);

  // Read in the data
  for (dealii::types::global_dof_index i : std::views::iota(0U, this->n_values))
    {
      std::array<char, sizeof(number)> buffer;
      data_file.read(buffer.data(), sizeof(number));
      this->data.push_back(std::bit_cast<number>(buffer));
    }
  data_file.close();
}

template <unsigned int dim, typename number>
inline void
ReadBinary<dim, number>::check_file_size()
{
  // Grab the file size of the binary file in bytess
  auto file_size = std::filesystem::file_size(this->ic_file.filename);

  // Compute the expected size of the binary file. This is simply the number of points
  // multiplied by the size of each point in bytes.
  auto expected_size_scalar =
    static_cast<std::uintmax_t>(this->n_points * sizeof(number));
  auto expected_size_vector = static_cast<std::uintmax_t>(dim * expected_size_scalar);

  // Make sure expected size is not zero
  AssertThrow(
    expected_size_scalar != 0 && expected_size_vector != 0,
    dealii::ExcMessage(
      "Expected input array size is zero, check that the number of data points "
      "in each used direction is set correctly in the input file for your binary file. "
      "You likely have the number of data points set to zero in all directions."));
  // Make sure the size matches for either a scalar or vector
  AssertThrow(file_size == expected_size_scalar || file_size == expected_size_vector,
              dealii::ExcMessage(
                "Expected binary file size (" + std::to_string(expected_size_scalar) +
                " bytes for scalar or " + std::to_string(expected_size_vector) +
                " bytes for vector) does not match actual file size (" +
                std::to_string(file_size) + " bytes)."));

  // Set the number of values
  this->n_values =
    file_size == expected_size_scalar ? this->n_points : dim * this->n_points;
}

template <unsigned int dim, typename number>
inline void
ReadBinary<dim, number>::write_file(const std::vector<number>  &data,
                                    const InitialConditionFile &ic_file)
{
  // Try to open the file
  std::ofstream data_file(ic_file.filename, std::ios::binary);
  AssertThrow(data_file,
              dealii::ExcMessage("Could not open binary file: " + ic_file.filename));

  // Write the data
  for (dealii::types::global_dof_index j : std::views::iota(0U, data.size()))
    {
      auto buffer = std::bit_cast<std::array<char, sizeof(number)>>(data[j]);
      data_file.write(buffer.data(), sizeof(number));
    }
  data_file.close();
}

PRISMS_PF_END_NAMESPACE
