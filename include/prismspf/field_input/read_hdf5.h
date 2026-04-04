// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <deal.II/base/hdf5.h>

#include <prismspf/field_input/read_structured_base.h>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, typename number>
class ReadHDF5 : public ReadStructuredGridBase<dim, number>
{
public:
  ReadHDF5(const InitialConditionFile       &_ic_file,
           const SpatialDiscretization<dim> &_spatial_discretization);

  static void
  write_file(const std::vector<number> &data, const InitialConditionFile &ic_file);

private:
  void
  check_dataset_size(const std::vector<dealii::HDF5::hsize_type> &dims);
};

template <unsigned int dim, typename number>
inline ReadHDF5<dim, number>::ReadHDF5(
  const InitialConditionFile       &_ic_file,
  const SpatialDiscretization<dim> &_spatial_discretization)
  : ReadStructuredGridBase<dim, number>(_ic_file, _spatial_discretization)
{
  AssertThrow(this->ic_file.dataset_format == DataFormatType::HDF5,
              dealii::ExcMessage("Dataset format must be HDF5"));

  AssertThrow(this->ic_file.file_variable_names.size() == 1 &&
                this->ic_file.simulation_variable_names.size() == 1,
              dealii::ExcMessage("Only one field can be read in from a raw HDF5 file"));

  AssertThrow(_spatial_discretization.type == TriangulationType::Rectangular,
              dealii::ExcMessage(
                "Only rectangular domains are supported for raw HDF5 input files"));

  for (unsigned int d : std::views::iota(0U, dim))
    this->n_points *= this->ic_file.n_data_points[d];

  dealii::HDF5::File data_file(this->ic_file.filename,
                               dealii::HDF5::File::FileAccessMode::read);

  std::string dataset_name = this->ic_file.file_variable_names[0];
  auto        dataset      = data_file.open_dataset(dataset_name);

  auto dims = dataset.get_dimensions();
  check_dataset_size(dims);

  this->data.resize(this->n_values);
  dataset.read(this->data);
}

template <unsigned int dim, typename number>
inline void
ReadHDF5<dim, number>::check_dataset_size(
  const std::vector<dealii::HDF5::hsize_type> &dims)
{
  dealii::types::global_dof_index file_size = 1;
  for (auto d : dims)
    file_size *= d;

  auto expected_size_scalar = this->n_points;
  auto expected_size_vector = dim * expected_size_scalar;

  AssertThrow(expected_size_scalar != 0,
              dealii::ExcMessage("Expected input array size is zero."));

  AssertThrow(file_size == expected_size_scalar || file_size == expected_size_vector,
              dealii::ExcMessage(
                "Expected HDF5 dataset size does not match actual dataset size."));

  this->n_values =
    (file_size == expected_size_scalar) ? this->n_points : dim * this->n_points;
}

template <unsigned int dim, typename number>
inline void
ReadHDF5<dim, number>::write_file(const std::vector<number>  &data,
                                  const InitialConditionFile &ic_file)
{
  dealii::HDF5::File data_file(ic_file.filename,
                               dealii::HDF5::File::FileAccessMode::create);

  std::vector<dealii::HDF5::hsize_type> dims = {data.size()};

  std::string dataset_name =
    ic_file.file_variable_names.empty() ? "data" : ic_file.file_variable_names[0];

  auto dataset = data_file.create_dataset<number>(dataset_name, dims);
  dataset.write(data);
}

PRISMS_PF_END_NAMESPACE
