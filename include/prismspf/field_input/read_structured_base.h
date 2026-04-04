// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include <prismspf/core/conditional_ostreams.h>
#include <prismspf/core/types.h>

#include <prismspf/field_input/read_field_base.h>

#include <array>
#include <cmath>
#include <iostream>
#include <ranges>
#include <string>
#include <vector>

PRISMS_PF_BEGIN_NAMESPACE

/**
 * @brief Base class for structured, flat array inputs (Binary, Raw HDF5)
 */
template <unsigned int dim, typename number>
class ReadStructuredGridBase : public ReadFieldBase<dim, number>
{
public:
  ReadStructuredGridBase(const InitialConditionFile       &_ic_file,
                         const SpatialDiscretization<dim> &_spatial_discretization);

  /**
   * @brief Print the file to text for debugging
   */
  void
  print_file() override;

  /**
   * @brief Get scalar value for a given point
   */
  number
  get_scalar_value(const dealii::Point<dim> &point,
                   const std::string        &scalar_name) override;

  /**
   * @brief Get vector value for a given point
   */
  dealii::Vector<number>
  get_vector_value(const dealii::Point<dim> &point,
                   const std::string        &vector_name) override;

protected:
  /**
   * @brief Get vector value for a given index
   */
  dealii::Vector<number>
  get_value(const dealii::types::global_dof_index index, const unsigned int n_components);

  /**
   * @brief Get vector value for a given point
   *
   * This function is necessary for binary files because we don't have a strict order of
   * points that deal.II gives us in the initial condition function. For that reason, we
   * have to interpolate the flat binary values assuming a rectangular grid with fixed
   * spacing. Furthermore, we assume that the first point is the origin (0,0,0) and the
   * last the furthest corner (increasing x, then y, and z).
   */
  dealii::Vector<number>
  interpolate(const dealii::Point<dim> &point, const unsigned int n_components);

  /**
   * @brief Number of grid points.
   *
   * We have to set the initial value to 1 so that when we multiply in the constructor
   * we don't end up with zero.
   */
  dealii::types::global_dof_index n_points = 1;

  /**
   * @brief Number of values (n_points * n_components).
   */
  dealii::types::global_dof_index n_values = 0;

  /**
   * @brief Data array to hold the read in values.
   */
  std::vector<number> data;
};

template <unsigned int dim, typename number>
inline ReadStructuredGridBase<dim, number>::ReadStructuredGridBase(
  const InitialConditionFile       &_ic_file,
  const SpatialDiscretization<dim> &_spatial_discretization)
  : ReadFieldBase<dim, number>(_ic_file, _spatial_discretization)
{}

template <unsigned int dim, typename number>
inline dealii::Vector<number>
ReadStructuredGridBase<dim, number>::get_value(
  const dealii::types::global_dof_index index,
  const unsigned int                    n_components)
{
  // Create vector to hold the value with the correct number of components
  dealii::Vector<number> value(n_components);

  // Check that the number of components matches what we expect from the number of values
  Assert(n_values % n_components == 0,
         dealii::ExcMessage("The number of components requested in the get_value call "
                            "does not match the number of values in the file."));

  // Fill the value vector
  for (unsigned int component : std::views::iota(0U, n_components))
    {
      Assert(((n_components * index) + component) < data.size(),
             dealii::ExcMessage(
               "Index out of bounds in ReadStructuredGridBase::get_value"));
      value[component] = data[(n_components * index) + component];
    }
  return value;
}

template <unsigned int dim, typename number>
inline dealii::Vector<number>
ReadStructuredGridBase<dim, number>::interpolate(const dealii::Point<dim> &point,
                                                 const unsigned int        n_components)
{
  Assert(n_components == 1 || n_components == dim,
         dealii::ExcMessage(
           "Number of components for interpolation must be 1 (scalar) or dim (vector)"));

  // Create a vector to hold the interpolated value
  dealii::Vector<number> value(n_components);

  // Get the spatial discretization
  const auto &spatial_discretization = this->spatial_discretization;

  // Compute the spacing in each direction
  std::array<number, dim> spacing;
  for (unsigned int d : std::views::iota(0U, dim))
    {
      spacing[d] = spatial_discretization.rectangular_mesh.size[d] /
                   static_cast<number>(this->ic_file.n_data_points[d]);
    }

  // Compute the indices of the lower corner of the cell containing the point
  std::array<dealii::types::global_dof_index, dim> lower_indices;
  for (unsigned int d : std::views::iota(0U, dim))
    {
      lower_indices[d] =
        static_cast<dealii::types::global_dof_index>(std::floor(point[d] / spacing[d]));
      // Make sure we don't go out of bounds
      if (lower_indices[d] >= this->ic_file.n_data_points[d] - 1)
        {
          lower_indices[d] = this->ic_file.n_data_points[d] - 2;
        }
    }

  // Compute the weights for interpolation in each direction
  std::array<number, dim> weights;
  for (unsigned int d : std::views::iota(0U, dim))
    {
      weights[d] = (point[d] - lower_indices[d] * spacing[d]) / spacing[d];
    }

  // Perform multilinear interpolation based on the dimension
  if constexpr (dim == 1)
    {
      // Here is the map of the nodes in 1D:
      // 0 — 1

      // Grab the index of the lower left corner of the cell
      auto lower_index = lower_indices[0];

      // Get the values of the two points in 1D
      auto value_0 = get_value(lower_index, n_components);
      auto value_1 = get_value(lower_index + 1, n_components);

      // Interpolate
      for (unsigned int c : std::views::iota(0U, n_components))
        value[c] = (1.0 - weights[0]) * value_0[c] + weights[0] * value_1[c];
    }
  else if constexpr (dim == 2)
    {
      // Here is the map of the nodes in 2D:
      // 01 — 11
      // |     |
      // 00 — 10

      // Grab the row length in the x-direction (0th direction)
      auto row_length_0 = this->ic_file.n_data_points[0];
      // Grab the index of the lower left corner of the cell
      auto lower_index = lower_indices[0] + (lower_indices[1] * row_length_0);

      // Get the values of the four points in 2D
      auto value_00 = get_value(lower_index, n_components);
      auto value_10 = get_value(lower_index + 1, n_components);
      auto value_01 = get_value(lower_index + row_length_0, n_components);
      auto value_11 = get_value(lower_index + row_length_0 + 1, n_components);

      // Interpolate
      for (unsigned int c : std::views::iota(0U, n_components))
        {
          value[c] = (1.0 - weights[0]) * (1.0 - weights[1]) * value_00[c] +
                     weights[0] * (1.0 - weights[1]) * value_10[c] +
                     (1.0 - weights[0]) * weights[1] * value_01[c] +
                     weights[0] * weights[1] * value_11[c];
        }
    }
  else if constexpr (dim == 3)
    {
      // Here is the map of the nodes in 3D:
      //
      //   011 ———— 111
      //   / |      / |
      //  /  001   /  |
      // 010 ——— 110 101
      // |  /      |  /
      // | /       | /
      // 000 ———— 100

      // Grab the row length in the x-direction (0th direction)
      auto row_length_0 = this->ic_file.n_data_points[0];
      // Grab the row length in the y-direction (1st direction)
      auto row_length_1 = this->ic_file.n_data_points[1];

      // Grab the index of the lower left corner of the cell
      auto lower_index = lower_indices[0] + (lower_indices[1] * row_length_0) +
                         (lower_indices[2] * row_length_0 * row_length_1);

      // Get the values of the eight points in 3D
      auto value_000 = get_value(lower_index, n_components);
      auto value_100 = get_value(lower_index + 1, n_components);
      auto value_010 = get_value(lower_index + row_length_0, n_components);
      auto value_110 = get_value(lower_index + row_length_0 + 1, n_components);
      auto value_001 =
        get_value(lower_index + (row_length_0 * row_length_1), n_components);
      auto value_101 =
        get_value(lower_index + (row_length_0 * row_length_1) + 1, n_components);
      auto value_011 =
        get_value(lower_index + (row_length_0 * row_length_1) + row_length_0,
                  n_components);
      auto value_111 =
        get_value(lower_index + (row_length_0 * row_length_1) + row_length_0 + 1,
                  n_components);

      // Interpolate
      for (unsigned int c : std::views::iota(0U, n_components))
        {
          value[c] =
            (1.0 - weights[0]) * (1.0 - weights[1]) * (1.0 - weights[2]) * value_000[c] +
            weights[0] * (1.0 - weights[1]) * (1.0 - weights[2]) * value_100[c] +
            (1.0 - weights[0]) * weights[1] * (1.0 - weights[2]) * value_010[c] +
            weights[0] * weights[1] * (1.0 - weights[2]) * value_110[c] +
            (1.0 - weights[0]) * (1.0 - weights[1]) * weights[2] * value_001[c] +
            weights[0] * (1.0 - weights[1]) * weights[2] * value_101[c] +
            (1.0 - weights[0]) * weights[1] * weights[2] * value_011[c] +
            weights[0] * weights[1] * weights[2] * value_111[c];
        }
    }
  return value;
}

template <unsigned int dim, typename number>
inline number
ReadStructuredGridBase<dim, number>::get_scalar_value(
  const dealii::Point<dim>           &point,
  [[maybe_unused]] const std::string &scalar_name)
{
  Assert(this->n_values == this->n_points,
         dealii::ExcMessage("The number of points should match the number of values in a "
                            "file for a scalar field. Make sure the file size is correct "
                            "and you are trying to access a scalar field."));
  return interpolate(point, 1)[0];
}

template <unsigned int dim, typename number>
inline dealii::Vector<number>
ReadStructuredGridBase<dim, number>::get_vector_value(
  const dealii::Point<dim>           &point,
  [[maybe_unused]] const std::string &vector_name)
{
  Assert((this->n_values / dim) == this->n_points,
         dealii::ExcMessage("The number of points should match the number of values "
                            "divided by the dimension in a file for a vector field. "
                            "Make sure the file size is correct and you are trying "
                            "to access a vector field."));
  return interpolate(point, dim);
}

template <unsigned int dim, typename number>
inline void
ReadStructuredGridBase<dim, number>::print_file()
{
  for (dealii::types::global_dof_index i : std::views::iota(0U, this->n_values))
    {
      Assert(i < data.size(),
             dealii::ExcMessage(
               "Index out of bounds in ReadStructuredGridBase::print_file"));
      ConditionalOStreams::pout_summary() << this->data.at(i) << "\n";
    }
  ConditionalOStreams::pout_summary() << std::flush;
}

PRISMS_PF_END_NAMESPACE
