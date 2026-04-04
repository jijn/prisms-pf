// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include <prismspf/field_input/read_field_base.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <vtkCellLocator.h>
#include <vtkDataArray.h>
#include <vtkGenericCell.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, typename number>
class ReadVTKGridBase : public ReadFieldBase<dim, number>
{
public:
  /**
   * @brief Constructor
   */
  ReadVTKGridBase(const InitialConditionFile       &_ic_file,
                  const SpatialDiscretization<dim> &_spatial_discretization);

  /**
   * @brief Get the vtk output
   */
  vtkUnstructuredGrid *
  get_output();

  /**
   * @brief Get the number of points
   */
  [[nodiscard]] dealii::types::global_dof_index
  get_n_points() const;

  /**
   * @brief Get the number of cells
   */
  [[nodiscard]] dealii::types::global_dof_index
  get_n_cells() const;

  /**
   * @brief Print the vtk file for debugging
   */
  void
  print_file() override;

  /**
   * @brief Get the names of the scalars in the vtk file.
   */
  std::vector<std::string>
  get_scalars_names();

  /**
   * @brief Get the names of the vectors in the vtk file.
   */
  std::vector<std::string>
  get_vectors_names();

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
   * @brief Pointer to the data from the vtk file
   */
  vtkSmartPointer<vtkUnstructuredGrid> output_grid;

  /**
   * @brief Number of points.
   */
  dealii::types::global_dof_index n_points = 0;

  /**
   * @brief Number of cells.
   */
  dealii::types::global_dof_index n_cells = 0;

  /**
   * @brief Names of scalars in file.
   */
  std::vector<std::string> scalars_names;
  /**
   * @brief Names of vectors in file.
   */
  std::vector<std::string> vectors_names;

  /**
   * @brief Number of points in a hex cell.
   */
  const unsigned int n_points_per_hex_cell = 8;

  /**
   * @brief Number of space coordinates in a point.
   */
  const unsigned int n_space_coordinates = 3;
};

template <unsigned int dim, typename number>
ReadVTKGridBase<dim, number>::ReadVTKGridBase(
  const InitialConditionFile       &_ic_file,
  const SpatialDiscretization<dim> &_spatial_discretization)
  : ReadFieldBase<dim, number>(_ic_file, _spatial_discretization)
{}

template <unsigned int dim, typename number>
inline vtkUnstructuredGrid *
ReadVTKGridBase<dim, number>::get_output()
{
  return output_grid;
}

template <unsigned int dim, typename number>
inline dealii::types::global_dof_index
ReadVTKGridBase<dim, number>::get_n_points() const
{
  return n_points;
}

template <unsigned int dim, typename number>
inline dealii::types::global_dof_index
ReadVTKGridBase<dim, number>::get_n_cells() const
{
  return n_cells;
}

template <unsigned int dim, typename number>
inline void
ReadVTKGridBase<dim, number>::print_file()
{
  if (output_grid != nullptr)
    output_grid->PrintSelf(std::cout, vtkIndent());
}

template <unsigned int dim, typename number>
inline std::vector<std::string>
ReadVTKGridBase<dim, number>::get_scalars_names()
{
  return scalars_names;
}

template <unsigned int dim, typename number>
inline std::vector<std::string>
ReadVTKGridBase<dim, number>::get_vectors_names()
{
  return vectors_names;
}

template <unsigned int dim, typename number>
inline number
ReadVTKGridBase<dim, number>::get_scalar_value(const dealii::Point<dim> &point,
                                               const std::string        &scalar_name)
{
  // Check that the scalar name is in the vtk file
  AssertThrow(std::find(scalars_names.begin(), scalars_names.end(), scalar_name) !=
                scalars_names.end(),
              dealii::ExcMessage("The provided dataset does not contain a field named " +
                                 scalar_name));

  // Convert the dealii point to a vector
  std::vector<double> point_vector = dealii_point_to_vector<dim, double>(point);

  // Grab the point data
  vtkPointData *point_data = output_grid->GetPointData();

  // Find the point id in the vtk file
  const vtkIdType point_id = output_grid->FindPoint(point_vector.data());

  // Check that point is inside the grid
  AssertThrow(point_id >= 0, dealii::ExcMessage("No matching point found in VTK grid"));

  // Check that the point is within some tolerance to know whether we have to interpolate
  // or not
  std::vector<double> point_in_dataset(n_space_coordinates);
  output_grid->GetPoint(point_id, point_in_dataset.data());
  bool interpolate = false;
  for (unsigned int i = 0; i < dim; i++)
    {
      // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
      if (std::abs(point_in_dataset[i] - point_vector[i]) > Defaults::mesh_tolerance)
        interpolate = true;
      // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
    }

  // Get the data array
  vtkDataArray *data_array = point_data->GetArray(scalar_name.c_str());
  AssertThrow(data_array != nullptr,
              dealii::ExcMessage(std::string("Data array not found: ") + scalar_name));

  // Get the value of the scalar at the point
  if (interpolate)
    {
      vtkNew<vtkCellLocator> cell_locator;
      cell_locator->SetDataSet(output_grid);
      cell_locator->BuildLocator();

      std::vector<double> pcoords(n_space_coordinates);
      std::vector<double> weights(n_points_per_hex_cell);
      int                 sub_id = 0;
      vtkGenericCell     *cell   = vtkGenericCell::New();

      // NOLINTBEGIN(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
      const vtkIdType cell_id = cell_locator->FindCell(point_vector.data(),
                                                       Defaults::mesh_tolerance,
                                                       cell,
                                                       sub_id,
                                                       pcoords.data(),
                                                       weights.data());
      // NOLINTEND(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

      AssertThrow(cell_id >= 0,
                  dealii::ExcMessage("Point not inside any cell for interpolation"));

      // Interpolate scalar value using weights and nodal values
      vtkIdList *point_ids          = output_grid->GetCell(cell_id)->GetPointIds();
      number     interpolated_value = 0.0;
      for (vtkIdType id = 0; id < point_ids->GetNumberOfIds(); ++id)
        {
          const vtkIdType pt_id = point_ids->GetId(id);
          // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
          interpolated_value += weights[id] * data_array->GetComponent(pt_id, 0);
          // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
        }
      cell->Delete();
      return interpolated_value;
    }

  // If we are not interpolating, we can just get the value at the point
  return data_array->GetComponent(point_id, 0);
}

template <unsigned int dim, typename number>
inline dealii::Vector<number>
ReadVTKGridBase<dim, number>::get_vector_value(const dealii::Point<dim> &point,
                                               const std::string        &vector_name)
{
  // Check that the scalar name is in the vtk file
  AssertThrow(std::find(vectors_names.begin(), vectors_names.end(), vector_name) !=
                vectors_names.end(),
              dealii::ExcMessage(
                "The provided vtk dataset does not contain a field named " +
                vector_name));

  // Convert the dealii point to a vector
  std::vector<double> point_vector = dealii_point_to_vector<dim, double>(point);

  // Grab the point data
  vtkPointData *point_data = output_grid->GetPointData();

  // Find the point id in the vtk file
  vtkIdType point_id = output_grid->FindPoint(point_vector.data());

  // Check that point is inside the grid
  AssertThrow(point_id >= 0, dealii::ExcMessage("No matching point found in VTK grid"));

  // Check that the point is within some tolerance to know whether we have to interpolate
  // or not
  std::vector<double> point_in_dataset(n_space_coordinates);
  output_grid->GetPoint(point_id, point_in_dataset.data());
  bool interpolate = false;
  for (unsigned int i = 0; i < dim; i++)
    {
      // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
      if (std::abs(point_in_dataset[i] - point_vector[i]) > Defaults::mesh_tolerance)
        interpolate = true;
      // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
    }

  // Get the data array
  vtkDataArray *data_array = point_data->GetArray(vector_name.c_str());
  AssertThrow(data_array != nullptr,
              dealii::ExcMessage(std::string("Data array not found: ") + vector_name));

  // Get the value of the vector at the point
  dealii::Vector<number> vector_value(dim);

  for (unsigned int i = 0; i < dim; i++)
    {
      if (interpolate)
        {
          vtkNew<vtkCellLocator> cell_locator;
          cell_locator->SetDataSet(output_grid);
          cell_locator->BuildLocator();

          std::vector<double> pcoords(n_space_coordinates);
          std::vector<double> weights(n_points_per_hex_cell);
          int                 sub_id = 0;
          vtkGenericCell     *cell   = vtkGenericCell::New();

          // NOLINTBEGIN(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
          const vtkIdType cell_id = cell_locator->FindCell(point_vector.data(),
                                                           Defaults::mesh_tolerance,
                                                           cell,
                                                           sub_id,
                                                           pcoords.data(),
                                                           weights.data());
          // NOLINTEND(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

          AssertThrow(cell_id >= 0,
                      dealii::ExcMessage("Point not inside any cell for interpolation"));

          // Interpolate scalar value using weights and nodal values
          vtkIdList *point_ids          = output_grid->GetCell(cell_id)->GetPointIds();
          number     interpolated_value = 0.0;
          for (vtkIdType id = 0; id < point_ids->GetNumberOfIds(); ++id)
            {
              Assert(id < data_array->GetNumberOfComponents(),
                     dealii::ExcMessage("Index out of bounds for data array components"));
              const vtkIdType pt_id = point_ids->GetId(id);
              // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
              interpolated_value +=
                weights[id] * data_array->GetComponent(pt_id, static_cast<int>(id));
              // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
            }
          vector_value[i] = interpolated_value;
          cell->Delete();
        }
      else
        {
          vector_value[i] = data_array->GetComponent(point_id, i);
        }
    }
  return vector_value;
}

PRISMS_PF_END_NAMESPACE
