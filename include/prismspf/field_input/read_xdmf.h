// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <prismspf/field_input/read_vtk_base.h>

#include <vtkXdmfReader.h>

PRISMS_PF_BEGIN_NAMESPACE

/**
 * @brief Class to read in an XDMF file (with associated HDF5 data) and populate an
 * unstructured grid.
 * This class inherits from ReadVTKGridBase. It utilizes VTK's XDMF reader to parse
 * the XDMF header and fetch the heavy data from the associated HDF5 file. Once read,
 * the data is stored in a vtkUnstructuredGrid in the base class, allowing seamless
 * point location and interpolation.
 */
template <unsigned int dim, typename number>
class ReadXDMF : public ReadVTKGridBase<dim, number>
{
public:
  /**
   * @brief Constructor
   * Initializes the XDMF reader, loads the dataset, verifies the grid topology (e.g.,
   * ensuring cells are hexahedrons or quads), and populates the available scalar and
   * vector field names.
   * @param _ic_file Struct containing the initialization file parameters (filename,
   * etc.).
   * @param _spatial_discretization Struct containing mesh and discretization info.
   */
  ReadXDMF(const InitialConditionFile       &_ic_file,
           const SpatialDiscretization<dim> &_spatial_discretization);

private:
  /**
   * @brief Smart pointer to the VTK XDMF reader. Automatically manages the memory for the
   * reader instance.
   */
  vtkNew<vtkXdmfReader> reader;
};

template <unsigned int dim, typename number>
ReadXDMF<dim, number>::ReadXDMF(const InitialConditionFile       &_ic_file,
                                const SpatialDiscretization<dim> &_spatial_discretization)
  : ReadVTKGridBase<dim, number>(_ic_file, _spatial_discretization)
{
  // Create a reader for the XDMF file and update it to trigger the read process.
  // vtkNew is a smart pointer so we don't need to manage it with delete.
  reader = vtkNew<vtkXdmfReader>();
  reader->SetFileName(this->ic_file.filename.c_str());
  reader->Update();

  // Extract the first block of data as a vtkUnstructuredGrid and safely downcast it.
  // The base class will use this output_grid for all point location and interpolation.
  this->output_grid = vtkUnstructuredGrid::SafeDownCast(reader->GetOutputDataObject(0));

  // Ensure that the file actually contained an unstructured grid
  AssertThrow(this->output_grid != nullptr,
              dealii::ExcMessage(
                "The XDMF file must contain a single unstructured grid"));

  // Verify that the mesh uses a single consistent cell type
  AssertThrow(
    this->output_grid->IsHomogeneous(),
    dealii::ExcMessage(
      "The xdmf grid must have homogeneous cells of type VTK_HEXAHEDRON or VTK_QUAD"));

  // Check that the cell types match the dimensionality of the simulation
  if constexpr (dim == 3)
    {
      AssertThrow(this->output_grid->GetCellType(0) == VTK_HEXAHEDRON,
                  dealii::ExcMessage(
                    "For 3D meshes, the cells must be of type VTK_HEXAHEDRON "));
    }
  else if constexpr (dim == 2)
    {
      AssertThrow(this->output_grid->GetCellType(0) == VTK_QUAD,
                  dealii::ExcMessage(
                    "For 2D meshes, the cells must be of type VTK_QUAD"));
    }
  else
    {
      AssertThrow(false,
                  dealii::ExcMessage("File read-in is not supported for 1D meshes"));
    }

  // Store the total number of points and cells into the base class variables
  this->n_points = this->output_grid->GetNumberOfPoints();
  this->n_cells  = this->output_grid->GetNumberOfCells();

  // Populate array names automatically from the PointData of the unstructured grid.
  // This allows the base class to know which fields are available for querying.
  vtkPointData *point_data = this->output_grid->GetPointData();
  if (point_data)
    {
      for (int i = 0; i < point_data->GetNumberOfArrays(); ++i)
        {
          vtkDataArray *array = point_data->GetArray(i);

          // Categorize as scalar if it has 1 component, otherwise treat it as a vector
          if (array->GetNumberOfComponents() == 1)
            {
              this->scalars_names.push_back(point_data->GetArrayName(i));
            }
          else
            {
              this->vectors_names.push_back(point_data->GetArrayName(i));
            }
        }
    }
}

PRISMS_PF_END_NAMESPACE
