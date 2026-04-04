// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <prismspf/field_input/read_vtk_base.h>

#include <vtkUnstructuredGridReader.h>
#include <vtkUnstructuredGridWriter.h>

PRISMS_PF_BEGIN_NAMESPACE

/**
 * @brief Class to read in a VTK file and populate an unstructured grid.
 * This class inherits from ReadVTKGridBase.
 */
template <unsigned int dim, typename number>
class ReadUnstructuredVTK : public ReadVTKGridBase<dim, number>
{
public:
  /**
   * @brief Constructor
   * @param _ic_file Struct containing the initialization file parameters (filename,
   * etc.).
   * @param _spatial_discretization Struct containing mesh and discretization info.
   */
  ReadUnstructuredVTK(const InitialConditionFile       &_ic_file,
                      const SpatialDiscretization<dim> &_spatial_discretization);

  /**
   * @brief Write a vtkUnstructuredGrid to a legacy .vtk file for testing/output.
   */
  static void
  write_file(const vtkSmartPointer<vtkUnstructuredGrid> &grid,
             const std::string                          &filename);

private:
  /**
   * @brief Smart pointer to the VTK reader. Automatically manages the memory for the
   * reader instance.
   */
  vtkNew<vtkUnstructuredGridReader> reader;
};

template <unsigned int dim, typename number>
ReadUnstructuredVTK<dim, number>::ReadUnstructuredVTK(
  const InitialConditionFile       &_ic_file,
  const SpatialDiscretization<dim> &_spatial_discretization)
  : ReadVTKGridBase<dim, number>(_ic_file, _spatial_discretization)
{
  // Create a reader for the vtk file and update it
  // vtkNew is a smart pointer so we don't need to manage it with delete
  reader = vtkNew<vtkUnstructuredGridReader>();
  reader->SetFileName(this->ic_file.filename.c_str());

  // Ensure all arrays are loaded upfront so we don't need to read dynamically later
  reader->ReadAllScalarsOn();
  reader->ReadAllVectorsOn();
  reader->Update();

  // Check that the file is an unstructured grid
  AssertThrow(reader->IsFileUnstructuredGrid(),
              dealii::ExcMessage("The vtk file must be an unstructured grid"));

  // Check that we only have one cell type
  this->output_grid = reader->GetOutput();

  AssertThrow(
    this->output_grid->IsHomogeneous(),
    dealii::ExcMessage(
      "The vtk file must have homogeneous cells of type VTK_HEXAHEDRON or VTK_QUAD"));

  // Check that the cells are hexahedra or quads
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

  // Get the number of points and cells. We first fill the variables in the same type as
  // the VTK return type so we can check for types mismatches with deal.II
  this->n_points = this->output_grid->GetNumberOfPoints();
  this->n_cells  = this->output_grid->GetNumberOfCells();

  // Check that the number of points and cells are not too large
  AssertThrow(n_points_vtk < std::numeric_limits<dealii::types::global_dof_index>::max(),
              dealii::ExcMessage(
                "The number of points being read-in from the vtk file is too large. Try "
                "recompiling deal.II with 64-bit indices."));
  AssertThrow(n_cells_vtk < std::numeric_limits<dealii::types::global_dof_index>::max(),
              dealii::ExcMessage(
                "The number of cells being read-in from the vtk file is too large. Try "
                "recompiling deal.II with 64-bit indices."));

  // Populate names using VTK Reader functions
  unsigned int n_scalars = reader->GetNumberOfScalarsInFile();
  for (unsigned int i = 0; i < n_scalars; ++i)
    this->scalars_names.push_back(reader->GetScalarsNameInFile(static_cast<int>(i)));

  unsigned int n_vectors = reader->GetNumberOfVectorsInFile();
  for (unsigned int i = 0; i < n_vectors; ++i)
    this->vectors_names.push_back(reader->GetVectorsNameInFile(static_cast<int>(i)));
}

template <unsigned int dim, typename number>
inline void
ReadUnstructuredVTK<dim, number>::write_file(
  const vtkSmartPointer<vtkUnstructuredGrid> &grid,
  const std::string                          &filename)
{
  vtkNew<vtkUnstructuredGridWriter> writer;
  writer->SetFileName(filename.c_str());

  // Pass the grid topology and data to the writer
  writer->SetInputData(grid);

  // Execute the disk write
  writer->Write();
}

PRISMS_PF_END_NAMESPACE
