// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <deal.II/base/exceptions.h>

#include <prismspf/core/types.h>

#include <prismspf/field_input/read_binary.h>
#include <prismspf/field_input/read_field_base.h>

#include <prismspf/user_inputs/load_initial_condition_parameters.h>
#include <prismspf/user_inputs/spatial_discretization.h>

#include <prismspf/utilities/utilities.h>

#ifdef PRISMS_PF_WITH_VTK
#  include <prismspf/field_input/read_vtk.h>
#endif
#ifdef PRISMS_PF_WITH_HDF5
// #include <prismspf/field_input/read_hdf5.h>
// #include <prismspf/field_input/read_xdmf.h>
#endif

PRISMS_PF_BEGIN_NAMESPACE

/**
 * @brief Factory function to create appropriate reader based on input file type
 * not a member of ReadFieldBase to avoid redundant template instantiations
 */

enum class Type
{
  ReadUnstructuredVTK,
  ReadBinary,
  ReadHDF5,
  ReadXDMF,
};

template <unsigned int dim, typename number>
std::shared_ptr<ReadFieldBase<dim, number>>
create_reader(const InitialConditionFile       &ic_file,
              const SpatialDiscretization<dim> &spatial_discretization)
{
  switch (ic_file.dataset_format)
    {
      case DataFormatType::VTKUnstructuredGrid:
#ifdef PRISMS_PF_WITH_VTK
        return std::make_shared<ReadUnstructuredVTK<dim, number>>(ic_file,
                                                                  spatial_discretization);
#else
        AssertThrow(false,
                    dealii::ExcMessage(
                      "You are trying to read a VTK file as an input; however, PRISMS-PF "
                      "was not built with VTK. Please reconfig PRISMS-PF with VTK using "
                      "-D PRISMS_PF_WITH_VTK=ON"));
#endif
        // case DataFormatType::XDMF:
        // #if defined(PRISMS_PF_WITH_VTK) && defined(PRISMS_PF_WITH_HDF5)
        //   return std::make_shared<ReadXDMF<dim, number>>(ic_file,
        //   spatial_discretization);
        // #else
        //   AssertThrow(
        //     false,
        //     dealii::ExcMessage(
        //       "You are trying to read a XDMF+H5 file as an input; however, PRISMS-PF "
        //       "was not built with VTK and HDF5. Please reconfig PRISMS-PF with VTK
        //       using "
        //       "-D PRISMS_PF_WITH_VTK=ON and -D PRISMS_PF_WITH_HDF5=ON"));
        // #endif
      case DataFormatType::FlatBinary:
        return std::make_shared<ReadBinary<dim, number>>(ic_file, spatial_discretization);
        // case DataFormatType::HDF5:
        // #ifdef PRISMS_PF_WITH_HDF5
        //   return std::make_shared<ReadHDF5<dim, number>>(ic_file,
        //   spatial_discretization);
        // #else
        //   AssertThrow(
        //     false,
        //     dealii::ExcMessage(
        //       "You are trying to read an HDF5 file as an input; however, PRISMS-PF "
        //       "was not built with HDF5. Please reconfig PRISMS-PF with HDF5 using "
        //       "-D PRISMS_PF_WITH_HDF5=ON"));
        // #endif
      default:
        AssertThrow(false, UnreachableCode());
    }
}

PRISMS_PF_END_NAMESPACE
