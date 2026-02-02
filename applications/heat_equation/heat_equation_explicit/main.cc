// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"

#include <prismspf/core/dof_handler.h>
#include <prismspf/core/grid_refiner.h>
#include <prismspf/core/invm_handler.h>
#include <prismspf/core/matrix_free_handler.h>
#include <prismspf/core/matrix_free_operator.h>
#include <prismspf/core/multigrid_info.h>
#include <prismspf/core/parse_cmd_options.h>
#include <prismspf/core/pde_problem.h>
#include <prismspf/core/phase_field_tools.h>
#include <prismspf/core/solution_handler.h>
#include <prismspf/core/solver_handler.h>
#include <prismspf/core/variable_attribute_loader.h>
#include <prismspf/core/variable_attributes.h>

#include <prismspf/user_inputs/input_file_reader.h>
#include <prismspf/user_inputs/user_input_parameters.h>

#include <prismspf/solvers/linear_solver_gmg.h>
#include <prismspf/solvers/linear_solver_identity.h>
#include <prismspf/solvers/solver_context.h>

#include <prismspf/utilities/element_volume.h>
#include <prismspf/utilities/integrator.h>

#include <prismspf/config.h>

#ifdef PRISMS_PF_WITH_CALIPER
#  include <caliper/cali-manager.h>
#  include <caliper/cali.h>
#endif

int
main(int argc, char *argv[])
{
  try
    {
      // Initialize MPI
      dealii::Utilities::MPI::MPI_InitFinalize
        mpi_init(argc, argv, dealii::numbers::invalid_unsigned_int);

      // Parse the command line options (if there are any) to get the name of the input
      // file
      prisms::ParseCMDOptions cli_options(argc, argv);
      std::string             parameters_filename = cli_options.get_parameters_filename();

      // Caliper config manager initialization
#ifdef PRISMS_PF_WITH_CALIPER
      cali::ConfigManager mgr;
      mgr.add(cli_options.get_caliper_configuration().c_str());

      // Check for configuration errors
      if (mgr.error())
        {
          std::cerr << "Caliper error: " << mgr.error_msg() << std::endl;
        }

      // Start configured performance measurements, if any
      mgr.start();
#endif

      // Restrict deal.II console printing
      dealii::deallog.depth_console(0);

      // Before fully parsing the parameter file, we need to know how many field
      // variables there are and whether they are scalars or vectors, how many
      // postprocessing variables there are, how many sets of elastic constants
      // there are, and how many user-defined constants there are.
      //
      // This is done with the derived class of `VariableAttributeLoader`,
      // `CustomAttributeLoader`.
      prisms::CustomAttributeLoader attribute_loader;
      attribute_loader.init_variable_attributes();
      std::map<unsigned int, prisms::VariableAttributes> var_attributes =
        attribute_loader.get_var_attributes();

      // Load in parameters
      prisms::InputFileReader input_file_reader(parameters_filename, var_attributes);
      // Make sure those parameters are consistent with your input file
      constexpr int dim    = 1;
      constexpr int degree = 1;

      // Run problem
      prisms::UserInputParameters<dim> user_inputs(
        input_file_reader,
        input_file_reader.get_parameter_handler());

      // Validate dim and degree
      assert(input_file_reader.get_dim() == dim && "Invalid number of dimensions");
      assert(user_inputs.get_spatial_discretization().get_degree() == degree &&
             "Invalid element degree");

      prisms::PhaseFieldTools<dim>                              pf_tools;
      std::shared_ptr<prisms::PDEOperator<dim, degree, double>> pde_operator =
        std::make_shared<prisms::CustomPDE<dim, degree, double>>(user_inputs, pf_tools);
      std::shared_ptr<prisms::PDEOperator<dim, degree, float>> pde_operator_float =
        std::make_shared<prisms::CustomPDE<dim, degree, float>>(user_inputs, pf_tools);

      prisms::PDEProblem<dim, degree, double> problem(user_inputs,
                                                      pf_tools,
                                                      pde_operator,
                                                      pde_operator_float);
      problem.run();

      // Caliper config manager closure
#ifdef PRISMS_PF_WITH_CALIPER
      // Flush output before finalizing MPI
      mgr.flush();
#endif
    }

  catch (std::exception &exc)
    {
      std::cerr << '\n'
                << '\n'
                << "----------------------------------------------------" << '\n';
      std::cerr << "Exception on processing: " << '\n'
                << exc.what() << '\n'
                << "Aborting!" << '\n'
                << "----------------------------------------------------" << '\n';
      return 1;
    }

  catch (...)
    {
      std::cerr << '\n'
                << '\n'
                << "----------------------------------------------------" << '\n';
      std::cerr << "Unknown exception!" << '\n'
                << "Aborting!" << '\n'
                << "----------------------------------------------------" << '\n';
      return 1;
    }

  return 0;
}
