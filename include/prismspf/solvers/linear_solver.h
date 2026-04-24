// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <prismspf/core/conditional_ostreams.h>
#include <prismspf/core/group_solution_handler.h>
#include <prismspf/core/invm_manager.h>
#include <prismspf/core/timer.h>
#include <prismspf/core/type_enums.h>
#include <prismspf/core/types.h>

#include <prismspf/solvers/mf_operator.h>
#include <prismspf/solvers/solver_base.h>

#include <prismspf/user_inputs/linear_solve_parameters.h>

#include <prismspf/config.h>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
class SolveContext;

/**
 * @brief This class handles the explicit solves of all explicit fields
 */
template <unsigned int dim, unsigned int degree, typename number>
class LinearSolver : public SolverBase<dim, degree, number>
{
protected:
  using SolverBase<dim, degree, number>::solutions;
  using SolverBase<dim, degree, number>::solve_context;
  using SolverBase<dim, degree, number>::solve_block;

public:
  /**
   * @brief Constructor.
   */
  LinearSolver(SolveBlock                               _solve_block,
               const SolveContext<dim, degree, number> &_solve_context)
    : SolverBase<dim, degree, number>(_solve_block, _solve_context)
    , lin_params(
        solve_context->get_user_inputs().linear_solve_parameters.linear_solvers.at(
          solve_block.id))
  {}

  /**
   * @brief Initialize the solver.
   */
  void
  init(const std::list<DependencyMap> &all_dependeny_sets) override
  {
    SolverBase<dim, degree, number>::init(all_dependeny_sets);
    unsigned int num_levels = solve_context->get_dof_manager().get_dof_handlers().size();
    rhs_vector.resize(num_levels);
    for (unsigned int relative_level = 0; relative_level < num_levels; ++relative_level)
      {
        rhs_vector[relative_level].reinit(
          solutions.get_solution_full_vector(relative_level));
      }
    // Initialize rhs_operators
    rhs_operators.reserve(num_levels);
    for (unsigned int relative_level = 0; relative_level < num_levels; ++relative_level)
      {
        rhs_operators.emplace_back(solve_context->get_pde_operator(),
                                   &PDEOperatorBase<dim, degree, number>::compute_rhs,
                                   solve_context->get_field_attributes(),
                                   solve_context->get_solution_indexer(),
                                   relative_level,
                                   solve_block.dependencies_rhs,
                                   solve_context->get_simulation_timer());
        rhs_operators[relative_level].initialize(solutions);
        rhs_operators[relative_level].set_scaling_diagonal(
          lin_params.tolerance_type != AbsoluteResidual,
          solve_context->get_invm_manager().get_invm_sqrt(
            solve_context->get_field_attributes(),
            solve_block.field_indices,
            relative_level));
      }
    // Initialize lhs_operators
    lhs_operators.reserve(num_levels);
    for (unsigned int relative_level = 0; relative_level < num_levels; ++relative_level)
      {
        lhs_operators.emplace_back(solve_context->get_pde_operator(),
                                   &PDEOperatorBase<dim, degree, number>::compute_lhs,
                                   solve_context->get_field_attributes(),
                                   solve_context->get_solution_indexer(),
                                   relative_level,
                                   solve_block.dependencies_lhs,
                                   solve_context->get_simulation_timer());
        lhs_operators[relative_level].initialize(solutions);
        lhs_operators[relative_level].set_scaling_diagonal(
          lin_params.tolerance_type != AbsoluteResidual,
          solve_context->get_invm_manager().get_invm_sqrt(
            solve_context->get_field_attributes(),
            solve_block.field_indices,
            relative_level));
      }
    linear_solver_control.set_max_steps(lin_params.max_iterations);
    linear_solver_control.set_tolerance(lin_params.tolerance * normalization_value());
    inhomogeneous_values.reinit(solutions.get_solution_full_vector(0));
    preconditioner_diagonal.reinit(solutions.get_solution_full_vector(0));
    jacobi_preconditioner.reinit(preconditioner_diagonal);
    solutions.apply_constraints(inhomogeneous_values, 0);
    inhomogeneous_rhs.reinit(solutions.get_solution_full_vector(0));
  }

  /**
   * @brief Reinitialize the solver.
   */
  void
  reinit() override
  {
    SolverBase<dim, degree, number>::reinit();
    const unsigned int num_levels = rhs_vector.size();
    for (unsigned int relative_level = 0; relative_level < num_levels; ++relative_level)
      {
        rhs_vector[relative_level].reinit(
          solutions.get_solution_full_vector(relative_level));
      }
    inhomogeneous_values.reinit(solutions.get_solution_full_vector(0));
    preconditioner_diagonal.reinit(solutions.get_solution_full_vector(0));
    jacobi_preconditioner.reinit(preconditioner_diagonal);
    solutions.apply_constraints(inhomogeneous_values, 0);
    inhomogeneous_rhs.reinit(solutions.get_solution_full_vector(0));
  }

  /**
   * @brief Solve for a single update step.
   */
  void
  solve_level(unsigned int relative_level) override
  {
    // Zero out the ghosts
    Timer::start_section("Zero ghosts");
    solutions.zero_out_ghosts(relative_level);
    Timer::end_section("Zero ghosts");

    // Set up rhs vector
    rhs_operators[relative_level].compute_operator(rhs_vector[relative_level]);
    if (relative_level == 0)
      {
        // Note 1. Use the previous result of the linear solve without nonzero dirichlet
        // as the initial guess in the next increment. See Note 2. `inhomogeneous_rhs` is
        // not actually what it is being used as here, we just don't want to allocate a
        // whole new vector for this purpose
        solutions.get_solution_full_vector(0).swap(inhomogeneous_rhs);
        // Get the homogeneous rhs
        lhs_operators[0].read_plain = true;
        lhs_operators[0].compute_operator(inhomogeneous_rhs, inhomogeneous_values);
        lhs_operators[0].read_plain = false;
        rhs_vector[0] -= inhomogeneous_rhs;
      }
    // Linear solve
    do_linear_solve(rhs_vector[relative_level],
                    lhs_operators[relative_level],
                    solutions.get_solution_full_vector(relative_level));

    if (relative_level == 0)
      {
        // Note 2. Make a copy of the solution to use as the initial guess in the next
        // increment. See Note 1. `inhomogeneous_rhs` is not actually what it is being
        // used as here, we just don't want to allocate a whole new vector for this
        // purpose
        inhomogeneous_rhs = solutions.get_solution_full_vector(0);
        // Add back in nonzero dirichlet conditions
        solutions.get_solution_full_vector(0) += inhomogeneous_values;
      }
    // Apply constraints
    solutions.apply_constraints(relative_level);

    // Update the ghosts
    Timer::start_section("Update ghosts");
    solutions.update_ghosts(relative_level);
    Timer::end_section("Update ghosts");
  }

  int
  do_linear_solve(BlockVector<number>             &b_vector,
                  MFOperator<dim, degree, number> &lhs_operator,
                  BlockVector<number>             &x_vector)
  {
    // Linear solve
    try
      {
        dealii::SolverCG<BlockVector<number>> cg_solver(linear_solver_control);

        // Switch based on the user-selected preconditioner
        switch (lin_params.preconditioner)
          {
            case PreconditionerType::None:
            case PreconditionerType::GMG: // not implemented, fallback to None
              {
                Timer::start_section("Linear solve: CG solve (Identity)");
                try
                  {
                    cg_solver.solve(lhs_operator,
                                    x_vector,
                                    b_vector,
                                    dealii::PreconditionIdentity());
                  }
                catch (...)
                  {
                    Timer::end_section("Linear solve: CG solve (Identity)");
                    ConditionalOStreams::pout_base()
                      << "CG stopped at iter " << linear_solver_control.last_step()
                      << " with residual " << linear_solver_control.last_value()
                      << "\n\n";
                    throw;
                  }

                Timer::end_section("Linear solve: CG solve (Identity)");

                const unsigned int iters     = linear_solver_control.last_step();
                const double       final_res = linear_solver_control.last_value();

                ConditionalOStreams::pout_summary()
                  << "CG iters: " << iters << " final residual: " << final_res << '\n';

                break;
              }
            case PreconditionerType::Jacobi:
              {
                /*
                // DEBUG: test symmetry
                {
                  BlockVector<number> u, v, Au, Av;
                  u.reinit(x_vector);
                  v.reinit(x_vector);
                  Au.reinit(x_vector);
                  Av.reinit(x_vector);

                  // Fill u and v with something nontrivial on owned entries
                  for (unsigned int b = 0; b < u.n_blocks(); ++b)
                    {
                      auto &ub = u.block(b);
                      auto &vb = v.block(b);

                      for (auto i : ub.locally_owned_elements())
                        ub[i] = 0.37; // simple deterministic values are fine
                      for (auto i : vb.locally_owned_elements())
                        vb[i] = 0.11;
                    }

                  // IMPORTANT: if your vmult reads ghost values from the source vector,
                  // you must update ghosts before calling vmult.
                  u.update_ghost_values();
                  v.update_ghost_values();

                  lhs_operator.vmult(Av, v);
                  lhs_operator.vmult(Au, u);

                  // If your vmult writes to vectors with ghosts, you might need compress
                  // here, but typically vmult writes only owned entries.
                  // Au.compress(VectorOperation::add); Av.compress(VectorOperation::add);

                  const double utAv = u * Av;
                  const double vtAu = v * Au;

                  const double denom    = std::max({1.0, std::abs(utAv), std::abs(vtAu)});
                  const double rel_diff = std::abs(utAv - vtAu) / denom;

                  ConditionalOStreams::pout_summary()
                    << "Symmetry check: u^T A v = " << utAv << ", v^T A u = " << vtAu
                    << ", rel diff = " << rel_diff << "\n\n";

                  u.zero_out_ghost_values();
                  v.zero_out_ghost_values();
                }
                // DEBUG: Check definiteness
                {
                  BlockVector<number> u, Au;
                  u.reinit(x_vector);
                  Au.reinit(x_vector);

                  // fill u with something nontrivial
                  for (unsigned int b = 0; b < u.n_blocks(); ++b)
                    for (auto i : u.block(b).locally_owned_elements())
                      u.block(b)[i] = 0.37;

                  u.update_ghost_values();
                  lhs_operator.vmult(Au, u);

                  const double uAu = u * Au;
                  const double uu  = u * u;

                  ConditionalOStreams::pout_summary()
                    << "Definiteness check: uAu=" << uAu << " , uu=" << uu
                    << " , Rayleigh=" << (uAu / uu) << "\n\n";
                }
                */
                // Compute the diagonal of the matrix-free operator
                Timer::start_section("Linear solve: compute diagonal");
                try
                  {
                    lhs_operator.compute_diagonal(preconditioner_diagonal);
                  }
                catch (...)
                  {
                    Timer::end_section("Linear solve: compute diagonal");
                    throw;
                  }
                Timer::end_section("Linear solve: compute diagonal");

                /*
                // DEBUG: check nan and nonpositive diagonal
                {
                  for (unsigned int b = 0; b < preconditioner_diagonal.n_blocks(); ++b)
                    {
                      const auto &v     = preconditioner_diagonal.block(b);
                      const auto &owned = v.locally_owned_elements();

                      double      local_min    = std::numeric_limits<double>::infinity();
                      double      local_max    = -std::numeric_limits<double>::infinity();
                      std::size_t local_nonpos = 0;

                      for (auto i : owned)
                        {
                          const double d = v[i];
                          local_min      = std::min(local_min, d);
                          local_max      = std::max(local_max, d);
                          if (!(d > 0.0) || !std::isfinite(d))
                            local_nonpos++;
                        }

                      // (Optionally MPI-reduce)
                      ConditionalOStreams::pout_summary()
                        << "diag block " << b << " min=" << local_min
                        << " max=" << local_max << " nonpos_or_nan=" << local_nonpos
                        << "\n\n";
                    }
                }
                */

                Timer::start_section("Linear solve: Jacobi reinit");
                jacobi_preconditioner.reinit(preconditioner_diagonal);
                Timer::end_section("Linear solve: Jacobi reinit");

                Timer::start_section("Linear solve: CG solve (Jacobi)");
                try
                  {
                    cg_solver.solve(lhs_operator,
                                    x_vector,
                                    b_vector,
                                    jacobi_preconditioner);
                  }
                catch (...)
                  {
                    Timer::end_section("Linear solve: CG solve (Jacobi)");

                    ConditionalOStreams::pout_base()
                      << "CG stopped at iter " << linear_solver_control.last_step()
                      << " with residual " << linear_solver_control.last_value()
                      << "\n\n";

                    throw;
                  }
                Timer::end_section("Linear solve: CG solve (Jacobi)");

                ConditionalOStreams::pout_summary()
                  << "CG iters: " << linear_solver_control.last_step()
                  << " final residual: " << linear_solver_control.last_value() << '\n';

                break;
              }
            case PreconditionerType::Chebyshev:
              {
                ConditionalOStreams::pout_summary()
                  << "Entered Chebyshev preconditioner branch\n";
                // Compute diagonal values
                ConditionalOStreams::pout_summary() << "Cheb: compute_diagonal begin\n";
                lhs_operator.compute_diagonal(preconditioner_diagonal);
                ConditionalOStreams::pout_summary() << "Cheb: compute_diagonal done\n";

                // Setup Chebyshev
                using ChebyshevPreconditioner = dealii::PreconditionChebyshev<
                  MFOperator<dim, degree, number>,
                  BlockVector<number>,
                  dealii::DiagonalMatrix<BlockVector<number>>>;
                typename ChebyshevPreconditioner::AdditionalData chebyshev_data;
                chebyshev_data.degree              = 30;
                chebyshev_data.smoothing_range     = 20.0;
                chebyshev_data.eig_cg_n_iterations = 20;
                chebyshev_data.eig_cg_residual     = 1e-4;
                chebyshev_data.eigenvalue_algorithm =
                  ChebyshevPreconditioner::AdditionalData::EigenvalueAlgorithm::lanczos;
                // chebyshev_data.polynomial_type =
                // Cheb::AdditionalData::PolynomialType::fourth_kind;

                chebyshev_data.preconditioner =
                  std::make_shared<dealii::DiagonalMatrix<BlockVector<number>>>();
                chebyshev_data.preconditioner->reinit(preconditioner_diagonal);

                ChebyshevPreconditioner chebyshev_preconditioner;
                ConditionalOStreams::pout_summary() << "Cheb: initialize begin\n";
                try
                  {
                    chebyshev_preconditioner.initialize(lhs_operator, chebyshev_data);
                  }
                catch (const std::exception &e)
                  {
                    ConditionalOStreams::pout_base()
                      << "Chebyshev initialize() threw: " << e.what() << "\n";
                    throw;
                  }
                ConditionalOStreams::pout_summary() << "Cheb: initialize done\n";

                ConditionalOStreams::pout_summary() << "Cheb: solve begin\n";
                Timer::start_section("Linear solve: CG solve (Chebyshev)");
                try
                  {
                    cg_solver.solve(lhs_operator,
                                    x_vector,
                                    b_vector,
                                    chebyshev_preconditioner);
                    ConditionalOStreams::pout_summary() << "Cheb: solve done\n";
                  }
                catch (...)
                  {
                    Timer::end_section("Linear solve: CG solve (Chebyshev)");

                    ConditionalOStreams::pout_base()
                      << "CG stopped at iter " << linear_solver_control.last_step()
                      << " with residual " << linear_solver_control.last_value()
                      << "\n\n";

                    throw;
                  }
                Timer::end_section("Linear solve: CG solve (Chebyshev)");

                ConditionalOStreams::pout_summary()
                  << "CG iters: " << linear_solver_control.last_step()
                  << " final residual: " << linear_solver_control.last_value() << '\n';

                break;
              }
          }

        if (solve_context->get_user_inputs().output_parameters.should_output(
              solve_context->get_simulation_timer().get_increment()))
          {
            ConditionalOStreams::pout_summary()
              << " Linear solve final residual : "
              << linear_solver_control.last_value() / normalization_value()
              << " Linear steps: " << linear_solver_control.last_step() << "\n"
              << std::flush;
          }
      }
    catch (...) // TODO: more specific catch
      {
        ConditionalOStreams::pout_base()
          << "[Increment " << solve_context->get_simulation_timer().get_increment()
          << "] "
          << "Warning: linear solver did not converge as per set tolerances before "
          << lin_params.max_iterations << " iterations.\n";
      }
    return linear_solver_control.last_step();
  }

protected:
  /**
   * @brief Matrix free operators for each level
   */
  std::vector<MFOperator<dim, degree, number>> rhs_operators;

  std::vector<MFOperator<dim, degree, number>> lhs_operators;
  std::vector<BlockVector<number>>             rhs_vector;

  double
  normalization_value()
  {
    SolverToleranceType type  = lin_params.tolerance_type;
    double              value = 1.0;
    if (type == RMSEPerField || type == RMSETotal)
      {
        value *= std::sqrt(solve_context->get_triangulation_manager().get_volume());
      }
    if (type == RMSEPerField || type == IntegratedPerField)
      {
        value *= std::sqrt(double(solve_block.field_indices.size()));
      }
    return value;
  }

private:
  /**
   * @brief Linear solver parameters
   */
  LinearSolverParameters lin_params;

  /**
   * @brief Solver control. Contains max iterations and tolerance.
   */
  dealii::SolverControl linear_solver_control;

  /**
   * @brief Vector containing only the inhomogeneous constraints (namely, non-zero
   * Dirichlet values)
   */
  BlockVector<number> inhomogeneous_values;

  /**
   * @brief Result of the linear operator applied to the inhomogeneous values.
   */
  BlockVector<number> inhomogeneous_rhs;

  /**
   * @brief Diagonal vector used for Jacobi and Chebyshev preconditioners
   */
  BlockVector<number> preconditioner_diagonal;

  /**
   * @brief Diagonal matrix preconditioner
   */
  dealii::DiagonalMatrix<BlockVector<number>> jacobi_preconditioner;
};

PRISMS_PF_END_NAMESPACE
