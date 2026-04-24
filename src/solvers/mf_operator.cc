// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <prismspf/core/exceptions.h>
#include <prismspf/core/field_container.h>

#include <prismspf/solvers/mf_operator.h>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
void
MFOperator<dim, degree, number>::compute_operator(BlockVector<number>       &dst,
                                                  const BlockVector<number> &src) const
{
  data->cell_loop(&MFOperator::compute_local_operator, this, dst, src, true);
  if (scale_by_diagonal)
    {
      for (unsigned int block_index = 0; block_index < dst.n_blocks(); block_index++)
        {
          dst.block(block_index).scale(*(scaling_diagonal[block_index]));
        }
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
MFOperator<dim, degree, number>::compute_local_operator(
  const MatrixFree<dim, number>               &_data,
  BlockVector<number>                         &dst,
  const BlockVector<number>                   &src,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  // Construct FEEvaluation objects
  // The reason this is constructed here, rather than as a private member is because
  // compute_local_rhs is called by cell_loop, which multithreads. There would be data
  // races.
  FieldContainer<dim, degree, number> variable_list(field_attributes,
                                                    *solution_indexer,
                                                    relative_level,
                                                    dependency_map,
                                                    solve_block,
                                                    _data);

  // Initialize, evaluate, and submit based on user function.
  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // Initialize, read DOFs, and set evaluation flags for each variable
      variable_list.reinit_and_eval(cell, &src, read_plain);

      // Evaluate the user-defined pde at each quadrature point
      for (unsigned int quad = 0; quad < variable_list.get_n_q_points(); ++quad)
        {
          variable_list.set_q_point(quad);
          // Evaluate the function pointer (the user-defined pde)
          try
            {
              (pde_operator->*pde_op)(variable_list, *sim_timer, solve_block.id);
            }
          catch (const std::exception &exc)
            {
              AssertThrow(false,
                          dealii::ExcMessage(
                            "Exception thrown in equations during solve block " +
                            std::to_string(solve_block.id) +
                            "! Original error: " + std::string(exc.what())));
            }
          catch (...)
            {
              AssertThrow(false,
                          dealii::ExcMessage(
                            "Unknown exception thrown in equations during solve block " +
                            std::to_string(solve_block.id) + "!"));
            }
        }

      // Integrate and add to global vector dst

      variable_list.integrate_and_distribute(&dst);
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
MFOperator<dim, degree, number>::compute_diagonal(BlockVector<number> &diagonal) const
{
  AssertThrow(data != nullptr, dealii::ExcNotInitialized());

  // Initialize the dummy source vector block-by-block.
  BlockVector<number> dummy;
  dummy.reinit(diagonal.n_blocks());

  auto field_it_dummy = solve_block.field_indices.begin();
  for (unsigned int j = 0; j < diagonal.n_blocks(); ++j, ++field_it_dummy)
    {
      // Pass the underlying distributed::Vector and its specific field index
      data->initialize_dof_vector(dummy.block(j), *field_it_dummy);
    }
  dummy.collect_sizes();
  dummy = 0.0;

  // Execute cell loop.
  data->cell_loop(&MFOperator<dim, degree, number>::compute_local_diagonal,
                  this,
                  diagonal,
                  dummy,
                  true);

  // Apply diagonal scaling
  if (scale_by_diagonal)
    {
      for (unsigned int block_index = 0; block_index < diagonal.n_blocks(); ++block_index)
        {
          diagonal.block(block_index).scale(*(scaling_diagonal[block_index]));
        }
    }

  // Enforce constraints (e.g., Dirichlet boundaries) to be 1.0 on the diagonal
  auto field_it = solve_block.field_indices.begin();
  for (unsigned int j = 0; j < diagonal.n_blocks(); ++j, ++field_it)
    {
      unsigned int field_index = *field_it;

      const auto &constrained_dofs = data->get_constrained_dofs(field_index);

      for (const auto constrained_dof : constrained_dofs)
        {
          diagonal.block(j).local_element(constrained_dof) = 1.0;
        }
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
MFOperator<dim, degree, number>::compute_local_diagonal(
  const MatrixFree<dim, number>               &_data,
  BlockVector<number>                         &diagonal,
  const BlockVector<number>                   &dummy,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  FieldContainer<dim, degree, number> variable_list(field_attributes,
                                                    *solution_indexer,
                                                    relative_level,
                                                    dependency_map,
                                                    solve_block,
                                                    _data);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // Reinit and zero out the cell
      variable_list.reinit_and_eval(cell, &dummy, true);

      for (unsigned int field_index : solve_block.field_indices)
        {
          if (field_attributes[field_index].field_type == TensorRank::Scalar)
            {
              compute_field_diagonal<TensorRank::Scalar>(variable_list, field_index);
            }
          else if (field_attributes[field_index].field_type == TensorRank::Vector)
            {
              compute_field_diagonal<TensorRank::Vector>(variable_list, field_index);
            }
        }

      // Distribute to the global matrix
      variable_list.distribute(&diagonal);
    }
}

template <unsigned int dim, unsigned int degree, typename number>
template <TensorRank Rank>
void
MFOperator<dim, degree, number>::compute_field_diagonal(
  FieldContainer<dim, degree, number> &variable_list,
  unsigned int                         field_index) const
{
  using ScalarArray = dealii::VectorizedArray<number>;

  if constexpr (Rank == TensorRank::Scalar)
    {
      const unsigned int n_dofs =
        variable_list.template get_dofs_per_cell<Rank>(field_index);
      dealii::AlignedVector<ScalarArray> cell_diagonal(n_dofs);

      for (unsigned int i = 0; i < n_dofs; ++i)
        {
          variable_list.template clear_field_dst<Rank>(field_index);
          for (unsigned int j = 0; j < n_dofs; ++j)
            {
              ScalarArray val = (i == j) ? ScalarArray(1.0) : ScalarArray(0.0);
              variable_list.template set_dof_value<Rank>(field_index, val, j);
            }

          variable_list.template evaluate_field<Rank>(field_index);

          for (unsigned int quad = 0; quad < variable_list.get_n_q_points(); ++quad)
            {
              variable_list.set_q_point(quad);
              (pde_operator->*pde_op)(variable_list, *sim_timer, solve_block.id);
            }

          variable_list.template integrate_field<Rank>(field_index);

          cell_diagonal[i] = variable_list.template get_dof_value<Rank>(field_index, i);
        }

      for (unsigned int i = 0; i < n_dofs; ++i)
        {
          variable_list.template set_dof_value<Rank>(field_index, cell_diagonal[i], i);
        }
    }
  else if constexpr (Rank == TensorRank::Vector)
    {
      using TensorArray = dealii::Tensor<1, dim, ScalarArray>;

      const unsigned int n_nodes =
        variable_list.template get_dofs_per_cell<Rank>(field_index);
      dealii::AlignedVector<TensorArray> cell_diagonal(n_nodes);

      // Initialize all tensors to zero
      for (unsigned int i = 0; i < n_nodes; ++i)
        cell_diagonal[i] = TensorArray();

      // For vector fields, we must probe each node (i) AND each dimension/component (c)
      for (unsigned int i = 0; i < n_nodes; ++i)
        {
          for (unsigned int c = 0; c < dim; ++c)
            {
              variable_list.template clear_field_dst<Rank>(field_index);
              // Inject unit vector at node i, component c
              for (unsigned int j = 0; j < n_nodes; ++j)
                {
                  if constexpr (dim == 1)
                    {
                      // 1D
                      ScalarArray val(0.0);
                      if (i == j)
                        val = 1.0;
                      variable_list.template set_dof_value<Rank>(field_index, val, j);
                    }
                  else
                    {
                      // 2D/3D
                      TensorArray val; // Defaults to zeroes
                      if (i == j)
                        val[c] = 1.0;
                      variable_list.template set_dof_value<Rank>(field_index, val, j);
                    }
                }

              // Evaluate
              variable_list.template evaluate_field<Rank>(field_index);

              // Apply PDE
              for (unsigned int quad = 0; quad < variable_list.get_n_q_points(); ++quad)
                {
                  variable_list.set_q_point(quad);
                  try
                    {
                      (pde_operator->*pde_op)(variable_list, *sim_timer, solve_block.id);
                    }
                  catch (...)
                    {
                      AssertThrow(false,
                                  dealii::ExcMessage(
                                    "Exception in equations during diagonal probing."));
                    }
                }

              // Integrate
              variable_list.template integrate_field<Rank>(field_index);

              // Extract the resulting component and build the diagonal tensor
              auto result = variable_list.template get_dof_value<Rank>(field_index, i);
              if constexpr (dim == 1)
                {
                  cell_diagonal[i][c] = result; // result is a ScalarArray
                }
              else
                {
                  cell_diagonal[i][c] = result[c]; // result is a TensorArray
                }
            }
        }

      // Store back the fully assembled diagonal tensor for distribution
      for (unsigned int i = 0; i < n_nodes; ++i)
        {
          if constexpr (dim == 1)
            {
              variable_list.template set_dof_value<Rank>(field_index,
                                                         cell_diagonal[i][0],
                                                         i);
            }
          else
            {
              variable_list.template set_dof_value<Rank>(field_index,
                                                         cell_diagonal[i],
                                                         i);
            }
        }
    }
}

template <unsigned int dim, unsigned int degree, typename number>
dealii::types::global_dof_index
MFOperator<dim, degree, number>::m() const
{
  // Assert(data.get() != nullptr, dealii::ExcNotInitialized());
  //
  // const unsigned int total_size =
  //  std::accumulate(selected_fields.begin(),
  //                  selected_fields.end(),
  //                  0U,
  //                  [this](unsigned int sum, unsigned int field)
  //                  {
  //                    return sum + data->get_vector_partitioner(field)->size();
  //                  });
  // return total_size;
  AssertThrow(false, FeatureNotImplemented("m()"));
  return 0;
}

template <unsigned int dim, unsigned int degree, typename number>
number
MFOperator<dim, degree, number>::el([[maybe_unused]] const unsigned int &row,
                                    [[maybe_unused]] const unsigned int &col) const
{
  AssertThrow(false, FeatureNotImplemented("el()"));
  return 0.0;
}

template <unsigned int dim, unsigned int degree, typename number>
void
MFOperator<dim, degree, number>::clear()
{
  data = nullptr;
  diagonal_entries.reset();
  inverse_diagonal_entries.reset();
}

// template <unsigned int dim, unsigned int degree, typename number>
// void
// MFOperator<dim, degree, number>::set_constrained_entries_to_one(SolutionVector<number>
// &dst) const
// {
//   for (unsigned int j = 0; j < dealii::MatrixFreeOperators::BlockHelper::n_blocks(dst);
//   ++j)
//     {
//       const std::vector<unsigned int> &constrained_dofs =
//         data->get_constrained_dofs(selected_fields[j]);
//       for (const auto constrained_dof : constrained_dofs)
//         {
//           dealii::MatrixFreeOperators::BlockHelper::subblock(dst, j).local_element(
//             constrained_dof) = 1.0;
//         }
//     }
// }

template <unsigned int dim, unsigned int degree, typename number>
const MatrixFree<dim, number> *
MFOperator<dim, degree, number>::get_matrix_free() const
{
  return data;
}

template <unsigned int dim, unsigned int degree, typename number>
const std::shared_ptr<dealii::DiagonalMatrix<SolutionVector<number>>> &
MFOperator<dim, degree, number>::get_matrix_diagonal_inverse() const
{
  Assert(inverse_diagonal_entries.get() != nullptr && inverse_diagonal_entries->m() > 0,
         dealii::ExcNotInitialized());
  return inverse_diagonal_entries;
}

template <unsigned int dim, unsigned int degree, typename number>
void
MFOperator<dim, degree, number>::vmult(BlockVector<number>       &dst,
                                       const BlockVector<number> &src) const
{
  compute_operator(dst, src);
}

// NOLINTBEGIN(readability-identifier-naming)

template <unsigned int dim, unsigned int degree, typename number>
void
MFOperator<dim, degree, number>::Tvmult(BlockVector<number>       &dst,
                                        const BlockVector<number> &src) const
{
  vmult(dst, src);
}

// NOLINTEND(readability-identifier-naming)

#include "solvers/mf_operator.inst"

PRISMS_PF_END_NAMESPACE
