// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"

#include <prismspf/core/type_enums.h>
#include <prismspf/core/variable_attribute_loader.h>
#include <prismspf/core/variable_container.h>

#include <prismspf/config.h>

#include <cmath>
#include <numbers>

PRISMS_PF_BEGIN_NAMESPACE

void
CustomAttributeLoader::load_variable_attributes()
{
  set_variable_name(0, "u");
  set_variable_type(0, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(0, ImplicitTimeDependent);
  set_dependencies_value_term_rhs(0, "u, old_1(u), gamma");
  set_dependencies_gradient_term_rhs(0, "grad(u)");
  set_dependencies_value_term_lhs(0, "change(u)");
  set_dependencies_gradient_term_lhs(0, "grad(change(u))");

  set_variable_name(3, "gamma"); // added to trigger sequantial_co_nonlinear_solver
  set_variable_type(3, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(3, Auxiliary);
  set_dependencies_value_term_rhs(3, "gamma, u");
  set_dependencies_gradient_term_rhs(3, "grad(u)");

  set_variable_name(1, "u_analytical");
  set_variable_type(1, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(1, ExplicitTimeDependent);
  set_is_postprocessed_field(1, true);
  set_dependencies_value_term_rhs(1, "u"); // this is needed even it does not depend on u
  set_dependencies_gradient_term_rhs(1, "");

  set_variable_name(2, "u_error");
  set_variable_type(2, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(2, ExplicitTimeDependent);
  set_is_postprocessed_field(2, true);
  set_dependencies_value_term_rhs(2, "u");
  set_dependencies_gradient_term_rhs(2, "");
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_explicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block) const
{}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_nonexplicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block,
  [[maybe_unused]] Types::Index                           index) const
{
  if (index == 0)
    {
      ScalarValue u     = variable_list.template get_value<ScalarValue>(0);
      ScalarValue old_u = variable_list.template get_value<ScalarValue>(0, OldOne);
      ScalarGrad  ux    = variable_list.template get_gradient<ScalarGrad>(0);

      ScalarValue eq_u  = old_u - u;
      ScalarGrad  eqx_u = -get_timestep() * DT * ux;

      variable_list.set_value_term(0, eq_u);
      variable_list.set_gradient_term(0, eqx_u);
    }
  if (index == 3)
    {
      const number pi    = std::numbers::pi_v<number>;
      ScalarValue  gamma = variable_list.template get_value<ScalarValue>(3);
      ScalarValue  u     = variable_list.template get_value<ScalarValue>(0);
      ScalarGrad   ux    = variable_list.template get_gradient<ScalarGrad>(0);
      variable_list.set_gradient_term(3, -1.0 * ux / (pi * pi));
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_nonexplicit_lhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block,
  [[maybe_unused]] Types::Index                           index) const
{
  if (index == 0)
    {
      ScalarValue change_u  = variable_list.template get_value<ScalarValue>(0, Change);
      ScalarGrad  change_ux = variable_list.template get_gradient<ScalarGrad>(0, Change);

      ScalarValue eq_change_u  = change_u;
      ScalarGrad  eqx_change_u = get_timestep() * DT * change_ux;

      variable_list.set_value_term(0, eq_change_u, Change);
      variable_list.set_gradient_term(0, eqx_change_u, Change);
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_postprocess_explicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block) const
{
  ScalarValue  u = variable_list.template get_value<ScalarValue>(0);
  const number time =
    static_cast<number>(get_user_inputs().get_temporal_discretization().get_time());
  auto bctype = get_user_inputs()
                  .get_boundary_parameters()
                  .get_boundary_condition_list()
                  .at(0) // index
                  .at(0) // component
                  .get_boundary_condition_map()
                  .at(0); // boundary_id???, how to specify boundary_id in input file?
  const number pi = std::numbers::pi_v<number>;

  ScalarValue u_analytical = std::exp(-1. * dim * DT * pi * pi * time);
  if (bctype == BoundaryCondition::Type::Dirichlet)
    for (unsigned int i = 0; i < dim; ++i)
      u_analytical *= std::sin(pi * q_point_loc[i]);
  else if (bctype == BoundaryCondition::Type::Natural)
    for (unsigned int i = 0; i < dim; ++i)
      u_analytical *= std::cos(pi * q_point_loc[i]);
  variable_list.set_value_term(1, u_analytical);
  variable_list.set_value_term(2, std::abs(u - u_analytical));
}

#include "custom_pde.inst"

PRISMS_PF_END_NAMESPACE
