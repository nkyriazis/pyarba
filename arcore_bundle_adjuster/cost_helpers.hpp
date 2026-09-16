#pragma once

#include <ceres/autodiff_cost_function.h>
#include <type_traits>

namespace arba
{

/**
 * This is a helper to define cost functions for ceres with more (not absolute)
 * type safety.
 */
namespace decl
{

/**
 * The detail namespace implements a simple syntactic sugar mechanism to declare
 * the type mapping needs for ceres costs.
 *
 * The basic idea is that one uses e.g.
 *
 * ```cpp
 * decltype(vec3 = vec3, vec3)
 * ```
 *
 * To declare that the cost maps 2 3D vectors to a 3D residual.
 */
namespace detail
{

/**
 * vec and quat_ are declaration elements. On their own they do nothing. Once
 * they compound a declaration through '=' and ',' they build up a meaningful
 * expression.
 */

template <int N>
struct vec;

struct quat_;

// simple traits

template <typename T>
struct is_decl_elem : std::false_type
{
};

template <int N>
struct is_decl_elem<vec<N>> : std::true_type
{
};

template <>
struct is_decl_elem<quat_> : std::true_type
{
};

// A declaration is minimaly type1 = type2 (1-1)
// It accepts expansion through ", added type"
template <typename Residual, typename... Params>
struct declaration
{

    template <typename T>
    auto operator,(const T&) const
    {
        return declaration<Residual, Params..., T>();
    }
};

/**
 * The only way to start a declaration is through a vec or quat_ and assignment.
 */

template <int N>
struct vec
{
    static constexpr int dim = N;

    template <typename T>
    auto operator=(const T& t) const
    {
        return start_declaration(*this, t);
    }

    template <typename T>
    static auto map(T* ptr)
    {
        return Eigen::Map<
          std::conditional_t<std::is_const_v<T>,
                             const Eigen::Matrix<std::decay_t<T>, N, 1>,
                             Eigen::Matrix<T, N, 1>>>(ptr);
    }
};

struct quat_
{
    static constexpr int dim = 4;

    template <typename T>
    auto operator=(const T& t) const
    {
        return start_declaration(*this, t);
    }

    template <typename T>
    static auto map(T* ptr)
    {
        return Eigen::Map<
          std::conditional_t<std::is_const_v<T>,
                             const Eigen::Quaternion<std::decay_t<T>>,
                             Eigen::Quaternion<T>>>(ptr);
    }
};

template <typename LHS, typename RHS>
auto start_declaration(const LHS& lhs, const RHS& rhs)
  -> std::enable_if_t<is_decl_elem<LHS>::value && is_decl_elem<RHS>::value,
                      declaration<LHS, RHS>>
{
    return declaration<LHS, RHS>();
}
} // namespace detail

/**
 * Standard elems. Quick access.
 */
namespace elems
{
static constexpr detail::vec<1> vec1;
static constexpr detail::vec<2> vec2;
static constexpr detail::vec<3> vec3;
static constexpr detail::vec<4> vec4;
static constexpr detail::vec<5> vec5;
static constexpr detail::quat_ quat;
} // namespace elems

template <typename... T>
struct CostBase;

/**
 * The cost function helper. CRP impl.
 *
 * This achieves two things:
 * - Automatically map pointers to Eigen types
 * - Easy access to type safe mapping to AutoDiffCostFunction
 */
template <typename Derived, typename Residual, typename... Params>
struct CostBase<Derived, detail::declaration<Residual, Params...>>
{
    using auto_diff_type =
      ceres::AutoDiffCostFunction<Derived, Residual::dim, Params::dim...>;

    template <typename... Args>
    bool operator()(Args... args) const
    {
        auto convert_tags = std::make_tuple(Params{}..., Residual{});
        auto params       = std::make_tuple(args...);

        // map the args
        auto new_args = convert_impl(
          convert_tags, params, std::index_sequence_for<Args...>{});

        // expand the args
        compute_impl(new_args, std::index_sequence_for<Args...>{});

        return true;
    }

    template <typename... Args>
    static auto create(Args&&... args)
    {
        return (new Derived(std::forward<Args>(args)...))->to_cost();
    }

    static constexpr auto residual_dimensions() { return Residual::dim; }

    static constexpr auto parameter_count() { return sizeof...(Params); }

    static constexpr auto parameter_dimensions(const int index)
    {
        constexpr int dims[] = {Params::dim...};
        return dims[index];
    }

   private:
    auto to_cost() { return new auto_diff_type(static_cast<Derived*>(this)); }

    template <typename... A, typename... B, std::size_t... I>
    static auto convert_impl(const std::tuple<A...>& a,
                             const std::tuple<B...>& b,
                             std::index_sequence<I...>)
    {
        return std::make_tuple(std::get<I>(a).map(std::get<I>(b))...);
    }

    template <typename... T, std::size_t... I>
    auto compute_impl(const std::tuple<T...>& args,
                      std::index_sequence<I...>) const
    {
        static_cast<const Derived*>(this)->compute(std::get<I>(args)...);
    }
};

} // namespace decl
} // namespace arba