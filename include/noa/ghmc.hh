/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Roland Grinis, GrinisRIT ltd. (roland.grinis@grinisrit.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "noa/utils/numerics.hh"

#include <iostream>
#include <chrono>

#include <torch/torch.h>

namespace noa::ghmc {

    using Parameters = utils::Tensor;
    using Momentum = utils::Tensor;
    using MomentumOpt = std::optional<Momentum>;
    using LogProbability = utils::Tensor;
    using LogProbabilityGraph = utils::ADGraph;
    using Spectrum = utils::Tensor;
    using Rotation = utils::Tensor;
    using MetricDecomposition = std::tuple<Spectrum, Rotation>;
    using MetricDecompositionOpt = std::optional<MetricDecomposition>;
    using Energy = utils::Tensor;
    using PhaseSpaceFoliation = std::tuple<Parameters, Momentum, Energy>;
    using PhaseSpaceFoliationOpt = std::optional<PhaseSpaceFoliation>;

    using ParametersFlow = std::vector<Parameters>;
    using MomentumFlow = std::vector<Momentum>;
    using EnergyLevel = std::vector<Energy>;

    using HamiltonianFlow = std::tuple<ParametersFlow, MomentumFlow, EnergyLevel>;
    using ParametersGradient = utils::Tensor;
    using MomentumGradient = utils::Tensor;
    using HamiltonianGradient = std::tuple<ParametersGradient, MomentumGradient>;
    using HamiltonianGradientOpt = std::optional<HamiltonianGradient>;
    using Samples = std::vector<Parameters>;


    template<typename Dtype>
    struct Configuration {
        uint32_t max_flow_steps = 3;
        Dtype step_size = 0.1f;
        Dtype binding_const = 100.f;
        Dtype cutoff = 1e-6f;
        Dtype jitter = 1e-6f;
        Dtype softabs_const = 1e6f;
        bool verbose = false;

        inline Configuration &set_max_flow_steps(const Dtype &max_flow_steps_) {
            max_flow_steps = max_flow_steps_;
            return *this;
        }

        inline Configuration &set_step_size(const Dtype &step_size_) {
            step_size = step_size_;
            return *this;
        }

        inline Configuration &set_binding_const(const Dtype &binding_const_) {
            binding_const = binding_const_;
            return *this;
        }

        inline Configuration &set_cutoff(const Dtype &cutoff_) {
            cutoff = cutoff_;
            return *this;
        }

        inline Configuration &set_jitter(const Dtype &jitter_) {
            jitter = jitter_;
            return *this;
        }

        inline Configuration &set_softabs_const(const Dtype &softabs_const_) {
            softabs_const = softabs_const_;
            return *this;
        }

        inline Configuration &set_verbosity(bool verbose_) {
            verbose = verbose_;
            return *this;
        }
    };

    template<typename Configurations>
    inline auto softabs_metric(const Configurations &conf) {
        return [conf](const LogProbabilityGraph &log_prob_graph) {
            const auto hess_ = utils::numerics::hessian(log_prob_graph);
            if (!hess_.has_value()) {
                if (conf.verbose)
                    std::cerr << "GHMC: failed to compute hessian for log probability\n"
                              << std::get<0>(log_prob_graph) << "\n";
                return MetricDecompositionOpt{};
            }

            const auto &hess = hess_.value();
            const auto n = hess.size(0);
            const auto[eigs, rotation] = torch::symeig(
                    -hess + conf.jitter * torch::eye(n, hess.options()) * torch::rand(n, hess.options()),
                    true);

            const Rotation check_rotation = rotation.detach().sum();
            if (torch::isnan(check_rotation).item<bool>() || torch::isinf(check_rotation).item<bool>()) {
                std::cerr << "GHMC: failed to compute local rotation matrix for log probability\n"
                          << std::get<0>(log_prob_graph) << "\n";
                return MetricDecompositionOpt{};
            }

            const auto reg_eigs = torch::where(eigs.abs() >= conf.cutoff, eigs,
                                               torch::tensor(conf.cutoff, hess.options()));
            const auto spectrum = torch::abs((1 / torch::tanh(conf.softabs_const * reg_eigs)) * reg_eigs);

            const Spectrum check_spectrum = spectrum.detach().sum();
            if (torch::isnan(check_spectrum).item<bool>() || torch::isinf(check_spectrum).item<bool>()) {
                std::cerr << "GHMC: failed to compute SoftAbs map for log probability\n"
                          << std::get<0>(log_prob_graph) << "\n";
                return MetricDecompositionOpt{};
            }

            return MetricDecompositionOpt{MetricDecomposition{spectrum, rotation}};
        };
    }

    template<typename LogProbabilityDensity, typename Configurations>
    inline auto hamiltonian(const LogProbabilityDensity &log_prob_density, const Configurations &conf) {
        const auto local_metric = softabs_metric(conf);
        return [log_prob_density, local_metric, conf](
                const Parameters &parameters,
                const MomentumOpt &momentum_ = std::nullopt) {
            const auto log_prob_graph = log_prob_density(parameters);
            const auto &log_prob = std::get<0>(log_prob_graph);
            const LogProbability check_log_prob = log_prob.detach();
            if (torch::isnan(check_log_prob).item<bool>() || torch::isinf(check_log_prob).item<bool>()) {
                if (conf.verbose)
                    std::cerr << "GHMC: failed to compute log probability.\n";
                return PhaseSpaceFoliationOpt{};
            }

            const auto metric = local_metric(log_prob_graph);
            if (!metric.has_value()) {
                if (conf.verbose)
                    std::cerr << "GHMC: failed to compute local metric for log probability\n"
                              << std::get<0>(log_prob_graph) << "\n";
                return PhaseSpaceFoliationOpt{};
            }
            const auto&[spectrum, rotation] = metric.value();

            auto energy = -log_prob;

            const auto momentum_lift = momentum_.has_value()
                                       ? momentum_.value()
                                       : rotation.detach().mv(
                            torch::sqrt(spectrum.detach()) * torch::randn_like(spectrum));

            const auto momentum = momentum_lift.detach().view_as(parameters).requires_grad_(true);

            const auto first_order_term = spectrum.log().sum() / 2;
            const auto mass = rotation.mm(torch::diag(1 / spectrum)).mm(rotation.t());

            const auto momentum_vec = momentum.flatten();
            const auto second_order_term = momentum_vec.dot(mass.mv(momentum_vec)) / 2;

            energy += first_order_term + second_order_term;
            const Energy check_energy = energy.detach();
            if (torch::isnan(check_energy).item<bool>() || torch::isinf(check_energy).item<bool>()) {
                if (conf.verbose)
                    std::cerr << "GHMC: failed to compute Hamiltonian for log probability\n"
                              << std::get<0>(log_prob_graph) << "\n";
                return PhaseSpaceFoliationOpt{};
            }

            return PhaseSpaceFoliationOpt{
                    PhaseSpaceFoliation{std::get<1>(log_prob_graph), momentum, energy}};
        };
    }

    template<typename Configurations>
    inline auto hamiltonian_gradient(const Configurations &conf) {
        return [conf](const PhaseSpaceFoliationOpt &foliation) {
            if (!foliation.has_value()) {
                if (conf.verbose)
                    std::cerr << "GHMC: no phase space foliation provided.\n";
                return HamiltonianGradientOpt{};
            }
            const auto &[params, momentum, energy] = foliation.value();

            const auto ham_grad = torch::autograd::grad({energy}, {params, momentum});

            auto params_grad = ham_grad[0];
            const auto check_params = params_grad.sum();
            if (torch::isnan(check_params).item<bool>() || torch::isinf(check_params).item<bool>()) {
                if (conf.verbose)
                    std::cerr << "GHMC: failed to compute parameters gradient for Hamiltonian\n"
                              << energy << "\n";
                return HamiltonianGradientOpt{};
            }

            auto momentum_grad = ham_grad[1];
            const auto check_momentum = momentum_grad.sum();
            if (torch::isnan(check_momentum).item<bool>() || torch::isinf(check_momentum).item<bool>()) {
                if (conf.verbose)
                    std::cerr << "GHMC: failed to compute momentum gradient for Hamiltonian\n"
                              << energy << "\n";
                return HamiltonianGradientOpt{};

            }

            return HamiltonianGradientOpt{HamiltonianGradient{params_grad, momentum_grad}};
        };
    }


    template<typename LogProbabilityDensity, typename Configurations>
    inline auto hamiltonian_flow(const LogProbabilityDensity &log_prob_density, const Configurations &conf) {
        const auto ham = hamiltonian(log_prob_density, conf);
        const auto ham_grad = hamiltonian_gradient(conf);
        const auto theta = 2 * conf.binding_const * conf.step_size;
        const auto rot = std::make_tuple(cos(theta), sin(theta));
        return [ham, ham_grad, conf, rot](const Parameters &parameters,
                                          const MomentumOpt &momentum_ = std::nullopt) {
            auto params_flow = ParametersFlow{};
            params_flow.reserve(conf.max_flow_steps + 1);

            auto momentum_flow = MomentumFlow{};
            momentum_flow.reserve(conf.max_flow_steps + 1);

            auto energy_level = EnergyLevel{};
            energy_level.reserve(conf.max_flow_steps + 1);

            auto foliation = ham(parameters, momentum_);
            if (!foliation.has_value()) {
                if (conf.verbose)
                    std::cerr << "GHMC: failed to initialise Hamiltonian flow.\n";
                return HamiltonianFlow{params_flow, momentum_flow, energy_level};
            }

            const auto &[initial_params, initial_momentum, initial_energy] = foliation.value();


            auto params = initial_params.detach();
            auto momentum_copy = initial_momentum.detach();

            params_flow.push_back(params);
            momentum_flow.push_back(momentum_copy);
            energy_level.push_back(initial_energy.detach());

            uint32_t iter_step = 0;
            if (iter_step >= conf.max_flow_steps)
                return HamiltonianFlow{params_flow, momentum_flow, energy_level};

            const auto error_msg = [&iter_step, &conf]() {
                if (conf.verbose)
                    std::cerr << "GHMC: failed to evolve flow at step "
                              << iter_step + 1 << "/" << conf.max_flow_steps << "\n";
            };

            auto dynamics = ham_grad(foliation);
            if (!dynamics.has_value()) {
                error_msg();
                return HamiltonianFlow{params_flow, momentum_flow, energy_level};
            }

            const auto delta = conf.step_size / 2;
            const auto &[c, s] = rot;

            auto params_copy = params + std::get<1>(dynamics.value()) * delta;
            auto momentum = momentum_copy - std::get<0>(dynamics.value()) * delta;

            for (iter_step = 0; iter_step < conf.max_flow_steps; iter_step++) {

                foliation = ham(params_copy, momentum);
                dynamics = ham_grad(foliation);
                if (!dynamics.has_value()) {
                    error_msg();
                    break;
                }

                params = params + std::get<1>(dynamics.value()) * delta;
                momentum_copy = momentum_copy - std::get<0>(dynamics.value()) * delta;

                params = (params + params_copy +
                          c * (params - params_copy) +
                          s * (momentum - momentum_copy)) / 2;
                momentum = (momentum + momentum_copy -
                            s * (params - params_copy) +
                            c * (momentum - momentum_copy)) / 2;
                params_copy = (params + params_copy -
                               c * (params - params_copy) -
                               s * (momentum - momentum_copy)) / 2;
                momentum_copy = (momentum + momentum_copy +
                                 s * (params - params_copy) -
                                 c * (momentum - momentum_copy)) / 2;


                foliation = ham(params_copy, momentum);
                dynamics = ham_grad(foliation);
                if (!dynamics.has_value()) {
                    error_msg();
                    break;
                }

                params = params + std::get<1>(dynamics.value()) * delta;
                momentum_copy = momentum_copy - std::get<0>(dynamics.value()) * delta;

                foliation = ham(params, momentum_copy);
                dynamics = ham_grad(foliation);
                if (!dynamics.has_value()) {
                    error_msg();
                    break;
                }

                params_copy = params_copy + std::get<1>(dynamics.value()) * delta;
                momentum = momentum - std::get<0>(dynamics.value()) * delta;

                foliation = ham(params, momentum);
                if (!foliation.has_value()) {
                    error_msg();
                    break;
                }

                params_flow.push_back(params);
                momentum_flow.push_back(momentum);
                energy_level.push_back(std::get<2>(foliation.value()).detach());

                if (iter_step < conf.max_flow_steps - 1) {
                    const auto rho = -torch::relu(energy_level.back() - energy_level.front());
                    if ((rho >= torch::log(torch::rand_like(rho))).item<bool>()) {
                        params_copy = params_copy + std::get<1>(dynamics.value()) * delta;
                        momentum = momentum - std::get<0>(dynamics.value()) * delta;
                    } else {
                        if (conf.verbose)
                            std::cout << "GHMC: rejecting sample at iteration "
                                      << iter_step + 1 << "/" << conf.max_flow_steps << "\n";
                        break;
                    }
                }
            }
            return HamiltonianFlow{params_flow, momentum_flow, energy_level};
        };
    }


    template<typename LogProbabilityDensity, typename Configurations>
    inline auto sampler(const LogProbabilityDensity &log_prob_density, const Configurations &conf) {
        const auto ham_flow = hamiltonian_flow(log_prob_density, conf);
        return [ham_flow, conf](const Parameters &initial_parameters, const uint32_t num_iterations) {
            const auto max_num_samples = conf.max_flow_steps * num_iterations;

            auto samples = Samples{};
            samples.reserve(max_num_samples + 1);

            if (conf.verbose)
                std::cout << "GHMC: Riemannian HMC simulation\n"
                          << "GHMC: generating MCMC chain of maximum length "
                          << max_num_samples << " ...\n";


            samples.push_back(initial_parameters.detach());
            uint32_t iter = 0;

            while (iter < num_iterations) {
                auto flow = ham_flow(samples.back());
                const auto &params_flow = std::get<0>(flow);
                if (params_flow.size() > 1)
                    samples.insert(samples.end(), params_flow.begin() + 1, params_flow.end());
                iter++;
            }

            if (conf.verbose)
                std::cout << "GHMC: generated "
                          << samples.size() << " samples.\n";

            return samples;
        };
    }

} // namespace noa::ghmc