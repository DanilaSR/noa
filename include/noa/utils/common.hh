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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>

#include <torch/torch.h>
#include <torch/script.h>

namespace noa::utils {

    using Path = std::filesystem::path;
    using Status = bool;
    using Line = std::string;
    using Tensor = torch::Tensor;
    using TensorOpt = std::optional<Tensor>;
    using Tensors = std::vector<Tensor>;
    using NamedTensors = std::unordered_map<std::string, Tensor>;
    using ScriptModule = torch::jit::Module;
    using ScriptModuleOpt = std::optional<ScriptModule>;
    using OutputLeaf = Tensor;
    using InputLeaf = Tensor;
    using ADGraph = std::tuple<OutputLeaf, InputLeaf>;

    constexpr double_t TOLERANCE = 1E-6;
    constexpr int32_t SEED = 987654;

    inline const auto num_pattern = std::regex{"[0-9.E\\+-]+"};

    inline Status check_path_exists(const Path &path) {
        if (!std::filesystem::exists(path)) {
            std::cerr << "Cannot find " << path << std::endl;
            return false;
        }
        return true;
    }

    inline TensorOpt load_tensor(const Path &path) {
        if (check_path_exists(path)) {
            auto tensor = Tensor{};
            try {
                torch::load(tensor, path);
            }
            catch (...) {
                std::cerr << "Failed to load tensor from " << path << "\n";
                return std::nullopt;
            }
            return std::make_optional(tensor);
        } else {
            return std::nullopt;
        }
    }

    inline std::optional<Line> find_line(
            std::ifstream &stream, const std::regex &line_pattern) {
        auto line = Line{};
        while (std::getline(stream, line))
            if (std::regex_search(line, line_pattern))
                return line;
        return std::nullopt;
    }

    template<typename Dtype>
    inline std::optional<std::vector<Dtype>> get_numerics(
            const std::string &line, int64_t size) {
        const auto no_data = std::sregex_iterator();
        auto nums = std::sregex_iterator(line.begin(), line.end(), num_pattern);
        if (std::distance(nums, no_data) != size)
            return std::nullopt;

        auto vec = std::vector<Dtype>{};
        vec.reserve(size);
        for (auto &num = nums; num != no_data; num++)
            vec.emplace_back(std::stod(num->str()));
        return vec;
    }

    template<typename Dtype, typename Lambda>
    inline void for_eachi(const Lambda &lambda, const Tensor &result) {
        Dtype *pres = result.data_ptr<Dtype>();
        const int64_t n = result.numel();
        for (int64_t i = 0; i < n; i++)
            lambda(i, pres[i]);
    }

    template<typename Dtype, typename Lambda>
    inline void pfor_eachi(const Lambda &lambda, const Tensor &result) {
        Dtype *pres = result.data_ptr<Dtype>();
        const int64_t n = result.numel();
#pragma omp parallel for default(none) shared(n, lambda, pres)
        for (int64_t i = 0; i < n; i++)
            lambda(i, pres[i]);
    }

    template<typename Dtype, typename Lambda>
    inline void for_each(const Lambda &lambda, const Tensor &result) {
        for_eachi<Dtype>([&lambda](const int64_t, Dtype &k) { lambda(k); }, result);
    }

    template<typename Dtype, typename Lambda>
    inline void pfor_each(const Lambda &lambda, const Tensor &result) {
        pfor_eachi<Dtype>([&lambda](const int64_t, Dtype &k) { lambda(k); }, result);
    }

    template<typename Dtype, typename Lambda>
    inline void vmapi(const Tensor &values, const Lambda &lambda, const Tensor &result) {
        const Dtype *pvals = values.data_ptr<Dtype>();
        for_eachi<Dtype>([&lambda, &pvals](const int64_t i, Dtype &k) { k = lambda(i, pvals[i]); }, result);
    }

    template<typename Dtype, typename Lambda>
    inline void pvmapi(const Tensor &values, const Lambda &lambda, const Tensor &result) {
        const Dtype *pvals = values.data_ptr<Dtype>();
        pfor_eachi<Dtype>([&lambda, &pvals](const int64_t i, Dtype &k) { k = lambda(i, pvals[i]); }, result);
    }


    template<typename Dtype, typename Lambda>
    inline void vmap(const Tensor &values, const Lambda &lambda, const Tensor &result) {
        vmapi<Dtype>(values,
                     [&lambda](const int64_t, const Dtype &k) { return lambda(k); },
                     result);
    }

    template<typename Dtype, typename Lambda>
    inline void pvmap(const Tensor &values, const Lambda &lambda, const Tensor &result) {
        pvmapi<Dtype>(values,
                      [&lambda](const int64_t, const Dtype &k) { return lambda(k); },
                      result);
    }

    template<typename Dtype, typename Lambda>
    inline Tensor vmapi(const Tensor &values, const Lambda &lambda) {
        const auto result = torch::zeros_like(values);
        vmapi<Dtype>(values, lambda, result);
        return result;
    }

    template<typename Dtype, typename Lambda>
    inline Tensor pvmapi(const Tensor &values, const Lambda &lambda) {
        const auto result = torch::zeros_like(values);
        pvmapi<Dtype>(values, lambda, result);
        return result;
    }

    template<typename Dtype, typename Lambda>
    inline Tensor vmap(const Tensor &values, const Lambda &lambda) {
        const auto result = torch::zeros_like(values);
        vmap<Dtype>(values, lambda, result);
        return result;
    }

    template<typename Dtype, typename Lambda>
    inline Tensor pvmap(const Tensor &values, const Lambda &lambda) {
        const auto result = torch::zeros_like(values);
        pvmap<Dtype>(values, lambda, result);
        return result;
    }

    inline Tensor relative_error(const Tensor &computed, const Tensor &expected) {
        auto res = Tensor{};
        AT_DISPATCH_FLOATING_TYPES(computed.scalar_type(), "relative_error", [&] {
            res = torch::abs(
                    (computed - expected) / (computed + std::numeric_limits<scalar_t>::min()))
                          .sum() /
                  computed.numel();
        });
        return res;
    }

    inline Tensor mean_error(const Tensor &computed, const Tensor &expected) {
        return torch::abs(computed - expected).mean();
    }

    template<typename NetData>
    inline Tensors to_tensors(const NetData &net_data) {
        auto res = std::vector<Tensor>{};
        for (const auto &val : net_data)
            res.push_back(val);
        return res;
    }


    template<typename NetNamedData>
    inline NamedTensors to_named_tensors(const NetNamedData &net_named_data) {
        auto res = std::unordered_map<std::string, Tensor>{};
        for (const auto &[name, val] : net_named_data)
            res[name] = val;
        return res;
    }

    template<typename Net>
    inline Tensors parameters(const Net &net) {
        return to_tensors(net.parameters());
    }

    template<typename Net>
    inline Tensors buffers(const Net &net) {
        return to_tensors(net.buffers());
    }

    template<typename Net>
    inline NamedTensors named_parameters(const Net &net) {
        return to_named_tensors(net.named_parameters());
    }

    template<typename Net>
    inline NamedTensors named_buffers(const Net &net) {
        return to_named_tensors(net.named_buffers());
    }

    template<typename NetData>
    inline Tensor flat_data(const NetData &net_data, bool detach) {
        auto res = std::vector<Tensor>{};
        for (const auto &val : net_data)
            res.push_back((detach ? val.detach() : val).flatten());
        return torch::cat(res);
    }

    template<typename Net>
    inline Tensor flat_parameters(const Net &net) {
        return flat_data(net.parameters(), true);
    }

    template<typename Net>
    inline Tensor flat_buffers(const Net &net) {
        return flat_data(net.buffers(), false);
    }

    template<typename NetData>
    inline bool set_flat_data(const NetData &net_data, const Tensor &data){
        if (data.dim() > 1)
        {
            std::cerr << "Invalid arguments to noa::utils::set_flat_data : "
                      << "expecting 1D parameters\n";
            return false;
        }
        int64_t i = 0;
        for (const auto &val : net_data)
        {
            const auto i0 = i;
            i += val.numel();
            val.set_data(data.slice(0, i0, i).view_as(val));
        }
        return true;
    }

    template <typename Net>
    inline bool set_flat_parameters(Net &net, const Tensor &parameters)
    {
        return set_flat_data(net.parameters(), parameters);
    }

    template <typename Net>
    inline bool set_flat_buffers(Net &net, const Tensor &buffers)
    {
        return set_flat_data(net.buffers(), buffers);
    }

    inline ScriptModuleOpt load_module(const Path &jit_module_pt){
        if (check_path_exists(jit_module_pt)) {
            try {
                return torch::jit::load(jit_module_pt);
            }
            catch (...) {
                std::cerr << "Failed to load JIT module from " << jit_module_pt << "\n";
                return std::nullopt;
            }
        } else {
            return std::nullopt;
        }
    }

} // namespace noa::utils