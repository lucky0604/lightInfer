#pragma once

#include "tensor.h"
#include "kvstorage.h"
#include "kernel/kernel.h"

namespace lightInfer {

using OpIOs = std::vector<std::shared_ptr<Tensor>>;
constexpr static size_t PACK_SIZE = 8;

// 这是一个操作（Op）的基类，调用步骤如下：

// 调用 deduce_output_shape 来获取输出张量的形状
// 调用 init 方法来初始化操作并计算工作空间
// 在执行前，应调用 pre_execute 来准备资源
// 调用 execute 来获取计算结果
// 调用 end execution 来回收资源
class OpBase {
public:
    OpBase(Device* device, const std::string& name, OpIOs inputs): m_device(device), m_inputs(inputs), m_name(name) {
        for (auto input: m_inputs)
        {
            input->add_user();
        }
    }

    virtual void pre_execute() {
        for (auto weight: m_weights)
        {
            weight->prepare_data();
        }
        for (auto output: m_outputs)
        {
            if (output->get_curr_user_count() == 0 && !output->shared())
            {
                output->resume_user_count();
                output->prepare_data();
            }
        }
    };

    virtual void execute(WorkSpace* workspace, uint32_t nr_past) {}

    virtual void end_execute() {
        for (auto input: m_inputs)
        {
            input->decrease_curr_user_count();
        }
    };

    

};
}