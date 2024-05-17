#pragma once

#include <memory>
#include <string>

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

namespace lightInfer {

struct ModelConfig{
    //! dtype include 'float32', 'float16', 'int8', 'int4'
    std::string compt_type = "float32";
    //! device_type include 'cpu', 'gpu'
    std::string device_type = "cpu";
    uint32_t nr_thread;
    uint32_t nr_ctx;
    int32_t device_id;
    bool enable_mmap;
};

class ModelImp;

class API Model {

public:
    // create a model by the model_name, the model_name must be registered internal before load it
    Model(const ModelConfig& config, const std::string& model_name);

    // load the model from model_path
    void load(const std::string& model_path);

    void init(uint32_t top_k, float top_p, float temp, float repeat_penalty, int repeat_last_n, int32_t seed, int32_t end_token);

    uint32_t get_remain_token();

    void reset_token();

    void prefill(const std::string& promote);

    std::string decode(const std::string& user_input, int& token);
    std::string decode_iter(int& token);
    std::string decode_summary() const;

private:
    std::shared_ptr<ModelImp> m_model_imp;

};

}