#include "task/inference/onnx_inferencer.h"
#include "log/logger.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

bool CompatibleBatchShape(const std::vector<int64_t>& shape, int feature_dim) {
    return shape.size() == 2 &&
           (shape[0] == 1 || shape[0] == -1) &&
           shape[1] == feature_dim;
}

std::string ShapeText(const std::vector<int64_t>& shape) {
    std::ostringstream stream;
    stream << "[";
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index > 0) {
            stream << ",";
        }
        stream << shape[index];
    }
    stream << "]";
    return stream.str();
}

}  // namespace

// ---- 构造函数 ----
OnnxInferencer::OnnxInferencer()
    : env_(ORT_LOGGING_LEVEL_WARNING, "AIServerInference") {
    // 单线程推理即可（每次推理 batch=1）
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
}

// ---- 加载 ONNX 模型（线程安全）----
bool OnnxInferencer::LoadModel(const std::string& model_path,
                              int expected_obs_dim,
                              int expected_action_dim,
                              std::string* error) {
    PreparedModel prepared;
    if (!PrepareModel(model_path, expected_obs_dim, expected_action_dim,
                      prepared, error)) {
        return false;
    }
    ActivatePreparedModel(std::move(prepared));
    return true;
}

bool OnnxInferencer::PrepareModel(const std::string& model_path,
                                 int expected_obs_dim,
                                 int expected_action_dim,
                                 PreparedModel& prepared,
                                 std::string* error) {
    std::lock_guard<std::mutex> lock(load_mutex_);
    prepared = PreparedModel{};

    try {
        // 创建候选 Session；只有完整加载成功后才替换当前 Session。
        auto new_session = std::make_shared<Ort::Session>(
            env_, model_path.c_str(), session_options_);

        if (new_session->GetInputCount() != 1 ||
            new_session->GetOutputCount() != 2) {
            throw std::runtime_error("expected one input and two outputs");
        }

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = new_session->GetInputNameAllocated(0, allocator);
        auto action_name = new_session->GetOutputNameAllocated(0, allocator);
        auto value_name = new_session->GetOutputNameAllocated(1, allocator);
        if (std::string(input_name.get()) != INPUT_NAME ||
            std::string(action_name.get()) != OUTPUT_ACTION_LOGITS ||
            std::string(value_name.get()) != OUTPUT_VALUE) {
            throw std::runtime_error("ONNX tensor names do not match configured model I/O");
        }

        auto input_type_info = new_session->GetInputTypeInfo(0);
        auto action_type_info = new_session->GetOutputTypeInfo(0);
        auto value_type_info = new_session->GetOutputTypeInfo(1);
        auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto action_info = action_type_info.GetTensorTypeAndShapeInfo();
        auto value_info = value_type_info.GetTensorTypeAndShapeInfo();
        const auto input_shape = input_info.GetShape();
        const auto action_shape = action_info.GetShape();
        const auto value_shape = value_info.GetShape();
        const auto input_type = input_info.GetElementType();
        const auto action_type = action_info.GetElementType();
        const auto value_type = value_info.GetElementType();
        if (input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            action_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            value_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            !CompatibleBatchShape(input_shape, expected_obs_dim) ||
            !CompatibleBatchShape(action_shape, expected_action_dim) ||
            !CompatibleBatchShape(value_shape, 1)) {
            std::ostringstream message;
            message << "ONNX tensor type/shape does not match configured model dimensions: input(type="
                    << static_cast<int>(input_type)
                    << ",shape=" << ShapeText(input_shape)
                    << "), action(type=" << static_cast<int>(action_type)
                    << ",shape=" << ShapeText(action_shape)
                    << "), value(type=" << static_cast<int>(value_type)
                    << ",shape=" << ShapeText(value_shape) << ")";
            throw std::runtime_error(message.str());
        }

        prepared.session = std::move(new_session);
        prepared.model_path = model_path;
        LOG_INFO("OnnxInferencer", "模型预加载成功: %s", model_path.c_str());
        return true;
    } catch (const Ort::Exception& e) {
        if (error) *error = e.what();
        LOG_ERROR("OnnxInferencer", "模型加载失败: %s, 错误: %s",
                  model_path.c_str(), e.what());
        return false;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        LOG_ERROR("OnnxInferencer", "模型校验失败: %s, 错误: %s",
                  model_path.c_str(), e.what());
        return false;
    }
}

void OnnxInferencer::ActivatePreparedModel(PreparedModel prepared) {
    if (!prepared.valid()) std::terminate();
    std::lock_guard<std::mutex> lock(load_mutex_);
    std::atomic_store(&session_, std::move(prepared.session));
    current_model_path_ = std::move(prepared.model_path);
    loaded_.store(true);
}

OnnxInferencer::PreparedModel OnnxInferencer::SnapshotPreparedModel() const {
    std::lock_guard<std::mutex> lock(load_mutex_);
    PreparedModel snapshot;
    snapshot.session = std::atomic_load(&session_);
    snapshot.model_path = current_model_path_;
    return snapshot;
}

// ---- 推理（线程安全，无锁读取）----
bool OnnxInferencer::Infer(const std::vector<float>& obs, int obs_dim,
                           std::vector<float>& action_logits, float& value) {
    // 原子读取 shared_ptr（与 LoadModel 端 atomic_store 配合，保证线程安全）
    auto session = std::atomic_load(&session_);
    return InferSession(session, obs, obs_dim, action_logits, value);
}

bool OnnxInferencer::InferPrepared(
    const PreparedModel& prepared,
    const std::vector<float>& obs,
    int obs_dim,
    std::vector<float>& action_logits,
    float& value) {
    return InferSession(
        prepared.session, obs, obs_dim, action_logits, value);
}

bool OnnxInferencer::InferSession(
    const std::shared_ptr<Ort::Session>& session,
    const std::vector<float>& obs,
    int obs_dim,
    std::vector<float>& action_logits,
    float& value) {
    if (!session) return false;

    try {
        // ---- 构建输入 Tensor ----
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(obs_dim)};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            const_cast<float*>(obs.data()),
            obs.size(),
            input_shape.data(),
            input_shape.size());

        // ---- 执行推理 ----
        const char* input_names[] = {INPUT_NAME};
        const char* output_names[] = {OUTPUT_ACTION_LOGITS, OUTPUT_VALUE};

        auto outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 2);

        // ---- 解析输出：action_logits [1, action_dim] ----
        float* logits_data = outputs[0].GetTensorMutableData<float>();
        auto logits_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int action_dim = static_cast<int>(logits_shape[1]);

        action_logits.assign(logits_data, logits_data + action_dim);

        // ---- 解析输出：value [1, 1] ----
        float* value_data = outputs[1].GetTensorMutableData<float>();
        value = value_data[0];

        return true;
    } catch (const Ort::Exception& e) {
        LOG_ERROR("OnnxInferencer", "推理失败: %s", e.what());
        return false;
    }
}

// ---- 是否已加载模型 ----
bool OnnxInferencer::IsLoaded() const {
    return loaded_.load();
}

// ---- 获取当前模型路径 ----
std::string OnnxInferencer::GetModelPath() const {
    std::lock_guard<std::mutex> lock(load_mutex_);
    return current_model_path_;
}
