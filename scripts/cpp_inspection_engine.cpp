#include <torch/script.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>

class IndustrialHysteresisFilter {
private:
    std::vector<int> window;
    size_t window_size;
    float margin_threshold;
    int last_stable_state;
    bool has_stable_state;

public:
    IndustrialHysteresisFilter(size_t size = 5, float margin = 0.25f)
        : window_size(size), margin_threshold(margin), last_stable_state(0), has_stable_state(false) {}

    int process_logits(const torch::Tensor& logits) {
        torch::Tensor probs = torch::softmax(logits, 1);
        float p0 = probs[0][0].item<float>();
        float p1 = probs[0][1].item<float>();
        float margin = std::abs(p1 - p0);
        int raw_pred = (p1 > p0) ? 1 : 0;

        window.push_back(raw_pred);
        if (window.size() > window_size) {
            window.erase(window.begin());
        }

        int sum = std::accumulate(window.begin(), window.end(), 0);
        int majority_pred = (sum > static_cast<int>(window.size() / 2)) ? 1 : 0;

        if (margin < margin_threshold && has_stable_state) {
            return last_stable_state;
        }

        last_stable_state = majority_pred;
        has_stable_state = true;
        return majority_pred;
    }
};

int main(int argc, const char* argv[]) {
    std::string model_path = "model_repository/ccvnn_inspector/1/model.pt";
    if (argc > 1) model_path = argv[1];

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
        module.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model." << std::endl;
        return -1;
    }

    IndustrialHysteresisFilter filter(5, 0.25f);
    torch::NoGradGuard no_grad;

    for (int i = 0; i < 1000; ++i) {
        torch::Tensor input_tensor = torch::rand({1, 9}, torch::kFloat32);
        torch::Tensor output = module.forward({input_tensor}).toTensor();
        int decision = filter.process_logits(output);
    }
    std::cout << "C++ Native execution completed successfully." << std::endl;
    return 0;
}
