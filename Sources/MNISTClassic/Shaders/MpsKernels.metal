#include <metal_stdlib>
using namespace metal;

kernel void add_bias(device float* data [[buffer(0)]],
                     device const float* bias [[buffer(1)]],
                     constant uint& rows [[buffer(2)]],
                     constant uint& cols [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    uint total = rows * cols;
    if (gid >= total) return;
    uint col = gid % cols;
    data[gid] += bias[col];
}

kernel void relu_inplace(device float* data [[buffer(0)]],
                         constant uint& count [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float v = data[gid];
    data[gid] = v > 0.0f ? v : 0.0f;
}

kernel void relu_grad(device const float* activations [[buffer(0)]],
                      device float* grads [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (activations[gid] <= 0.0f) {
        grads[gid] = 0.0f;
    }
}

kernel void softmax_rows(device float* data [[buffer(0)]],
                         constant uint& rows [[buffer(1)]],
                         constant uint& cols [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= rows) return;
    uint base = gid * cols;
    float maxVal = data[base];
    for (uint c = 1; c < cols; ++c) {
        float v = data[base + c];
        if (v > maxVal) maxVal = v;
    }
    float sum = 0.0f;
    for (uint c = 0; c < cols; ++c) {
        float e = exp(data[base + c] - maxVal);
        data[base + c] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (uint c = 0; c < cols; ++c) {
        data[base + c] *= inv;
    }
}

kernel void sum_rows(device const float* data [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     constant uint& rows [[buffer(2)]],
                     constant uint& cols [[buffer(3)]],
                     constant float& scale [[buffer(4)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) return;
    float acc = 0.0f;
    for (uint r = 0; r < rows; ++r) {
        acc += data[r * cols + gid];
    }
    out[gid] = acc * scale;
}

kernel void delta_and_loss(device const float* outputs [[buffer(0)]],
                           device const uchar* labels [[buffer(1)]],
                           device float* delta [[buffer(2)]],
                           device float* loss [[buffer(3)]],
                           constant uint& rows [[buffer(4)]],
                           constant uint& cols [[buffer(5)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= rows) return;
    uint base = gid * cols;
    uint label = labels[gid];
    float prob = outputs[base + label];
    if (prob < 1e-9f) prob = 1e-9f;
    loss[gid] = -log(prob);
    for (uint c = 0; c < cols; ++c) {
        float v = outputs[base + c];
        if (c == label) v -= 1.0f;
        delta[base + c] = v;
    }
}

kernel void sgd_update(device float* weights [[buffer(0)]],
                       device const float* grads [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       constant float& lr [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    weights[gid] -= lr * grads[gid];
}
