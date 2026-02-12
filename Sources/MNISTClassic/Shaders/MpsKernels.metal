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

kernel void conv_add_bias_relu(device float* data [[buffer(0)]],
                               device const float* bias [[buffer(1)]],
                               constant uint& batch [[buffer(2)]],
                               constant uint& channels [[buffer(3)]],
                               constant uint& height [[buffer(4)]],
                               constant uint& width [[buffer(5)]],
                               uint gid [[thread_position_in_grid]]) {
    uint totalElements = batch * channels * height * width;
    if (gid >= totalElements) return;

    uint spatialSize = height * width;
    uint channel = (gid / spatialSize) % channels;

    float value = data[gid] + bias[channel];
    data[gid] = value > 0.0f ? value : 0.0f;
}

// Transpose [channels, batch*spatial] → [batch, channels, spatial] + bias + ReLU
// Input: [channels, colWidth] where colWidth = spatial * batch
// Output: [batch, channels, spatial]
kernel void conv_transpose_bias_relu(device const float* input [[buffer(0)]],
                                      device float* output [[buffer(1)]],
                                      device const float* bias [[buffer(2)]],
                                      constant uint& batch [[buffer(3)]],
                                      constant uint& channels [[buffer(4)]],
                                      constant uint& spatial [[buffer(5)]],
                                      uint gid [[thread_position_in_grid]]) {
    uint totalElements = batch * channels * spatial;
    if (gid >= totalElements) return;

    // Decode output position: [batch, channels, spatial]
    uint s = gid % spatial;
    uint c = (gid / spatial) % channels;
    uint b = gid / (spatial * channels);

    // Input is in [channels, batch*spatial] layout
    // Input index = c * (batch * spatial) + b * spatial + s
    uint inputIdx = c * (batch * spatial) + b * spatial + s;

    float value = input[inputIdx] + bias[c];
    output[gid] = value > 0.0f ? value : 0.0f;
}

// Reshape [batch, channels, spatial] → [channels, batch*spatial]
// Input: [batch, channels, spatial]
// Output: [channels, colWidth] where colWidth = batch * spatial
kernel void reshape_bcs_to_cbs(device const float* input [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& batch [[buffer(2)]],
                                constant uint& channels [[buffer(3)]],
                                constant uint& spatial [[buffer(4)]],
                                uint gid [[thread_position_in_grid]]) {
    uint totalElements = batch * channels * spatial;
    if (gid >= totalElements) return;

    // Decode input position: [batch, channels, spatial]
    uint s = gid % spatial;
    uint c = (gid / spatial) % channels;
    uint b = gid / (spatial * channels);

    // Output is in [channels, batch*spatial] layout
    // Output index = c * (batch * spatial) + b * spatial + s
    uint outputIdx = c * (batch * spatial) + b * spatial + s;

    output[outputIdx] = input[gid];
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

kernel void max_pool_forward(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& batch [[buffer(2)]],
                             constant uint& channels [[buffer(3)]],
                             constant uint& inHeight [[buffer(4)]],
                             constant uint& inWidth [[buffer(5)]],
                             constant uint& outHeight [[buffer(6)]],
                             constant uint& outWidth [[buffer(7)]],
                             constant uint& poolSize [[buffer(8)]],
                             constant uint& stride [[buffer(9)]],
                             uint gid [[thread_position_in_grid]]) {
    uint totalOut = batch * channels * outHeight * outWidth;
    if (gid >= totalOut) return;

    uint ow = gid % outWidth;
    uint oh = (gid / outWidth) % outHeight;
    uint c = (gid / (outWidth * outHeight)) % channels;
    uint b = gid / (outWidth * outHeight * channels);

    uint ih_start = oh * stride;
    uint iw_start = ow * stride;

    float maxVal = -INFINITY;
    for (uint ph = 0; ph < poolSize; ++ph) {
        for (uint pw = 0; pw < poolSize; ++pw) {
            uint ih = ih_start + ph;
            uint iw = iw_start + pw;
            if (ih < inHeight && iw < inWidth) {
                uint inIdx = ((b * channels + c) * inHeight + ih) * inWidth + iw;
                float v = input[inIdx];
                if (v > maxVal) maxVal = v;
            }
        }
    }

    output[gid] = maxVal;
}

kernel void max_pool_backward(device const float* input [[buffer(0)]],
                              device const float* outputGrad [[buffer(1)]],
                              device float* inputGrad [[buffer(2)]],
                              constant uint& batch [[buffer(3)]],
                              constant uint& channels [[buffer(4)]],
                              constant uint& inHeight [[buffer(5)]],
                              constant uint& inWidth [[buffer(6)]],
                              constant uint& outHeight [[buffer(7)]],
                              constant uint& outWidth [[buffer(8)]],
                              constant uint& poolSize [[buffer(9)]],
                              constant uint& stride [[buffer(10)]],
                              uint gid [[thread_position_in_grid]]) {
    uint totalOut = batch * channels * outHeight * outWidth;
    if (gid >= totalOut) return;

    uint ow = gid % outWidth;
    uint oh = (gid / outWidth) % outHeight;
    uint c = (gid / (outWidth * outHeight)) % channels;
    uint b = gid / (outWidth * outHeight * channels);

    uint ih_start = oh * stride;
    uint iw_start = ow * stride;

    float maxVal = -INFINITY;
    uint maxIdx = 0;
    for (uint ph = 0; ph < poolSize; ++ph) {
        for (uint pw = 0; pw < poolSize; ++pw) {
            uint ih = ih_start + ph;
            uint iw = iw_start + pw;
            if (ih < inHeight && iw < inWidth) {
                uint inIdx = ((b * channels + c) * inHeight + ih) * inWidth + iw;
                float v = input[inIdx];
                if (v > maxVal) {
                    maxVal = v;
                    maxIdx = inIdx;
                }
            }
        }
    }

    float grad = outputGrad[gid];
    atomic_fetch_add_explicit((device atomic_float*)&inputGrad[maxIdx], grad, memory_order_relaxed);
}

kernel void im2col(device const float* input [[buffer(0)]],
                   device float* output [[buffer(1)]],
                   constant uint& batch [[buffer(2)]],
                   constant uint& inChannels [[buffer(3)]],
                   constant uint& inHeight [[buffer(4)]],
                   constant uint& inWidth [[buffer(5)]],
                   constant uint& outHeight [[buffer(6)]],
                   constant uint& outWidth [[buffer(7)]],
                   constant uint& kernelSize [[buffer(8)]],
                   constant uint& stride [[buffer(9)]],
                   constant uint& padding [[buffer(10)]],
                   uint gid [[thread_position_in_grid]]) {
    uint kernelArea = kernelSize * kernelSize;
    uint outputCols = batch * outHeight * outWidth;
    uint outputRows = inChannels * kernelArea;
    uint totalElements = outputRows * outputCols;

    if (gid >= totalElements) return;

    uint col = gid % outputCols;
    uint row = gid / outputCols;

    uint kh = (row / inChannels) / kernelSize;
    uint kw = (row / inChannels) % kernelSize;
    uint c = row % inChannels;

    uint b = col / (outHeight * outWidth);
    uint oh = (col / outWidth) % outHeight;
    uint ow = col % outWidth;

    int ih = int(oh * stride + kh) - int(padding);
    int iw = int(ow * stride + kw) - int(padding);

    float value = 0.0f;
    if (ih >= 0 && ih < int(inHeight) && iw >= 0 && iw < int(inWidth)) {
        uint inIdx = ((b * inChannels + c) * inHeight + uint(ih)) * inWidth + uint(iw);
        value = input[inIdx];
    }

    output[gid] = value;
}

kernel void col2im(device const float* input [[buffer(0)]],
                   device float* output [[buffer(1)]],
                   constant uint& batch [[buffer(2)]],
                   constant uint& inChannels [[buffer(3)]],
                   constant uint& inHeight [[buffer(4)]],
                   constant uint& inWidth [[buffer(5)]],
                   constant uint& outHeight [[buffer(6)]],
                   constant uint& outWidth [[buffer(7)]],
                   constant uint& kernelSize [[buffer(8)]],
                   constant uint& stride [[buffer(9)]],
                   constant uint& padding [[buffer(10)]],
                   uint gid [[thread_position_in_grid]]) {
    uint kernelArea = kernelSize * kernelSize;
    uint inputCols = batch * outHeight * outWidth;
    uint inputRows = inChannels * kernelArea;
    uint totalElements = inputRows * inputCols;

    if (gid >= totalElements) return;

    uint col = gid % inputCols;
    uint row = gid / inputCols;

    uint kh = (row / inChannels) / kernelSize;
    uint kw = (row / inChannels) % kernelSize;
    uint c = row % inChannels;

    uint b = col / (outHeight * outWidth);
    uint oh = (col / outWidth) % outHeight;
    uint ow = col % outWidth;

    int ih = int(oh * stride + kh) - int(padding);
    int iw = int(ow * stride + kw) - int(padding);

    if (ih >= 0 && ih < int(inHeight) && iw >= 0 && iw < int(inWidth)) {
        uint outIdx = ((b * inChannels + c) * inHeight + uint(ih)) * inWidth + uint(iw);
        float value = input[gid];
        atomic_fetch_add_explicit((device atomic_float*)&output[outIdx], value, memory_order_relaxed);
    }
}
