// Quick test to verify layout transformations
let batch = 2
let channels = 3
let spatial = 4

// Test reshape_bcs_to_cbs
print("Testing reshape [batch, channels, spatial] -> [channels, batch*spatial]")
var input = [Float](repeating: 0, count: batch * channels * spatial)
for b in 0..<batch {
    for c in 0..<channels {
        for s in 0..<spatial {
            let idx = b * (channels * spatial) + c * spatial + s
            input[idx] = Float(b * 100 + c * 10 + s)
        }
    }
}

print("Input [batch, channels, spatial]:")
for b in 0..<batch {
    for c in 0..<channels {
        for s in 0..<spatial {
            let idx = b * (channels * spatial) + c * spatial + s
            print("[\(b),\(c),\(s)] = \(input[idx])")
        }
    }
}

var output = [Float](repeating: 0, count: channels * batch * spatial)
for i in 0..<(batch * channels * spatial) {
    let s = i % spatial
    let c = (i / spatial) % channels
    let b = i / (spatial * channels)
    let outputIdx = c * (batch * spatial) + b * spatial + s
    output[outputIdx] = input[i]
}

print("\nOutput [channels, batch*spatial]:")
for c in 0..<channels {
    for b in 0..<batch {
        for s in 0..<spatial {
            let idx = c * (batch * spatial) + b * spatial + s
            print("[(\(c),\(b),\(s))] = \(output[idx])")
        }
    }
}
