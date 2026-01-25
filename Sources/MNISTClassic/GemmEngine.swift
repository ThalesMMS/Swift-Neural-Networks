import Foundation

#if canImport(MetalPerformanceShaders)
import Metal
import MetalPerformanceShaders
#endif

protocol GemmEngine {
    func gemm(
        m: Int,
        n: Int,
        k: Int,
        a: UnsafePointer<Float>,
        lda: Int,
        b: UnsafePointer<Float>,
        ldb: Int,
        c: UnsafeMutablePointer<Float>,
        ldc: Int,
        transposeA: Bool,
        transposeB: Bool,
        alpha: Float,
        beta: Float
    )
}

// Backend selected at runtime.
enum GemmBackend {
    case cpu(CpuGemmEngine)
    #if canImport(MetalPerformanceShaders)
    case mps(MpsGemmEngine)
    #endif
}

// Select backend: --mps tries GPU, otherwise use CPU.
func selectGemmBackend(useMPS: Bool) -> GemmBackend {
    if useMPS {
        #if canImport(MetalPerformanceShaders)
        if let engine = MpsGemmEngine() {
            print("Using MPS GEMM backend (shared buffers).")
            return .mps(engine)
        }
        #endif
        print("MPS not available, falling back to CPU.")
    }
    return .cpu(CpuGemmEngine())
}
