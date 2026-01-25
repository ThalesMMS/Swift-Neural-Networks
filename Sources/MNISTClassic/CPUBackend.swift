import Foundation
import Accelerate

// CPU backend using vDSP (Accelerate).
final class CpuGemmEngine: GemmEngine {
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
    ) {
        let _ = lda
        let _ = ldb
        let _ = ldc

        let aRows = transposeA ? k : m
        let aCols = transposeA ? m : k
        let bRows = transposeB ? n : k
        let bCols = transposeB ? k : n

        var aTransposed: [Float]? = nil
        var bTransposed: [Float]? = nil

        if transposeA {
            var buffer = [Float](repeating: 0.0, count: m * k)
            buffer.withUnsafeMutableBufferPointer { outBuf in
                guard let outPtr = outBuf.baseAddress else { return }
                vDSP_mtrans(a, 1, outPtr, 1, vDSP_Length(aRows), vDSP_Length(aCols))
            }
            aTransposed = buffer
        }

        if transposeB {
            var buffer = [Float](repeating: 0.0, count: k * n)
            buffer.withUnsafeMutableBufferPointer { outBuf in
                guard let outPtr = outBuf.baseAddress else { return }
                vDSP_mtrans(b, 1, outPtr, 1, vDSP_Length(bRows), vDSP_Length(bCols))
            }
            bTransposed = buffer
        }

        func withMatrixPointer<T>(
            _ matrix: [Float]?,
            fallback: UnsafePointer<Float>,
            _ body: (UnsafePointer<Float>) -> T
        ) -> T {
            if let matrix = matrix {
                return matrix.withUnsafeBufferPointer { buf in
                    guard let ptr = buf.baseAddress else {
                        return body(fallback)
                    }
                    return body(ptr)
                }
            }
            return body(fallback)
        }

        withMatrixPointer(aTransposed, fallback: a) { aPtr in
            withMatrixPointer(bTransposed, fallback: b) { bPtr in
                let count = m * n
                if alpha == 1.0 && beta == 0.0 {
                    vDSP_mmul(
                        aPtr,
                        1,
                        bPtr,
                        1,
                        c,
                        1,
                        vDSP_Length(m),
                        vDSP_Length(n),
                        vDSP_Length(k)
                    )
                } else {
                    var temp = [Float](repeating: 0.0, count: count)
                    temp.withUnsafeMutableBufferPointer { tempBuf in
                        guard let tempPtr = tempBuf.baseAddress else { return }
                        vDSP_mmul(
                            aPtr,
                            1,
                            bPtr,
                            1,
                            tempPtr,
                            1,
                            vDSP_Length(m),
                            vDSP_Length(n),
                            vDSP_Length(k)
                        )

                        var alphaVar = alpha
                        vDSP_vsmul(tempPtr, 1, &alphaVar, tempPtr, 1, vDSP_Length(count))

                        if beta != 0.0 {
                            var betaVar = beta
                            vDSP_vsma(c, 1, &betaVar, tempPtr, 1, tempPtr, 1, vDSP_Length(count))
                        }

                        c.update(from: tempPtr, count: count)
                    }
                }
            }
        }
    }
}
