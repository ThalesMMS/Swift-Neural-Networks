import Foundation

#if canImport(Metal)
import Metal
#endif

#if canImport(MetalPerformanceShaders)
import MetalPerformanceShaders

// GPU backend using MPSMatrixMultiplication with persistent buffers.
final class MpsGemmEngine {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              MPSSupportsMTLDevice(device),
              let queue = device.makeCommandQueue() else {
            return nil
        }
        self.device = device
        self.commandQueue = queue
    }

    func makeBuffer(count: Int, label: String, initial: [Float]? = nil) -> MpsBuffer {
        return MpsBuffer(device: device, count: count, label: label, initial: initial)
    }

    // GEMM GPU: C = alpha * A * B + beta * C (encode only).
    func encodeGemm(
        commandBuffer: MTLCommandBuffer,
        m: Int,
        n: Int,
        k: Int,
        a: MpsBuffer,
        b: MpsBuffer,
        c: MpsBuffer,
        transposeA: Bool,
        transposeB: Bool,
        alpha: Float,
        beta: Float
    ) {
        let stride = MemoryLayout<Float>.size
        let aRows = transposeA ? k : m
        let aCols = transposeA ? m : k
        let bRows = transposeB ? n : k
        let bCols = transposeB ? k : n
        let aDesc = MPSMatrixDescriptor(
            rows: aRows,
            columns: aCols,
            rowBytes: aCols * stride,
            dataType: .float32
        )
        let bDesc = MPSMatrixDescriptor(
            rows: bRows,
            columns: bCols,
            rowBytes: bCols * stride,
            dataType: .float32
        )
        let cDesc = MPSMatrixDescriptor(
            rows: m,
            columns: n,
            rowBytes: n * stride,
            dataType: .float32
        )

        let aMat = MPSMatrix(buffer: a.buffer, descriptor: aDesc)
        let bMat = MPSMatrix(buffer: b.buffer, descriptor: bDesc)
        let cMat = MPSMatrix(buffer: c.buffer, descriptor: cDesc)

        let op = MPSMatrixMultiplication(
            device: device,
            transposeLeft: transposeA,
            transposeRight: transposeB,
            resultRows: m,
            resultColumns: n,
            interiorColumns: k,
            alpha: Double(alpha),
            beta: Double(beta)
        )

        op.encode(commandBuffer: commandBuffer, leftMatrix: aMat, rightMatrix: bMat, resultMatrix: cMat)
    }

    // GEMM GPU: C = alpha * A * B + beta * C.
    func gemm(
        m: Int,
        n: Int,
        k: Int,
        a: MpsBuffer,
        b: MpsBuffer,
        c: MpsBuffer,
        transposeA: Bool,
        transposeB: Bool,
        alpha: Float,
        beta: Float
    ) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }
        encodeGemm(
            commandBuffer: commandBuffer,
            m: m,
            n: n,
            k: k,
            a: a,
            b: b,
            c: c,
            transposeA: transposeA,
            transposeB: transposeB,
            alpha: alpha,
            beta: beta
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
#endif
