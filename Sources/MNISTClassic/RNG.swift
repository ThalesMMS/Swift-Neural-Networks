import MNISTCommon

// Re-export SimpleRng from MNISTCommon
// The SimpleRng implementation has been moved to a shared module to eliminate duplication
public typealias SimpleRng = MNISTCommon.SimpleRng
