#ifndef SHARP_FAULT_TOLERANCE_HPP
#define SHARP_FAULT_TOLERANCE_HPP

#include <string>
#include <atomic>

/**
 * @brief Self-Healing Approximate Resilient Programming (SHARP) Telemetry Monitor.
 * Tracks real frame processing latency, ingestion queue pressure, and anomaly MSE
 * to report current edge health state.
 */
enum class SystemHealthState {
    NOMINAL,
    DEGRADED_ADAPTIVE,
    SELF_HEALING
};

class SHARPFaultTolerance {
private:
    SystemHealthState current_health_{SystemHealthState::NOMINAL};
    std::atomic<float> queue_pressure_{0.0f};
    std::atomic<float> current_latency_ms_{0.0f};

public:
    SHARPFaultTolerance() = default;

    void updateTelemetry(float queue_fill_ratio, float latency_ms, float noise_level) {
        queue_pressure_ = queue_fill_ratio;
        current_latency_ms_ = latency_ms;

        // Edge target budget is >=15 FPS (<=66.6 ms/frame)
        if (queue_fill_ratio > 0.85f || latency_ms > 66.6f) {
            current_health_ = SystemHealthState::SELF_HEALING;
        } else if (queue_fill_ratio > 0.60f || latency_ms > 35.0f || noise_level > 0.15f) {
            current_health_ = SystemHealthState::DEGRADED_ADAPTIVE;
        } else {
            current_health_ = SystemHealthState::NOMINAL;
        }
    }

    std::string getHealthStateString() const {
        switch (current_health_) {
            case SystemHealthState::NOMINAL: return "NOMINAL (100% Edge Efficiency)";
            case SystemHealthState::DEGRADED_ADAPTIVE: return "ADAPTIVE (Noise Suppression Active)";
            case SystemHealthState::SELF_HEALING: return "SELF-HEALING (High Queue/Latency Load)";
        }
        return "UNKNOWN";
    }

    float getLastLatencyMs() const { return current_latency_ms_.load(); }
    SystemHealthState getHealthState() const { return current_health_; }
};

#endif // SHARP_FAULT_TOLERANCE_HPP
