#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "video_ingestor.hpp"
#include "anomaly_calculus.hpp"
#include "sharp_fault_tolerance.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

namespace py = pybind11;

/**
 * Helper for the Zero-Copy Inter-Process Memory Bridge.
 * Wraps the pixel buffer of a cv::Mat in a NumPy ndarray without duplicating memory.
 * The ndarray keeps its own reference to the Mat's buffer (OpenCV reference counting),
 * so the returned array stays valid and unchanged no matter what the engine does next.
 */
template <typename T>
py::array_t<T> mat_to_numpy_zerocopy(const cv::Mat& mat) {
    if (mat.empty()) return py::array_t<T>();

    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;

    if (mat.channels() == 1) {
        shape = { static_cast<ssize_t>(mat.rows), static_cast<ssize_t>(mat.cols) };
        strides = { static_cast<ssize_t>(mat.step[0]), static_cast<ssize_t>(mat.step[1]) };
    } else {
        shape = { static_cast<ssize_t>(mat.rows), static_cast<ssize_t>(mat.cols), static_cast<ssize_t>(mat.channels()) };
        strides = { static_cast<ssize_t>(mat.step[0]), static_cast<ssize_t>(mat.step[1]), static_cast<ssize_t>(mat.elemSize1()) };
    }

    // Heap-allocated Mat header shares (and ref-counts) the pixel buffer; freed when the ndarray dies.
    cv::Mat* holder = new cv::Mat(mat);
    py::capsule owner(holder, [](void* p) { delete static_cast<cv::Mat*>(p); });
    return py::array_t<T>(shape, strides, reinterpret_cast<const T*>(holder->data), owner);
}

cv::Mat numpy_to_mat(py::array_t<uint8_t> input_array) {
    py::buffer_info buf = input_array.request();
    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);
    int channels = (buf.ndim == 3) ? static_cast<int>(buf.shape[2]) : 1;
    int type = (channels == 3) ? CV_8UC3 : CV_8UC1;
    
    return cv::Mat(rows, cols, type, buf.ptr).clone();
}

cv::Mat numpy_float_to_mat(py::array_t<float, py::array::c_style | py::array::forcecast> input_array) {
    py::buffer_info buf = input_array.request();
    if (buf.ndim != 2) {
        throw py::value_error("expected a 2-D float32 array [H, W]");
    }
    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);

    return cv::Mat(rows, cols, CV_32FC1, buf.ptr).clone();
}

/**
 * Unified Scan Core Engine exposed to Python.
 */
class ScanEngine {
private:
    VideoIngestor ingestor_;
    AnomalyCalculus anomaly_calc_;
    SHARPFaultTolerance sharp_;

    uint64_t next_frame_id_{1};
    uint64_t pending_frame_id_{0};  // 0 = no frame awaiting analysis
    std::chrono::steady_clock::time_point frame_pop_time_{};

public:
    ScanEngine(float gaussian_sigma = 1.5f) : ingestor_(30), anomaly_calc_(gaussian_sigma) {}

    bool startStream(const std::string& source) {
        if (source == "0" || source == "camera") {
            ingestor_.openCamera(0);
        } else {
            ingestor_.openSource(source);
        }
        ingestor_.start();
        return true;
    }

    void stopStream() {
        ingestor_.stop();
    }

    /**
     * Step 1: pop the next raw frame. Returns (has_frame, frame_id, raw_frame).
     * The caller despeckles and reconstructs THIS frame, then passes frame_id back to
     * computeAnomaly(), so the heatmap is always computed for the frame it describes.
     */
    py::tuple nextFrame() {
        cv::Mat raw;
        bool has_frame = ingestor_.getNextFrame(raw, 20);
        if (!has_frame) {
            return py::make_tuple(false, 0, py::none());
        }

        pending_frame_id_ = next_frame_id_++;
        frame_pop_time_ = std::chrono::steady_clock::now();
        return py::make_tuple(true, pending_frame_id_, mat_to_numpy_zerocopy<uint8_t>(raw));
    }

    /**
     * Step 2: compute the anomaly heatmap between the DESPECKLED frame and its autoencoder
     * reconstruction (speckle on the raw frame would otherwise cause false positives).
     * Returns (mse_heatmap, visual_heatmap, mean_mse).
     */
    py::tuple computeAnomaly(uint64_t frame_id,
                             py::array_t<float, py::array::c_style | py::array::forcecast> despeckled,
                             py::array_t<float, py::array::c_style | py::array::forcecast> recon) {
        if (pending_frame_id_ == 0 || frame_id != pending_frame_id_) {
            throw py::value_error("stale or unknown frame_id: call next_frame() and analyse that frame");
        }

        cv::Mat despeckled_mat = numpy_float_to_mat(despeckled);
        cv::Mat recon_mat = numpy_float_to_mat(recon);

        // Fresh output buffers per frame: previously returned arrays are never overwritten.
        cv::Mat mse_heatmap, visual_heatmap;
        float mean_mse = anomaly_calc_.computePixelwiseMSE(despeckled_mat, recon_mat, mse_heatmap);
        anomaly_calc_.generateVisualHeatmap(mse_heatmap, visual_heatmap, true);

        // Telemetry update to SHARP engine using real measured frame-to-anomaly latency
        auto now = std::chrono::steady_clock::now();
        float latency_ms = std::chrono::duration<float, std::milli>(now - frame_pop_time_).count();
        float queue_fill = static_cast<float>(ingestor_.getQueueSize()) / static_cast<float>(ingestor_.getQueueCapacity());
        sharp_.updateTelemetry(queue_fill, latency_ms, mean_mse);

        pending_frame_id_ = 0;  // frame consumed

        return py::make_tuple(mat_to_numpy_zerocopy<float>(mse_heatmap),
                              mat_to_numpy_zerocopy<uint8_t>(visual_heatmap),
                              mean_mse);
    }

    std::string getHealthStatus() const {
        return sharp_.getHealthStateString();
    }

    double getFPS() const {
        return ingestor_.getCurrentFPS();
    }

    size_t getQueueSize() const {
        return ingestor_.getQueueSize();
    }
};

PYBIND11_MODULE(scan_core, m) {
    m.doc() = "C++ Zero-Copy Hardware Ingestion and Anomaly Calculus Core for Medical Imaging Anomaly Detection";

    py::class_<ScanEngine>(m, "ScanEngine")
        .def(py::init<float>(), py::arg("gaussian_sigma") = 1.5f)
        .def("start_stream", &ScanEngine::startStream, py::arg("source"))
        .def("stop_stream", &ScanEngine::stopStream)
        .def("next_frame", &ScanEngine::nextFrame)
        .def("compute_anomaly", &ScanEngine::computeAnomaly,
             py::arg("frame_id"), py::arg("despeckled"), py::arg("recon"))
        .def("get_health_status", &ScanEngine::getHealthStatus)
        .def("get_fps", &ScanEngine::getFPS)
        .def("get_queue_size", &ScanEngine::getQueueSize);

    py::class_<AnomalyCalculus>(m, "AnomalyCalculus")
        .def(py::init<float>(), py::arg("gaussian_sigma") = 1.5f)
        .def("set_gaussian_sigma", &AnomalyCalculus::setGaussianSigma)
        .def("get_gaussian_sigma", &AnomalyCalculus::getGaussianSigma);
}
