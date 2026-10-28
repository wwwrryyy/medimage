#include "anomaly_calculus.hpp"
#include <algorithm>

AnomalyCalculus::AnomalyCalculus(float gaussian_sigma)
    : gaussian_sigma_(gaussian_sigma) {}

float AnomalyCalculus::computePixelwiseMSE(const cv::Mat& despeckled_frame, const cv::Mat& recon_frame, cv::Mat& heatmap_out) {
    if (despeckled_frame.empty() || recon_frame.empty()) {
        return 0.0f;
    }

    cv::Mat despeckled_float, recon_float;

    // Normalize inputs to range [0.0, 1.0] CV_32F
    if (despeckled_frame.type() != CV_32F && despeckled_frame.type() != CV_32FC3) {
        despeckled_frame.convertTo(despeckled_float, CV_32F, 1.0 / 255.0);
    } else {
        despeckled_float = despeckled_frame.clone();
    }

    if (recon_frame.type() != CV_32F && recon_frame.type() != CV_32FC3) {
        recon_frame.convertTo(recon_float, CV_32F, 1.0 / 255.0);
    } else {
        recon_float = recon_frame.clone();
    }

    // Ensure matching spatial resolutions
    if (despeckled_float.size() != recon_float.size()) {
        cv::resize(recon_float, recon_float, despeckled_float.size(), 0, 0, cv::INTER_LINEAR);
    }

    // Convert multi-channel inputs to single channel grayscale if necessary
    if (despeckled_float.channels() > 1) {
        cv::cvtColor(despeckled_float, despeckled_float, cv::COLOR_BGR2GRAY);
    }
    if (recon_float.channels() > 1) {
        cv::cvtColor(recon_float, recon_float, cv::COLOR_BGR2GRAY);
    }

    // Compute pixel-wise squared difference matrix: (Despeckled - Recon)^2
    cv::Mat diff;
    cv::absdiff(despeckled_float, recon_float, diff);
    cv::multiply(diff, diff, heatmap_out);

    // Apply spatial smoothing filter to suppress isolated pixel noise
    if (gaussian_sigma_ > 0.0f) {
        cv::GaussianBlur(heatmap_out, heatmap_out, cv::Size(5, 5), gaussian_sigma_);
    }

    // Calculate mean MSE value across entire frame
    cv::Scalar mean_scalar = cv::mean(heatmap_out);
    return static_cast<float>(mean_scalar[0]);
}

void AnomalyCalculus::generateVisualHeatmap(const cv::Mat& mse_matrix, cv::Mat& visual_heatmap, bool apply_colormap) {
    if (mse_matrix.empty()) return;

    // Normalize MSE floating point matrix (0.0 - max_val) to uint8 (0 - 255)
    double min_val, max_val;
    cv::minMaxLoc(mse_matrix, &min_val, &max_val);

    double upper_bound = std::max(max_val, 0.25);
    cv::Mat normalized;
    mse_matrix.convertTo(normalized, CV_8U, 255.0 / upper_bound);

    if (apply_colormap) {
        cv::applyColorMap(normalized, visual_heatmap, cv::COLORMAP_JET);
    } else {
        visual_heatmap = normalized;
    }
}
