#ifndef ANOMALY_CALCULUS_HPP
#define ANOMALY_CALCULUS_HPP

#include <opencv2/opencv.hpp>

/**
 * @brief Anomaly Calculus Kernel
 * Computes pixel-wise Mean Squared Error (MSE) between despeckled input frames and
 * autoencoder reconstructed baselines, with optional spatial Gaussian smoothing.
 * Thresholding, ROI extraction, and confidence scoring live in Python (ROIDecisionLogic).
 */
class AnomalyCalculus {
private:
    float gaussian_sigma_;

public:
    explicit AnomalyCalculus(float gaussian_sigma = 1.5f);
    ~AnomalyCalculus() = default;

    /**
     * @brief Computes pixel-wise MSE heatmap matrix.
     * @param despeckled_frame Despeckled input OpenCV frame (NOT the raw speckled frame) (BGR or Grayscale, CV_8U or CV_32F).
     * @param recon_frame Reconstructed baseline frame from Convolutional Autoencoder.
     * @param heatmap_out Output CV_32FC1 floating point MSE matrix.
     * @return Mean MSE score across the entire frame.
     */
    float computePixelwiseMSE(const cv::Mat& despeckled_frame, const cv::Mat& recon_frame, cv::Mat& heatmap_out);

    /**
     * @brief Generates normalized 8-bit visual heatmap (0-255 uint8) suitable for Python / UI display.
     */
    void generateVisualHeatmap(const cv::Mat& mse_matrix, cv::Mat& visual_heatmap, bool apply_colormap = false);

    void setGaussianSigma(float sigma) { gaussian_sigma_ = sigma; }
    float getGaussianSigma() const { return gaussian_sigma_; }
};

#endif // ANOMALY_CALCULUS_HPP
