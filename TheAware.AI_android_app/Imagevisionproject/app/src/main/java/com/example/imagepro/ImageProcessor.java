package com.example.imagepro;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.List;

class ImageProcessor {

    private static final int STABILITY_THRESHOLD = 10;
    private static final double PYR_SCALE = 0.3;
    private static final int LEVELS = 3;
    private static final int WINSIZE = 15;
    private static final int ITERATIONS = 3;
    private static final int POLY_N = 5;
    private static final double POLY_SIGMA = 1.2;
    private static final int FLAGS = 0;
    private static final double STABLE_FRAME_DURATION_MS = 1;
    private static final String SAVED_FRAME_MESSAGE = "Frame Saved";

    private static final int MAX_AREA_INDEX = 0;
    private static final int MIN_CONTOUR_AREA = 5000;

    private static final int MIN_PIXEL = 0;
    private static final int MAX_PIXEL = 255;

    private static final int TEXT_THICKNESS = 3;
    private static final double TEXT_REGION_SCALE = 3.5;
    private static final Scalar TEXT_COLOR_2 = new Scalar(0, 255, 0);

    private Mat prevGray;
    private Mat mRgba;
    private Mat mGray;
    private long stableFrameStartTime;
    private double stableFrameDuration;
    private boolean frameSaved;
    private ImageSaver imageSaver;

    private int counter;

    private int PROCESS_FREQUENCY = 8;

    ImageProcessor(ImageSaver imageSaver) {
        this.imageSaver = imageSaver;
        this.counter = 0;
    }

    void initialize(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        prevGray = new Mat();
        stableFrameStartTime = 0;
        stableFrameDuration = 0.0;
        frameSaved = false;
    }

    Mat processFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgbaFrame = inputFrame.rgba();
        Mat gray = new Mat();
        Imgproc.cvtColor(rgbaFrame, gray, Imgproc.COLOR_RGBA2GRAY);

        if (!prevGray.empty() && counter == 0) {
            Mat flow = new Mat();
            Video.calcOpticalFlowFarneback(prevGray, gray, flow, PYR_SCALE, LEVELS, WINSIZE, ITERATIONS, POLY_N, POLY_SIGMA, FLAGS);
            Mat magnitude = new Mat();
            List<Mat> xy = new ArrayList<>();
            Core.split(flow, xy);
            Core.magnitude(xy.get(0), xy.get(1), magnitude);

            if (frameSaved) {
                resetStableFrameTracker();
                frameSaved = false;
                return rgbaFrame;
            }
            System.out.println("my message"+Core.mean(magnitude).val[0]);
            if (Core.mean(magnitude).val[0] < STABILITY_THRESHOLD) {
                Mat processedImage = removeHorizontalLines(gray);
                List<MatOfPoint> contours = findContours(processedImage);

                if (!contours.isEmpty()) {
                    MatOfPoint maxContour = contours.get(MAX_AREA_INDEX);
                    System.out.println("Contour Area "+Imgproc.contourArea(maxContour));
                    if (Imgproc.contourArea(maxContour) >MIN_CONTOUR_AREA ) {
                        trackStableFrame(rgbaFrame);

                    }
                }
            } else {
                resetStableFrameTracker();
            }
        }

        prevGray = gray;
        counter = (counter + 1) % PROCESS_FREQUENCY;


        return rgbaFrame;
    }

    private void trackStableFrame(Mat rgbaFrame) {
        long currentTime = System.currentTimeMillis();
        if (stableFrameStartTime == 0) {
            stableFrameStartTime = currentTime;
        }
        stableFrameDuration = currentTime - stableFrameStartTime;
        System.out.println("stable_frame_Duration"+stableFrameDuration+" "+STABLE_FRAME_DURATION_MS);
        if (stableFrameDuration >= STABLE_FRAME_DURATION_MS) {
            imageSaver.saveFrameToGallery(rgbaFrame);
            drawTextOnFrame(rgbaFrame, SAVED_FRAME_MESSAGE);
            System.out.println("newmessage0");
            resetStableFrameTracker();
            frameSaved = true;
        }

    }

    private void resetStableFrameTracker() {
        stableFrameStartTime = 0;
        stableFrameDuration = 0.0;
    }

    private Mat removeHorizontalLines(Mat image) {
        Mat imageBin = new Mat();
        Imgproc.threshold(image, imageBin, 120, MAX_PIXEL, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
        Mat imageInv = new Mat();
        Core.subtract(Mat.ones(image.size(), CvType.CV_8UC1), imageBin, imageInv);
        int kernelLen = image.cols() / 100;
        Mat horizontalKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(kernelLen, 20));
        Imgproc.erode(imageInv, imageInv, horizontalKernel, new Point(-1, -1), 3);
        Mat horizontalLines = new Mat();
        Imgproc.dilate(imageInv, horizontalLines, horizontalKernel, new Point(-1, -1), 3);
        Mat imageWithoutHorizontalLines = new Mat();
        Core.subtract(Mat.ones(image.size(), CvType.CV_8UC1), horizontalLines, imageWithoutHorizontalLines);
        return imageWithoutHorizontalLines;
    }

    private List<MatOfPoint> findContours(Mat processedImage) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(processedImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        return contours;
    }

    private void drawTextOnFrame(Mat rgbaFrame, String text) {
        Size textSize = Imgproc.getTextSize(text, Core.FONT_HERSHEY_SIMPLEX, TEXT_REGION_SCALE, TEXT_THICKNESS, null);
        int textX = (rgbaFrame.width() - (int) textSize.width) / 2;
        int textY = (rgbaFrame.height() + (int) textSize.height) / 2;
        Imgproc.putText(rgbaFrame, text, new Point(textX, textY), Core.FONT_HERSHEY_SIMPLEX, TEXT_REGION_SCALE, TEXT_COLOR_2, TEXT_THICKNESS, Core.LINE_AA, false);

    }

    void release() {
        mRgba.release();
    }
}