package com.project.computervision;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.FeatureDetector;


import android.app.Activity;
import android.content.Context;


public class ComputerVision {
	private Context mContext;
	private Activity mActivity;
	private ComputerVisionCallback mCallback;

	private final static boolean DEBUG = true;
	private final static String TAG = "ComputerVision.java";

	public ComputerVision(Context ctx, Activity activity,
			ComputerVisionCallback callback) {
		mContext = ctx;
		mCallback = callback;
		mActivity = activity;
	}

	// Used to hook with the OpenCV service
	private BaseLoaderCallback mOpenCVCallBack = new BaseLoaderCallback(
			mActivity) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
			}
				if(DEBUG) Logd("Service hook-up finished successfully!");
				mCallback.onInitServiceFinished();
				break;
			default: {
				Loge("Service hook-up failed!");
				mCallback.onInitServiceFailed();
			}
				break;
			}
		}
	};

	/*
	 * Asynchronous OpenCV service loader.
	 * 
	 * The first thing to do before using any other functions Try to connect
	 * with the OpenCV service.
	 * 
	 * mCallback.onInitServiceFinished() would be invoked once the
	 * initialization is done
	 */
	public void initializeService() {
		if (!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_2,
				mContext, mOpenCVCallBack)) {
			Loge("Couldn't load OpenCV Engine!");
		}
		if(DEBUG) Logd("OpenCV Engine loaded");
	}
	
	private static final int KNNNUMBER = 2;
	
	/**
     * return true is the given pixel value is black
     * @param rgbx
     * @return
     */
    private boolean isBlack(double[] rgbx) {
    	return rgbx[0] == 0 && rgbx[1] == 0 && rgbx[2] == 0;
    }
    
    /**
	 * Find the keypoints of a matrix
	 * featureDetector can be obtained from the FeatureDetector class
	 * (eg. FeatureDetector.FAST)
	 */
	public synchronized MatOfKeyPoint findKeyPoints(FeatureDetector detector, Mat images){
		MatOfKeyPoint results = new MatOfKeyPoint();	
		detector.detect(images, results);
		return results;
	}
	
	// Logging function that propagates to the callback
		public void Logd(String msg) {
			mCallback.cvLogd(TAG, msg);
		}

		public void Loge(String msg) {
			mCallback.cvLoge(TAG, msg);
		}
}
