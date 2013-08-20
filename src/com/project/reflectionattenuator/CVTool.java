package com.project.reflectionattenuator;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.TreeMap;

import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.calib3d.StereoBM;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.KeyPoint;
import org.opencv.imgproc.Imgproc;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import android.widget.ImageView;

public class CVTool {
	
	private static String TAG = "CVTool"; 
	
	
	
	/**
	 * return true is the given pixel value is black
	 * @param rgbx
	 * @return
	 */
	public static boolean isBlack(double[] rgbx) {
		return rgbx[0] == 0 && rgbx[1] == 0 && rgbx[2] == 0;
	}

	/**
	 * Find the keypoints of a matrix
	 * featureDetector can be obtained from the FeatureDetector class
	 * (eg. FeatureDetector.FAST)
	 */
	public static synchronized MatOfKeyPoint findKeyPoints(FeatureDetector detector, Mat images){
		MatOfKeyPoint results = new MatOfKeyPoint();	
		detector.detect(images, results);
		return results;
	}


	/*
	 * Do EqualizeHist to given Mat with single channel
	 * Return the equalized Mat 
	 * 
	 */
	public static synchronized Mat equalizeHist(Mat src) {
		Mat dst = new Mat();
		Imgproc.equalizeHist(src, dst);
		return dst;
	}

	/*
	 * Do EqualizeHist to given Mat with RGB channels
	 * Return the equalized Mat 
	 * 
	 */
	public static synchronized Mat equalizeMat(Mat src) {
		List<Mat> mv = new ArrayList<Mat>();
		List<Mat> eqMv = new ArrayList<Mat>();
		Mat equalizeMat = new Mat();
		Core.split(src, mv);

		for (int i = 0; i < src.channels() - 1; i++) {
			eqMv.add(i, equalizeHist(mv.get(i)));
		}

		eqMv.add(src.channels() - 1, mv.get(src.channels() - 1));
		Core.merge(eqMv, equalizeMat);
		return equalizeMat;
	}



	/*
	 * Given the keypoints, compute the feature descriptors
	 */
	public static synchronized Mat computeDescriptors(Mat img, MatOfKeyPoint kp, DescriptorExtractor descriptorExtractor) {
		Mat desc = new Mat();
		descriptorExtractor.compute(img, kp, desc);

		return desc;
	}

	/*
	 * Given two descriptors, compute the matches
	 */
	public static synchronized MatOfDMatch getMatchingCorrespondences(Mat queryDescriptors,
			Mat trainDescriptors, DescriptorMatcher dm) {
		// Holds the result
		MatOfDMatch matches = new MatOfDMatch();
		// Compute matches
		dm.match(queryDescriptors, trainDescriptors, matches);

		return matches;
	}


	/*
	 * Given two images, their key points and matching
	 * Return a matching image between.
	 */
	public static synchronized Mat getMatchingDraw(Mat trainMat, MatOfKeyPoint trainMatOfKeyPoint, Mat queryMat, MatOfKeyPoint queryMatOfKeyPoint, MatOfDMatch matchMat) {
		Mat matchLineMat = new Mat();
		Features2d.drawMatches(trainMat, trainMatOfKeyPoint , queryMat, queryMatOfKeyPoint , matchMat, matchLineMat);
		return matchLineMat;
	}

	/*
	 * Given two images, and matching image
	 * Return a combined image with matching lines
	 */
	public static Mat getMatchingDrawOnImage(Mat trainMat, Mat queryMat, Mat matchLineMat) {
				
		Mat combineMat = new Mat(queryMat.height(), queryMat.width() + trainMat.width(), queryMat.type());

    	Mat left = new Mat(combineMat, new Rect(0, 0, trainMat.width(), trainMat.height()));
    	trainMat.copyTo(left);
    	
    	Mat right = new Mat(combineMat, new Rect(trainMat.width(), 0, queryMat.width(), queryMat.height()));
    	queryMat.copyTo(right);
		
    	rgb2Rgba(matchLineMat).copyTo(combineMat, rgb2Gray1C(matchLineMat));

    	return combineMat;
    	
	}
	

	/*
	 * Given a feature descriptor, a MatOfDmatch, which describes the reference
	 * and target image and also MatOfKeyPoint for the reference and the target
	 * image, this method returns MatOfPoints2f for the reference and target
	 * image to be used for homography computation
	 * 
	 * Return: [0] = reference
	 *         [1] = target
	 */
	public static synchronized MatOfPoint2f[] getCorrespondences(MatOfDMatch descriptors,
			MatOfKeyPoint train_kp, MatOfKeyPoint query_kp) {

		// The source of computation
		logi("HomoTrans::: find match point array 1");
		DMatch[] descriptors_array = descriptors.toArray();
		KeyPoint[] train_kp_array = train_kp.toArray();
		KeyPoint[] query_kp_array = query_kp.toArray();

		// The result
		logi("HomoTrans::: find match point array 2");
		Point[] train_pts_array = new Point[descriptors_array.length];
		Point[] query_pts_array = new Point[descriptors_array.length];

		logi("HomoTrans::: find match point array 3");
		logi("HomoTrans::: descriptor length: " + descriptors_array.length);
		logi("HomoTrans::: train_kp_array length: " + train_kp_array.length);
		logi("HomoTrans::: query_kp_array length: " + query_kp_array.length);

		//		double dist = descriptors_array.get(0).distance;


		for (int i = 0; i < descriptors_array.length; i++) {
			train_pts_array[i] = train_kp_array[descriptors_array[i].trainIdx].pt;
			query_pts_array[i] = query_kp_array[descriptors_array[i].queryIdx].pt;
		}



		logi("HomoTrans::: find match point array 4");
		MatOfPoint2f train_pts = new MatOfPoint2f(train_pts_array);
		MatOfPoint2f query_pts = new MatOfPoint2f(query_pts_array);

		logi("HomoTrans::: find match point array 5");
		MatOfPoint2f[] results = new MatOfPoint2f[2];
		results[0] = train_pts;
		results[1] = query_pts;
		return results;
	}




	/*
	 * Given forward and backward MatOfDMatch, training MatOfKeyPoint and 
	 * query MatOfKeyPoint to compute cross matches.
	 * 
	 * Return: reduced MatOfDMatch by cross check
	 */
	public static synchronized MatOfDMatch getCrossMatches(MatOfDMatch matches12, 
			MatOfDMatch matches21, MatOfKeyPoint train_kp, MatOfKeyPoint query_kp) {


		DMatch[] matches12_array = matches12.toArray();
		DMatch[] matches21_array = matches21.toArray();

		KeyPoint[] train_kp_array = train_kp.toArray();
		KeyPoint[] query_kp_array = query_kp.toArray();


		MatOfDMatch new_matches = new MatOfDMatch();
		List<DMatch> new_matchesList = new ArrayList<DMatch>();

		MatOfKeyPoint new_train_kp = new MatOfKeyPoint();
		List<KeyPoint> new_train_kpList = new ArrayList<KeyPoint>();

		MatOfKeyPoint new_query_kp = new MatOfKeyPoint();
		List<KeyPoint> new_query_kpList = new ArrayList<KeyPoint>();

		int count = 0;
		//matches12_array.length

		float distance =  matches12_array[0].distance;
		float distanceMin = distance;
		float distanceMax = 0;

		Point ptQuery = query_kp_array[matches12_array[0].queryIdx].pt;
		logi("HomoTrans::: CrossCheck :: Distance: " + distance);
		logi("HomoTrans::: CrossCheck :: Point: " + ptQuery);


		logi("matches12 queryIdx: " + matches12_array[0]);
		logi("matches12 queryIdx's point: " + query_kp_array[matches12_array[0].queryIdx]);


		for (int i = 0 ; i < matches12_array.length; i++) {


			//			logi("");
			//			logi("HomoTrans::: CrossCheck :: scan matches12 ");
			//			logi("i= " + i + " matches12 queryIdx: " + matches12_array[i].queryIdx);
			//			logi("i= " + i + " matches12 queryIdx's point: " + query_kp_array[matches12_array[i].queryIdx].pt.toString());
			//			logi("i= " + i + " matches12 trainIdx: " + matches12_array[i].trainIdx);
			//			logi("i= " + i + " matches12 trainIdx's point: " + train_kp_array[matches12_array[i].trainIdx].pt.toString());
			//		
			//			 
			//
			////			logi("!!i= " + i + " matches21 queryIdx: " + matches21_array[i].trainIdx);
			////			logi("i= " + i + " matches21 queryIdx's point: " + query_kp_array[matches21_array[i].trainIdx].pt.toString());
			////		
			////			logi("i= " + i + " matches21 trainIdx: " + matches21_array[i].queryIdx);
			////			logi("i= " + i + " matches21 trainIdx's point: " + train_kp_array[matches21_array[i].queryIdx].pt.toString());
			//		
			//		
			//			logi("HomoTrans::: CrossCheck :: scan matches21 ");
			//			int q2tId = matches12_array[i].trainIdx;
			//			logi("i= " + i + " matches21 queryIdx: " + matches21_array[q2tId].trainIdx);
			//			logi("i= " + i + " matches21 queryIdx's point: " + query_kp_array[matches21_array[q2tId].trainIdx].pt.toString());
			//			logi("i= " + i + " matches21 trainIdx: " + matches21_array[q2tId].queryIdx);
			//			logi("i= " + i + " matches21 trainIdx's point: " + train_kp_array[matches21_array[q2tId].queryIdx].pt.toString());


			distance =  matches12_array[i].distance;
			ptQuery = query_kp_array[matches12_array[i].queryIdx].pt;
			//			logi("HomoTrans::: CrossCheck :: Distance: " + distance);
			//			logi("HomoTrans::: CrossCheck :: Point: " + ptQuery);

			if (distance < distanceMin) {
				distanceMin = distance;
			}else if (distance > distanceMax) {
				distanceMax = distance;
			}

			if (matches12_array[i].queryIdx == matches21_array[matches12_array[i].trainIdx].trainIdx 
					&& matches12_array[i].trainIdx == matches21_array[matches12_array[i].trainIdx].queryIdx 
					&& distance < 100000) {


				//				if (distance < distanceMin) {
				//					distanceMin = distance;
				//				}else if (distance > distanceMax) {
				//					distanceMax = distance;
				//				}
				new_matchesList.add(matches12_array[i]);		
				new_query_kpList.add(query_kp_array[matches12_array[i].queryIdx]);			
				new_train_kpList.add(train_kp_array[matches12_array[i].trainIdx]);			
				count = count + 1;

			}

			//			logi("HomoTrans::: CrossCheck :: crossMatch count: " + count);
		}

		logi("HomoTrans::: CrossCheck :: maximum distance: " + distanceMax);
		logi("HomoTrans::: CrossCheck :: minimum distance: " + distanceMin);

		new_matches.fromList(new_matchesList);


		new_train_kp.fromList(new_train_kpList);

		new_query_kp.fromList(new_query_kpList);

		return new_matches;

	}


	/*
	 * Given MatOfDMatch, training MatOfKeyPoint, query MatOfKeyPoint, and chessBoard zones
	 * to compute cross matches.
	 * 
	 * Return: reduced MatOfDMatch gathered from each zone
	 */
	public static synchronized MatOfDMatch getLocalMatches(MatOfDMatch matches, MatOfKeyPoint train_kp, MatOfKeyPoint query_kp, int zones, int imgHeight, int imgWidth) {

		MatOfDMatch newMatches = new MatOfDMatch();
		DMatch[] localMatchesArray = new DMatch[zones * zones];
		List<DMatch> newMatchesList = new ArrayList<DMatch>();
		DMatch[] matchesArray = matches.toArray();


		KeyPoint[] train_kp_array = train_kp.toArray();
		KeyPoint[] query_kp_array = query_kp.toArray();


		logi("length: " + matchesArray.length);
		logi("height: " + imgHeight);
		logi("width: " + imgWidth);


		for (int i = 0; i < matchesArray.length; i++) {
			//			logi("i= " + i + " matches12 queryIdx's point: " + query_kp_array[matchesArray[i].queryIdx].pt.toString());
			//			logi("i= " + i + " matches12 trainIdx's point: " + train_kp_array[matchesArray[i].trainIdx].pt.toString());
			//			logi("i= " + i + " matches12 distance: " + matchesArray[i].distance);


			double intervalX = imgWidth / (double) zones;
			double intervalY = imgHeight / (double) zones;

			double chessX = Math.floor( query_kp_array[matchesArray[i].queryIdx].pt.x / intervalX );
			double chessY = Math.floor( query_kp_array[matchesArray[i].queryIdx].pt.y / intervalY );

			
			int index = (int) (chessX + chessY * zones);

			//			logi("intervalX= " + intervalX + ", intervalY= " + intervalY);
			//			logi("chessX= " + chessX + ", chessY= " + chessY);
			//			logi("index= " + index);

			if (localMatchesArray[index] == null || matchesArray[i].distance < localMatchesArray[index].distance) {
				localMatchesArray[index] = matchesArray[i];
			}
		}

		for (int i = 0; i < localMatchesArray.length; i++) {

			//			logi("location: (" + i % zones + ", " + i / zones + ")");
			if (localMatchesArray[i] != null) {
				newMatchesList.add(localMatchesArray[i]);
				//				logi("i= " + i + " localMatchesArray queryIdx's point: " + query_kp_array[localMatchesArray[i].queryIdx].pt.toString());
				//				logi("i= " + i + " localMatchesArray trainIdx's point: " + train_kp_array[localMatchesArray[i].trainIdx].pt.toString());
				//				logi("i= " + i + " localMatchesArray distance: " + localMatchesArray[i].distance);
			}
		}

		logi("local check count: " + newMatchesList.size());
		newMatches.fromList(newMatchesList);
		return newMatches;
	}

	/*
	 * Given two descriptors, compute the matchesList
	 * 
	 * Return: List of MatOfDMatch
	 */
	public static synchronized List<MatOfDMatch> getKnnMatchList(Mat queryDescriptors, Mat trainDescriptors, int numberOfMatches, DescriptorMatcher dm) {

		List<MatOfDMatch> matchesList = new ArrayList<MatOfDMatch>();
		logi("HomoTrans::: find knnMatches :: number of matches: " + numberOfMatches);
		dm.knnMatch(queryDescriptors, trainDescriptors, matchesList, numberOfMatches);

		return matchesList;
	}


	/*
	 * Given List of MatOfDMatch from knnMatches, training MatOfKeyPoint and 
	 * query MatOfKeyPoint to compute good matches.
	 * 
	 * Return: reduced MatOfDMatch by distance check
	 */
	public static synchronized MatOfDMatch getDistanceMatches(List<MatOfDMatch> knnMatchesList, MatOfKeyPoint trainMatOfKeyPoint, MatOfKeyPoint queryMatOfKeyPoint, int n, int threshold){

		logi("HomoTrans::: DistanceCheck :: matchesList size " + knnMatchesList.size());
		logi("HomoTrans::: DistanceCheck :: matchesList depth " + knnMatchesList.get(0).size());

		MatOfDMatch newMatches = new MatOfDMatch();
		List<DMatch> newMatchesList = new ArrayList<DMatch>();

		double sum = 0;
		int count = 0;
		for (int i = 0; i < knnMatchesList.size(); i++) {


			//    		logi("HomoTrans::: DistanceCheck :: matchesList 1.1 " + knnMatchesList.get(i).toList().get(0).toString());
			//        	logi("HomoTrans::: DistanceCheck :: matchesList 1.2 " + knnMatchesList.get(i).toList().get(1).toString());
			double diffDistance = Math.abs(knnMatchesList.get(i).toList().get(1).distance - knnMatchesList.get(0).toList().get(0).distance);
			//        	logi("HomoTrans::: DistanceCheck :: matchesList Distance:  " + diffDistance);
			sum = sum + diffDistance;
			if (diffDistance > threshold) {
				newMatchesList.add(knnMatchesList.get(i).toList().get(0));
				count++;
			}
		}

		logi("HomoTrans::: DistanceCheck :: count: " + count);
		logi("HomoTrans::: DistanceCheck :: distance average: " + sum/knnMatchesList.size());
		newMatches.fromList(newMatchesList);		

		return newMatches;
	}



	/*
	 * Given the reference points and the other keypoints
	 * Returns the homography matrix to transform the other to be 
	 * of the same perspective as the reference.
	 * RANSAC method is used.
	 */
	public static synchronized Mat findHomography(MatOfPoint2f trainKeyPoints, MatOfPoint2f queryKeyPoint,
			int method, int ransac_treshold){

		return Calib3d.findHomography(trainKeyPoints, queryKeyPoint, method, ransac_treshold);
	}

	/*
	 * Given the reference points and the other keypoints
	 * Returns the homography matrix to transform the other to be 
	 * of the same perspective as the reference.
	 * RANSAC method is used.
	 */
	public static synchronized Mat findHomography(Point[] trainKeyPoints, Point[] queryKeyPoints, int method,
			int ransac_treshold){
		// Intermediate data structures expected by the findHomography function
		// provided by the library
		MatOfPoint2f matReference, matOther;
		Mat result;
		matReference = new MatOfPoint2f(trainKeyPoints);
		matOther = new MatOfPoint2f(queryKeyPoints);

		result = Calib3d.findHomography(matReference, matOther, method, ransac_treshold);

		return result;
	}

	/**
	 * 
	 * @param queryMat
	 * @param homoResultMat
	 * @return Mat[0]: reduction result
	 * @return Mat[1]: pixels in queryMat covered by reflection
	 * @return Mat[2]: pixels in homoResultMat covered by reflection
	 */
	public static synchronized Mat[] reflReduction(Mat queryMat, Mat homoResultMat) {

		logi("ReflReduction::: Creat reflection layer");
		logi("ReflReduction::: Creat reflection layer:: Mat type: " + queryMat.type());
		Mat queryMatRef = new Mat(queryMat.rows(), queryMat.cols(), queryMat.type());
		Mat trainHomoMatRef = new Mat(homoResultMat.rows(), homoResultMat.cols(), homoResultMat.type());
		Mat reducResultMat = new Mat();
		queryMat.copyTo(reducResultMat);
		logi("===================");


		logi("ReflReduction::: Pixel Substition");
		Mat queryGrayMat = new Mat();
		Mat homoResultGrayMat = new Mat();
		Imgproc.cvtColor(queryMat, queryGrayMat, Imgproc.COLOR_RGBA2GRAY, 4);
		Imgproc.cvtColor(homoResultMat, homoResultGrayMat, Imgproc.COLOR_RGBA2GRAY, 4);

		for (int i = 0; i < homoResultMat.rows(); i++) {
			for (int j = 0; j < homoResultMat.cols(); j++) {
				if (queryGrayMat.get(i,j)[0] > homoResultGrayMat.get(i,j)[0] && !isBlack(homoResultMat.get(i,j))) {
					// put Reflection from queryMat to queryMatRef
					queryMatRef.put(i, j, queryMat.get(i,j));

					// replace queryPixel by smaller trainPixel
					reducResultMat.put(i, j, homoResultMat.get(i,j));
				}else {
					// put Reflection from trainMat to queryMatRef
					trainHomoMatRef.put(i, j, homoResultMat.get(i,j));
				}


			}
		}
		logi("ReflReduction::: Pixel Substition success");
		logi("===================");

		Mat[] result = {reducResultMat, queryMatRef, trainHomoMatRef};	
		return result;
	}

	/**
     * 
     * @param homoTrainMat
     * @param queryMat
     * @param iterate, numbers of looping to find contours with different contrast. Defaut = 3
     * @param sizeSE, structuring element size for opening morphology
     * @param padding, boulder buffer thinkness
     * @return Contour result Mat with 1 channel
     */
    public static synchronized Mat getSegmentation (Mat homoTrainMat, Mat queryMat, int iterate, int sizeSE, int padding, ImageView tran1ImageView, ImageView tran2ImageView, ImageView resultImageView) {

    	Mat finalContourResult = Mat.zeros(queryMat.rows(), queryMat.cols(), CvType.CV_8UC3);
    	Mat queryContourResult = Mat.zeros(queryMat.rows(), queryMat.cols(), CvType.CV_8UC3);
    	Mat trainContourResult = Mat.zeros(homoTrainMat.rows(), homoTrainMat.cols(), CvType.CV_8UC3);
    	Mat queryConMat = Mat.zeros(queryMat.size(), queryMat.type());
    	Mat trainConMat = Mat.zeros(homoTrainMat.size(), homoTrainMat.type());
    	
    	
    	
    	// find contours and draw contours of query and train image
    	for (int j = 0; j < iterate; j++) {

    		queryMat.convertTo(queryConMat, -1, 1.6 + 0.1*j, -40); // not necessary
    		homoTrainMat.convertTo(trainConMat, -1, 1.6 + 0.1*j, -40); // not necessary


    		Mat cannyQuery = Canny(queryConMat, 250 - 30 * j, 350 - 30 * j);
    		Mat cannyTrain = Canny(trainConMat, 250 - 30 * j, 350 - 30 * j);


    		
    		// fill defect 
    		int sizeSE_ = j + 1;
//    		int sizeSE_ = 6;
    		Mat SE = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(sizeSE_, sizeSE_), new Point(sizeSE_ / 2, sizeSE_ / 2));
        	Mat morCannyQuery = morphologyEx(cannyQuery, Imgproc.MORPH_CLOSE, SE, new Point(sizeSE_ / 2, sizeSE_ / 2), 1);
        	Mat morCannyTrain = morphologyEx(cannyTrain, Imgproc.MORPH_CLOSE, SE, new Point(sizeSE_ / 2, sizeSE_ / 2), 1);
    		///
    		
        	
    		
//    		showImage(cannyQuery, tran1ImageView, "show final image");
//    		showImage(morCannyQuery, tran2ImageView, "show final image");
    		
    		
    		// RETR_CCOMP
    		// CV_RETR_EXTERNAL
    		// CV_RETR_TREE
    		Mat hierarchy = new Mat();
    		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    		Imgproc.findContours(morCannyQuery, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);///test

    		for (int i = 0; i <  hierarchy.size().width; i++) {
    			if (hierarchy.get(0, i)[0] == -1 && hierarchy.get(0, i)[1] == -1) {
    				Mat countourResult = drawContours(morCannyQuery, contours, i, new Scalar(255, 0, 0), Core.FILLED, Core.LINE_8, hierarchy, 1, new Point(0,0));
    				queryContourResult = max(queryContourResult, countourResult);
    			}
    		}
    		
//    		Mat countourResult = drawContours(morCannyQuery, contours, -1, new Scalar(255, 0, 0), Core.FILLED, Core.LINE_8, hierarchy, 1, new Point(0,0));
//    		queryContourResult = max(queryContourResult, countourResult);
//    		showImage(queryContourResult, tran2ImageView, "show final image");
    		
    		
    		
    		hierarchy = new Mat();
    		contours = new ArrayList<MatOfPoint>();
    		Imgproc.findContours(morCannyTrain, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);

    		for (int i = 0; i <  hierarchy.size().width; i++) {
    			if (hierarchy.get(0, i)[0] == -1 && hierarchy.get(0, i)[1] == -1) {
    				Mat countourResult = drawContours(morCannyTrain, contours, i, new Scalar(0, 255, 0), Core.FILLED, Core.LINE_8, hierarchy, 1, new Point(0,0));
    				trainContourResult = max(trainContourResult, countourResult);
    			}
    		}
    		
//    		countourResult = drawContours(morCannyQuery, contours, -1, new Scalar(255, 0, 0), Core.FILLED, Core.LINE_8, hierarchy, 1, new Point(0,0));
//    		trainContourResult = max(queryContourResult, countourResult);
//    		showImage(queryContourResult, tran1ImageView, "show final image");
    		
    	}
		
    	
//    	showImage(queryContourResult, tran2ImageView, "show final image");///
//		showImage(trainContourResult, tran1ImageView, "show final image");///
    	
    	// Morphology on mask
    	Mat SE = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(sizeSE, sizeSE), new Point(sizeSE / 2, sizeSE / 2));
    	Mat morTrainContourResult = morphologyEx(trainContourResult, Imgproc.MORPH_OPEN, SE, new Point(sizeSE / 2, sizeSE / 2), 1);
    	Mat morQueryContourResult = morphologyEx(queryContourResult, Imgproc.MORPH_OPEN, SE, new Point(sizeSE / 2, sizeSE / 2), 1);

    	logi("Segmentation::: combine");
    	finalContourResult = combineRG21C(morTrainContourResult, morQueryContourResult);
    	logi("Segmentation::: combine success");

    	// make padding buffer at boulders 
    	SE = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(padding, padding), new Point(padding / 2, padding / 2));
    	finalContourResult = dilation(finalContourResult, SE, new Point(padding / 2, padding / 2), 1);
    	
//    	showImage(finalContourResult, resultImageView, "show final image");/////
    	
    	logi("Segmentation::: drawContour");
//    	int maxLevel = 100;
//    	Mat testResult = drawContours(queryMat, contours, -1, Core.FILLED, Core.LINE_8, hierarchy, maxLevel, new Point(0,0));
//    	Mat testResult = drawContours(queryMat, contours, -1, Core.FILLED);
    	logi("Segmentation::: drawContour success");
    	logi("/////////////////////");
     	
    	return finalContourResult;
    }
	
	
    /**
     * 
     * @param queryMat
     * @param homoResultMat
     * @param ksize, size of Gaussian blur matrix
     * @param sigmaX, sigmaX for Gaussian blur matrix
     * @param sigmaY, sigmaY for Gaussian blur matrix
     * @param segMat, segment map
     * @param threshold, threshold to define difference map
     * @param erodeEdgeWidth, the edge effect after divide should be removed due to homography
     * @return Mat[0] = difference map without doing morphology
     * @return Mat[1] = difference map with doing morphology
     */
    public static synchronized Mat[] getDifferenceMap(Mat queryMat, Mat homoResultMat, Size ksize, int sizeSE, double sigmaX, double sigmaY, double threshold, int erodeEdgeWidth) {
    	logi("ReflReduction::: Creat reflection layer");
	
		Mat queryGaussianGrayMat =  rgb2Gray1C(GussianBlur(queryMat, ksize, sigmaX, sigmaY));
		Mat homoResulGaussianGrayMat = rgb2Gray1C(GussianBlur(homoResultMat, ksize, sigmaX, sigmaY));


		Mat diffMap = divide(queryGaussianGrayMat, homoResulGaussianGrayMat);
		logi("diffMap channels:" + diffMap.channels());
		
		Mat diffThresMapWithEdge = threshold(diffMap, threshold, 255);
		Mat diffThresMap = new Mat();
		Mat morDiffMap;
		
		
		
		// remove edge effect
    	Mat homoResultGray = CVTool.rgb2Gray1C(homoResultMat);
		Mat homoResultGrayThres = CVTool.threshold(homoResultGray, 0.01, 255);
		
		// fill holes 
		int padding = 15;
		Mat SE = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(padding, padding), new Point(padding / 2, padding / 2));
		homoResultGrayThres = CVTool.dilation(homoResultGrayThres, SE, new Point(padding / 2, padding / 2), 1);
		
		// erode edge width
		padding = padding + erodeEdgeWidth;
		SE = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(padding, padding), new Point(padding / 2, padding / 2));
		homoResultGrayThres = CVTool.erosion(homoResultGrayThres, SE, new Point(padding / 2, padding / 2), 1);
		
    	diffThresMapWithEdge.copyTo(diffThresMap, homoResultGrayThres);
		

    	// morphology on diffmap
		SE = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(sizeSE, sizeSE), new Point(sizeSE / 2, sizeSE / 2));
//		morDiffMap = morphologyEx(diffThresMap, Imgproc.MORPH_OPEN, SE, new Point(sizeSE / 2, sizeSE / 2), 1);
		morDiffMap = morphologyEx(diffThresMap, Imgproc.MORPH_CLOSE, SE, new Point(sizeSE / 2, sizeSE / 2), 1);
//		morDiffMap = erosion(diffThresMap, SE, new Point(10, 10), 1);
		
//		SE = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(20, 20), new Point(10, 10));
//		morDiffMap = dilation(morDiffMap, SE, new Point(0, 0), 10);
		morDiffMap = morphologyEx(morDiffMap, Imgproc.MORPH_OPEN, SE, new Point(sizeSE / 2, sizeSE / 2), 1);
    	
    	Mat[] result = {diffThresMap, morDiffMap};
    	return result;
    }
    
	/**
	 * 
	 * @param queryMat
	 * @param homoResultMat
	 * @param segMat, segmentation map
	 * @param diffMapArray, diffMapArray[0] = diffThresMap
	 * 						diffMapArray[1] = morDiffMap
	 * 		  please look into getDifferenceMap
	 * @param pptlThreshold, value: 0~1, the threshold to determine which algorithm will
	 * 									 be used to calculate reduction reflection
	 * @param goodHomography, homography is good or not
	 * @return
	 */
    public static synchronized Mat[] reflReductionTest (Mat queryMat, Mat homoResultMat, Mat segMat, Mat[] diffMapArray, double pptlThreshold, boolean goodHomography) {

		Mat diffThresMap = diffMapArray[0];
		Mat morDiffMap = diffMapArray[1];
		Mat finalMask = Mat.zeros(queryMat.size(), CvType.CV_8UC3);
		

		logi("ReflReduction::: Re define segMat");
		Queue<Coordinate> qPoint = new LinkedList<Coordinate>();			
		
		for (int i = 0; i < segMat.rows(); i++) {
			for (int j = 0; j < segMat.cols(); j++) {
				if (segMat.get(i, j)[0] > 0 && isBlack(homoResultMat.get(i, j))) {
					segMat.put(i, j, 0);
				}
				if (morDiffMap.get(i, j)[0] > 1 && segMat.get(i, j)[0] > 1) {
					qPoint.add(new Coordinate(i,j));
				}
			}
		}
		logi("ReflReduction::: Re define segMat success");
		
		
		logi("ReflReduction::: find countour");
		Mat hierarchy = new Mat();
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Imgproc.findContours(segMat, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
		logi("ReflReduction::: find countour success");
		
		
		
		long startTime1 = System.currentTimeMillis();

		logi("ReflReduction::: define finalMask");
		Map<Integer, Integer[]> contourInfoMap= new HashMap<Integer, Integer[]>();
		Map<Integer, Mat> contourMap= new HashMap<Integer, Mat>();
		for (int k = 0; k < hierarchy.size().width; k++) {
			
			int maxLevel = 1;
			Mat countourResult = drawContours(segMat, contours, k, new Scalar(255, 0, 0), Core.FILLED, Core.LINE_8, hierarchy, maxLevel, new Point(0,0));
			contourMap.put(k, countourResult);
			
			// {pNumber, area}
			Integer[] temp = {0, countourArea(contours.get(k))};
			finalMask = max(countourResult, finalMask);/////////
			
			int end = qPoint.size();
			for (int i = 0; i < end; i++) {
				Coordinate p = qPoint.remove();

				if (countourResult.get(p.row, p.col)[0] > 0) {
					temp[0] = temp[0] + 1;
					double[] d = {0,0,255};
					finalMask.put(p.row, p.col, d);/////////
				}else {
					qPoint.add(p);
				}	
			}
			contourInfoMap.put(k, temp);
		}
		logi("ReflReduction::: define finalMask success");
		
		
		long endTime1 = System.currentTimeMillis();
		logi("TIME1 = " + (endTime1 - startTime1));
		
		
		// reduce reflection by min
		logi("ReflReduction::: subsitution");
		Mat minQuery = min(queryMat, homoResultMat);
		//draw contour and substitute pixels
		Mat subMask = new Mat();
		Mat reducResultMat = new Mat();
		
		

		
		// determine homography good or not;
		if (goodHomography) {	
			// use minResult to be background

			reducResultMat = getResultbyDamp(queryMat, homoResultMat, morDiffMap, 30);

			// padding
		}else {
			// use queryMat to be background
			queryMat.copyTo(reducResultMat);
		}
				
		
		boolean allRefOutOfObject = true;
		for (int k : contourInfoMap.keySet()) {
			if (contourInfoMap.get(k)[0] > 0) {
				// occupied proportion
				double pass = (double) contourInfoMap.get(k)[0] / contourInfoMap.get(k)[1]; 
				logi("k = " + k + ", ref pixel= " + contourInfoMap.get(k)[0] + ", total area= " + contourInfoMap.get(k)[1] + ", occupied proportion= " + pass);
				
				// replace the object #k with homographic image
				if (pass > pptlThreshold) {
					logi("object replacement from homography, taken k:" + k);
					subMask = getMaskFromCountour(contourMap.get(k));
					finalMask = max(contourMap.get(k), finalMask);///////////
					homoResultMat.copyTo(reducResultMat, subMask);
					allRefOutOfObject = false;
				
				// replace the reflection pixels in object #k with pixels in homographic image
				} else {
					logi("in object pixel replacement from homography, taken k:" + k);
//					subMask = getMaskFromCountour(contourMap.get(k));
//					finalMask = max(contourMap.get(k), finalMask);///////////
//					Mat minMask = min(subMask, morDiffMap);
//					minQuery.copyTo(reducResultMat, minMask);
				}
			
			
			// replace the object #k with query object
			} else {
				logi("object replacement from query image, taken k:" + k);
				subMask = getMaskFromCountour(contourMap.get(k));
				finalMask = max(contourMap.get(k), finalMask);///////////
				Mat minMask = min(subMask, morDiffMap);
				queryMat.copyTo(reducResultMat, minMask);///
			}
			
		}
		
		if(allRefOutOfObject) {
			logi("no segmentation available");
			reducResultMat = getResultbyDamp(queryMat, homoResultMat, morDiffMap, 10);
		}
		
//		showImage(homoResultMat, tran1ImageView, "show gaussian query gray image");	
//		showImage(reducResultMat, resultImageView, "show final image");
		
		Mat[] result = {reducResultMat, morDiffMap, finalMask};	
		return result;
	}
    
    
    public static synchronized Mat getResultbyDamp(Mat queryMat, Mat homoResultMat, Mat diffMap, int padding) {
    	
    	// make padding buffer at boulders 
    	Mat SE = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(padding, padding), new Point(padding / 2, padding / 2));
    	diffMap = dilation(diffMap, SE, new Point(padding / 2, padding / 2), 1);
    	
    	
		Mat reducResultMat = new Mat();
		queryMat.copyTo(reducResultMat);
		for (int i = 0; i < diffMap.rows(); i++) {
			for (int j = 0; j < diffMap.cols(); j++) {
				if(diffMap.get(i, j)[0] > 1 && !isBlack(homoResultMat.get(i,j))) {
					reducResultMat.put(i, j, homoResultMat.get(i,j));
				}
			}
		}
    	return reducResultMat;
    }
    
    
	/*
	 * Given the source Mat and destination Mat
	 * Returns the destination Mat with mean shift
	 * @param src The source 8-bit, 3-channel image
	 * @param sp The spatial window radius.
	 * @param sr The color window radius.
	 * 
	 */
	public static synchronized Mat getMeanShift(Mat src, double sp, double sr){
		Mat dst = new Mat();
		Mat srcRGB = new Mat(); 
		Imgproc.cvtColor(src, srcRGB, Imgproc.COLOR_RGBA2RGB, 4);
		//		srcTemp.assignTo(src, CvType.CV_8UC3);
		Imgproc.pyrMeanShiftFiltering(srcRGB, dst, sp, sr);
		return dst;
	}

	/*
	 * Given the source Mat and destination Mat
	 * Returns the destination Mat with mean shift
	 * @param src The source 8-bit, 3-channel image
	 * @param sp The spatial window radius.
	 * @param sr The color window radius.
	 * 
	 */
	public static synchronized Mat getMeanShift(Mat src, double sp, double sr, int maxLevel, TermCriteria termcrit){
		Mat dst = new Mat();
		Mat srcRGB = new Mat(); 
		Imgproc.cvtColor(src, srcRGB, Imgproc.COLOR_RGBA2RGB, 4);
		//		srcTemp.assignTo(src, CvType.CV_8UC3);
		logi("test mean shift");
		Imgproc.pyrMeanShiftFiltering(srcRGB, dst, sp, sr, maxLevel, termcrit);
		logi("test mean shift success");
		return dst;
	}

	
	
	/**
	 * Calculates perspective warp of a reference matrix with homography and returns 
	 * resutls
	 * 
	 * @param trainMatInv Reference image to be used in transformation
	 * @param homography 3x3 transfromation matrix for transform operations
	 * @param invert true if findHomography finds inverse matrix with using Imgproc.WARP_INVERSE_MAP, 
	 * 			false if normal transformation 
	 * @return transformed matrix
	 */
	public static synchronized Mat getWarpedImage(Mat trainImage, Mat homography, boolean invert){
		Mat result = new Mat(trainImage.size(), trainImage.type());
		if (invert)
			Imgproc.warpPerspective(trainImage, result, homography, trainImage.size(),Imgproc.WARP_INVERSE_MAP);
		else
			Imgproc.warpPerspective(trainImage, result, homography, trainImage.size());
		return result;
	}


	/**
	 * <p>findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99);</p>
	 *
	 * @param points1 Array of <code>N</code> points from the first image. The point
	 * coordinates should be floating-point (single or double precision).
	 * @param points2 Array of the second image points of the same size and format
	 * as <code>points1</code>.
	 * @param method Method for computing a fundamental matrix.
	 * <ul>
	 *   <li> CV_FM_7POINT for a 7-point algorithm. <em>N = 7</em>
	 *   <li> CV_FM_8POINT for an 8-point algorithm. <em>N >= 8</em>
	 *   <li> CV_FM_RANSAC for the RANSAC algorithm. <em>N >= 8</em>
	 *   <li> CV_FM_LMEDS for the LMedS algorithm. <em>N >= 8</em>
	 * </ul>
	 * @param param1 Parameter used for RANSAC. It is the maximum distance from a
	 * point to an epipolar line in pixels, beyond which the point is considered an
	 * outlier and is not used for computing the final fundamental matrix. It can be
	 * set to something like 1-3, depending on the accuracy of the point
	 * localization, image resolution, and the image noise.
	 * @param param2 Parameter used for the RANSAC or LMedS methods only. It
	 * specifies a desirable level of confidence (probability) that the estimated
	 * matrix is correct.
	 */
	public static synchronized Mat findFundamentalMat(MatOfPoint2f points1, MatOfPoint2f points2, int method, double param1, double param2) {
		return Calib3d.findFundamentalMat(points1, points2, method, param1, param2);
	}


	/** <p>Computes a rectification transform for an uncalibrated stereo camera.</p>
	 * @param points1 Array of feature points in the first image.
	 * @param points2 The corresponding points in the second image. The same formats
	 * as in "findFundamentalMat" are supported.
	 * @param F Input fundamental matrix. It can be computed from the same set of
	 * point pairs using "findFundamentalMat".
	 * @param imgSize Size of the image.
	 * @param H1 Output rectification homography matrix for the first image.
	 * @param H2 Output rectification homography matrix for the second image.
	 * @param threshold Optional threshold used to filter out the outliers. If the
	 * parameter is greater than zero, all the point pairs that do not comply with
	 * the epipolar geometry (that is, the points for which <em>|points2[i]^T*F*points1[i]|&gtthreshold</em>)
	 * are rejected prior to computing the homographies. Otherwise,all the points
	 * are considered inliers.
	 *
	 * @see <a href="http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#stereorectifyuncalibrated">org.opencv.calib3d.Calib3d.stereoRectifyUncalibrated</a>
	 */
	public static synchronized Mat[] stereoRectifyUncalibrated(Mat points1, Mat points2, Mat F, Size imgSize, double threshold) {
		Mat H1 = new Mat();
		Mat H2 = new Mat();
		Mat[] result = new Mat[2];
		Calib3d.stereoRectifyUncalibrated(points1, points2, F, imgSize, H1, H2, threshold);
		logi("disparity" + H1.size());
		result[0] = H1;
		result[1] = H2;
		return result;
	}

	/**
	 * @param F 3x3 fundamental matrix.
	 * @param points1 1xN array containing the first set of points.
	 * @param points2 1xN array containing the second set of points.
	 * @param newPoints1 The optimized points1.
	 * @param newPoints2 The optimized points2.
	 *
	 * @see <a href="http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#correctmatches">org.opencv.calib3d.Calib3d.correctMatches</a>
	 */
	public static synchronized MatOfPoint2f[] correctMatches(Mat F, MatOfPoint2f pts1, MatOfPoint2f pts2) {
		MatOfPoint2f[] results = new MatOfPoint2f[2];
		//		Mat points1 = pts1.t();
		//		Mat points2 = pts2.t();

		Mat points1 = pts1;
		Mat points2 = pts2;

		logi("Mat points: " + points1.size());
		logi("Mat points: " + points2.size());

		MatOfPoint2f newPoints1 = new MatOfPoint2f();
		MatOfPoint2f newPoints2 = new MatOfPoint2f();

		logi("here");
		Calib3d.correctMatches(F, pts1, pts2, newPoints1, newPoints2);		
		return results;
	}

	/**
	 * 
	 * @param queryMat: image1 for disparity map
	 * @param homoMat: image2 for disparity map
	 * @return disparity value passed threshold
	 */
	public static synchronized Mat getDisparityMap(Mat queryMat, Mat homoMat){
		// Disparity-like Map FAIL


		logi("Disparity Map::: Find disparity map");
		StereoBM stereoComputer = new StereoBM();

		Mat disparity = new Mat(queryMat.size(), queryMat.type());
		Mat homoGrayResultMat = new Mat();
		Mat queryGrayMat = new Mat();


		Imgproc.cvtColor(queryMat, queryGrayMat, Imgproc.COLOR_RGBA2GRAY, 1);
		Imgproc.cvtColor(homoMat, homoGrayResultMat, Imgproc.COLOR_RGBA2GRAY, 1);


		stereoComputer.compute(homoGrayResultMat, queryGrayMat, disparity);
		logi("Disparity Map::: type: " + CvType.typeToString(disparity.type()));

		Mat disparityMat = new Mat(disparity.rows(), disparity.cols(), CvType.CV_8UC1);

		queryMat.copyTo(disparityMat);

		logi("Disparity Map::: create disparityMat success ");

		// change pixels whose disparity values passed the threshold
		for (int i = 0; i < disparity.rows(); i++) {
			for (int j = 0; j < disparity.cols(); j++) {
				logi("Disparity Map::: disparity: (" + i + "," + j + ")= " + disparity.get(i, j)[0]);
				if (disparity.get(i, j)[0] > 0) {
					double[] temp = {255, 0, 0, 255};
					disparityMat.put(i, j, temp);
				}
			}
		}

		logi("Disparity Map::: visualized success");
		logi("Disparity Map::: Find disparity map success");
		logi("===================");
		return disparityMat;
	}
	
	/**
	 * 
	 * @param trainMat
	 * @param queryMat
	 * @param trainMatPt
	 * @param queryMatPt
	 * @param fundamentalMatrix: fundamental matrix 
	 * @param imgSize: trainMat.size()
	 * @param threshold: 0.5
	 * @return Mat[0]: Rectified trainMat
	 * @return Mat[1]: Rectified queryMat
	 */
	public static synchronized Mat[] getRectifiedMat(Mat trainMat, Mat queryMat, Mat trainMatPt, Mat queryMatPt, Mat fundamentalMatrix, double threshold){
		////////////////////Disparity Map ////////////////// Fail

		//Get rectified matrix
		logi("Disparity Map::: get rectified matrix");
		Mat recMatrix[] = stereoRectifyUncalibrated(trainMatPt, queryMatPt, fundamentalMatrix, trainMat.size(), threshold);
		Mat trainRecTransMat = recMatrix[0];
		Mat queryRecTransMat = recMatrix[1];
		logi("Disparity Map::: get rectified matrix success");
		logi("===================");

		//Rectified Transformation
		logi("Disparity Map::: Do rectified transfrom");
		Mat  trainRecMat = getWarpedImage(trainMat, trainRecTransMat, false);
		Mat  queryRecMat = getWarpedImage(queryMat, queryRecTransMat, false);
		
		Mat[] result = {trainRecMat, queryRecMat};
		
		logi("Disparity Map::: Do rectified transfrom success");
		logi("===================");
		return result;
		/////////////////////////////////////////////////
	}
	
	// Gussian filter
	public static Mat GussianBlur(Mat src, Size ksize, double sigmaX, double sigmaY) {
		logi("Reflection Reduction::: GaussianBlur");
		Mat dst = new Mat();
		Imgproc.GaussianBlur(src, dst, ksize, sigmaX, sigmaY);
		logi("Reflection Reduction::: GaussianBlur");
		return dst;
	}
	
	// Return pixels with smaller value between two Mat in each channel
	public static Mat min(Mat src1, Mat src2) {
		logi("Reflection Reduction::: min");
		Mat dst = new Mat();
		Core.min(src1, src2, dst);
		logi("Reflection Reduction::: min success");
		return dst;
	}
	
	// Return pixels with smaller value between two Mat in each channel
	public static Mat max(Mat src1, Mat src2) {
//		logi("Reflection Reduction::: max");
		Mat dst = new Mat();
		Core.max(src1, src2, dst);
//		logi("Reflection Reduction::: max success");
		return dst;
	}
	
	
	
	// Retrun the pixels pass threshold
	// dst(x,y) = maxval
	public static Mat threshold(Mat src, double thresh, double maxval) {
		logi("Reflection Reduction::: threshold");
		Mat dst = new Mat();
		int type = Imgproc.THRESH_BINARY;	
		Imgproc.threshold(src, dst, thresh, maxval, type);
		logi("Reflection Reduction::: threshold success");
		return dst;
	}
	
	 /**@param src Source image. The number of channels can be arbitrary. The depth
	 * should be one of <code>CV_8U</code>, <code>CV_16U</code>, <code>CV_16S</code>,
	 * <code>CV_32F" or </code>CV_64F".
	 * @param dst Destination image of the same size and type as <code>src</code>.
	 * @param op Type of a morphological operation that can be one of the following:
	 * <ul>
	 *   <li> MORPH_OPEN - an opening operation
	 *   <li> MORPH_CLOSE - a closing operation
	 *   <li> MORPH_GRADIENT - a morphological gradient
	 *   <li> MORPH_TOPHAT - "top hat"
	 *   <li> MORPH_BLACKHAT - "black hat"
	 * </ul>
	 * @param kernel a kernel
	 * @param anchor a anchor
	 * @param iterations Number of times erosion and dilation are applied.
	 * @param borderType Pixel extrapolation method. See "borderInterpolate" for
	 * details.
	 * @param borderValue Border value in case of a constant border. The default
	 * value has a special meaning. See "createMorphologyFilter" for details.
	 * */
	public static Mat morphologyEx(Mat src, int op, Mat kernel, Point anchor, int iterations) {
		logi("Reflection Reduction::: morphology");
		Mat dst = new Mat();
		Imgproc.morphologyEx(src, dst, op, kernel, anchor, iterations);
		logi("Reflection Reduction::: morphology success");
		return dst;
	}
	
	// Homography
	public static Mat dilation(Mat src,Mat kernel, Point anchor, int iterations) {
		logi("Reflection Reduction::: Dilation");
		Mat dst = new Mat();
		Imgproc.dilate(src, dst, kernel, anchor, iterations);
		logi("Reflection Reduction::: Dilation success");
		return dst;
	}
	
	// Homography
	public static Mat erosion(Mat src, Mat kernel, Point anchor, int iterations) {
		logi("Reflection Reduction::: Erosion");
		Mat dst = new Mat();
		Imgproc.erode(src, dst, kernel, anchor, iterations);
		logi("Reflection Reduction::: Erosion success");
		return dst;
	}
	
	// Sampling down by Gaussian Pyramid 
	public static Mat pyrDown(Mat src, Size dstsize) {
		logi("Image Processing::: DownSampling");
		logi("Image Processing::: Origianl Size= " + src.size().toString());
		Mat dst = new Mat();
		Imgproc.pyrDown(src, dst, dstsize);
		logi("Image Processing::: New Size= " + dst.size().toString());
		logi("Image Processing::: DownSampling success");
		return dst;
	}
	
	/**
	 * @param image Source, an 8-bit single-channel image. Non-zero pixels are
	 * treated as 1's. Zero pixels remain 0's, so the image is treated as
	 * @param contours Detected contours. Each contour is stored as a vector of
	 * points.
	 * @param hierarchy Optional output vector containing information about the
	 * image topology. It has as many elements as the number of contours. For each
	 * contour <code>contours[i]</code>, the elements <code>hierarchy[i][0]</code>,
	 * <code>hiearchy[i][1]</code>, <code>hiearchy[i][2]</code>, and
	 * <code>hiearchy[i][3]</code> are set to 0-based indices in <code>contours</code>
	 * of the next and previous contours at the same hierarchical level: the first
	 * child contour and the parent contour, respectively. If for a contour
	 * <code>i</code> there are no next, previous, parent, or nested contours, the
	 * corresponding elements of <code>hierarchy[i]</code> will be negative.
	 * @param mode Contour retrieval mode (if you use Python see also a note below).
	 * <ul>
	 *   <li> CV_RETR_EXTERNAL retrieves only the extreme outer contours. It sets
	 * <code>hierarchy[i][2]=hierarchy[i][3]=-1</code> for all the contours.
	 *   <li> CV_RETR_LIST retrieves all of the contours without establishing any
	 * hierarchical relationships.
	 *   <li> CV_RETR_CCOMP retrieves all of the contours and organizes them into a
	 * two-level hierarchy. At the top level, there are external boundaries of the
	 * components. At the second level, there are boundaries of the holes. If there
	 * is another contour inside a hole of a connected component, it is still put at
	 * the top level.
	 *   <li> CV_RETR_TREE retrieves all of the contours and reconstructs a full
	 * hierarchy of nested contours. This full hierarchy is built and shown in the
	 * OpenCV <code>contours.c</code> demo.
	 * </ul>
	 * @param method Contour approximation method (if you use Python see also a note
	 * below).
	 * <ul>
	 *   <li> CV_CHAIN_APPROX_NONE stores absolutely all the contour points. That
	 * is, any 2 subsequent points <code>(x1,y1)</code> and <code>(x2,y2)</code> of
	 * the contour will be either horizontal, vertical or diagonal neighbors, that
	 * is, <code>max(abs(x1-x2),abs(y2-y1))==1</code>.
	 *   <li> CV_CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal
	 * segments and leaves only their end points. For example, an up-right
	 * rectangular contour is encoded with 4 points.
	 *   <li> CV_CHAIN_APPROX_TC89_L1,CV_CHAIN_APPROX_TC89_KCOS applies one of the
	 * flavors of the Teh-Chin chain approximation algorithm. See [TehChin89] for
	 * details.
	 * </ul>
	 *
	 * @see <a href="http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#findcontours">org.opencv.imgproc.Imgproc.findContours</a>
	 */
	public static List<MatOfPoint> findContours(Mat image, int mode, int method) {
		Mat hierarchy = new Mat();
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Imgproc.findContours(image, contours, hierarchy, mode, method);
		return contours;
	}
	
	
	public static Mat drawContours(Mat src, List<MatOfPoint> contours, int contourIdx, int thickness) {
		Mat image = Mat.zeros(src.rows(), src.cols(), CvType.CV_8UC3);
		Scalar color = new Scalar(255, 0, 0);
		Imgproc.drawContours(image, contours, contourIdx, color, thickness);
		return image;
		
	}
	
	//  Core.LINE_8
	public static Mat drawContours(Mat src, List<MatOfPoint> contours, int contourIdx, Scalar color, int thickness, int lineType, Mat hierarchy, int maxLevel, Point offset) {
		Mat image = Mat.zeros(src.rows(), src.cols(), CvType.CV_8UC3);
//		Scalar color = new Scalar(255, 0, 0);
		Imgproc.drawContours(image, contours, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset);
		return image;
		
	}
	
	
	public static int countourArea(Mat contour) {
		return (int) Imgproc.contourArea(contour);
	}
	
	
	// comnbine the G chanel of src1 and R chanel of src2 and return and Mat with 1 chanel
	public static synchronized Mat combineRG21C(Mat src1, Mat src2) {
		List<Mat> mv1 = new ArrayList<Mat>();
		List<Mat> mv2 = new ArrayList<Mat>();
		Core.split(src1, mv1);
		Core.split(src2, mv2);
		Mat result = max(threshold(mv1.get(1), 1, 255), threshold(mv2.get(0), 1, 255));
		return result;
	}
	
	/**
	 * 
	 * @param dim dimension index along which the matrix is reduced. 0 means that
	 * the matrix is reduced to a single row. 1 means that the matrix is reduced to
	 * a single column.
	 * @param rtype reduction operation that could be one of the following:
	 * <ul>
	 *   <li> CV_REDUCE_SUM: the output is the sum of all rows/columns of the
	 * matrix.
	 *   <li> CV_REDUCE_AVG: the output is the mean vector of all rows/columns of
	 * the matrix.
	 *   <li> CV_REDUCE_MAX: the output is the maximum (column/row-wise) of all
	 * rows/columns of the matrix.
	 *   <li> CV_REDUCE_MIN: the output is the minimum (column/row-wise) of all
	 * rows/columns of the matrix.
	 * </ul>
	 */
	public static Mat reduce(Mat src, int dim, int rtype) {
		Mat dst = new Mat();
		Core.reduce(src, dst, dim, rtype);
		return dst;
	}
	
	/**
	 * 
	 *   <li> CV_REDUCE_SUM: the output is the sum of all rows/columns of the
	 * matrix.
	 *   <li> CV_REDUCE_AVG: the output is the mean vector of all rows/columns of
	 * the matrix.
	 *   <li> CV_REDUCE_MAX: the output is the maximum (column/row-wise) of all
	 * rows/columns of the matrix.
	 *   <li> CV_REDUCE_MIN: the output is the minimum (column/row-wise) of all
	 * rows/columns of the matrix.
	 */
	public static double getMatByType(Mat src, int rtype) {
		return reduce(reduce(src, 0, rtype), 1, rtype).get(0, 0)[0];
	}
	
	
	
	
	
	public static List<HashMap<Double, Integer>> calcHist(Mat src) {
		Map<Double, Integer> histogramR = new HashMap<Double, Integer>();
		Map<Double, Integer> histogramG = new HashMap<Double, Integer>();
		Map<Double, Integer> histogramB = new HashMap<Double, Integer>();

		List<HashMap<Double, Integer>> histogram = new ArrayList<HashMap<Double, Integer>>();
		histogram.add((HashMap<Double, Integer>) histogramR);
		histogram.add((HashMap<Double, Integer>) histogramG);
		histogram.add((HashMap<Double, Integer>) histogramB);
		
		Mat histo = Mat.zeros(new Size(256, 3), CvType.CV_32SC1);
		
		long startTime1 = System.currentTimeMillis();
		for (int i = 0; i < src.rows(); i++) {
			for (int j = 0; j < src.cols(); j++) {
//				logi("i= " + i + ", j= " + j);
				double[] temp = src.get(i, j);
//				logi("size" + temp.length);

				for (int k = 0; k < 3; k++) {
//					logi("k= " + k + ", i= " + i + ", j= " + j + ", intensity value= " + temp[k]);
					if (histogram.get(k).keySet().contains(temp[k])) {
//						logi("histogram value= " + histogram.get(k).get(temp[k]));
						histogram.get(k).put(temp[k], histogram.get(k).get(temp[k]) + 1);
					} else {
//						logi("k= " + k + ", i= " + i + ", j=" + j + ", intensity value= " + temp[k]);
						histogram.get(k).put(temp[k], 1);
					}
				}
				

			}
			
		}
		long endTime1 = System.currentTimeMillis();
		logi("TIME1 = " + (endTime1 - startTime1));
		
		int sum = 0;
		for (double i: histogram.get(0).keySet()) {
			sum = sum + histogram.get(0).get(i);
		}
		

		
		logi("rows=" + src.rows() + "cols=" + src.cols());
		
		return histogram;
	}
	
	
	// return  Histogram
	public static Mat calcHist(Mat src, boolean accumulate) {

		Mat histogram = Mat.zeros(new Size(256, 3), CvType.CV_32SC1);
		
		long startTime1 = System.currentTimeMillis();
		for (int i = 0; i < src.rows(); i++) {
			for (int j = 0; j < src.cols(); j++) {
				double[] temp = src.get(i, j);		
				for (int k = 0; k < 3; k++) {
					double value = temp[k];
					double[] values = histogram.get(k, (int)value);
					values[0]++;
					histogram.put(k, (int) value, values);
				}
			}
			
		}
		long endTime1 = System.currentTimeMillis();
		logi("TIME1 = " + (endTime1 - startTime1));
//		
//		for (int k = 0; k < 3; k++) {
//			int sum = 0;
//			for (int i = 0; i < 256; i++) {
//				sum = (int) (sum + histogram.get(k, i)[0]);
//			}
//			logi("sum=" + sum);
//		}
//		
//		logi("rows=" + src.rows() + "cols=" + src.cols());
		return histogram;
	}

	
	// return accumulative Histogram
	public static Mat calcAcumHist(Mat histogram) {
		
		Mat acumHistogram = Mat.zeros(new Size(256, 3), CvType.CV_32SC1);
		
		double[] sum = {0, 0, 0};
		for (int i = 0; i < 256; i++) {
			for (int k = 0; k < 3; k++) {
				double[] values = histogram.get(k, i);
				sum[k] = sum[k] + values[0];
				values[0] = sum[k];
				acumHistogram.put(k, i, values);
			}
		}
		
//		for (int k = 0; k < 1; k++) {
//			for (int i = 0; i < 256; i++) {
//				logi("intensity= " + i + ", quantity= " + acumHistogram.get(k, i)[0]);
//			}
//		}
		
		return acumHistogram;
	}
	
	
	// return transfrom value of acumHistoSrc1 (map src1 to src2)
	public static Mat getTransformValue(Mat acumHistoSrc1, Mat acumHistoSrc2) {
		
		Mat transHistogram = Mat.zeros(new Size(256, 3), CvType.CV_8UC1);
		
		for (int k = 0; k < 3; k++) {
			
			int v = 0; 
			for (int i = 0; i < 256; i++) {
				while(acumHistoSrc1.get(k, i)[0] > acumHistoSrc2.get(k, v)[0]) {
					v++;
				}
				
				
				double[] values = transHistogram.get(k, i);
				values[0] = v;
				transHistogram.put(k, i, values); 
			}
		}
		
//		for (int k = 0; k < 3; k++) {
//			for (int i = 0; i < 256; i++) {
//				logi("intensity= " + i + ", quantity= " + transHistogram.get(k, i)[0]);
//			}
//		}
		return transHistogram;
	}
	
	
	public static Mat histogramSpecificaiton(Mat trainMat, Mat queryMat, Mat transMat) {

		Mat newTrainMat = new Mat();
		trainMat.copyTo(newTrainMat);
		
		for (int i = 0; i < trainMat.rows(); i++) {
			for (int j = 0; j < trainMat.cols(); j++) {
				double[] temp = trainMat.get(i, j);		
				for (int k = 0; k < 3; k++) {
					temp[k] = transMat.get(k, (int)temp[k])[0];
				}
				newTrainMat.put(i, j, temp);
			}
		}
		return newTrainMat;
	}
	
	
	
	// dst = src1/src2
	public static Mat divide(Mat src1, Mat src2) {
		logi("Reflection Reduction::: divide");
		Mat dst = new Mat();
		Core.divide(src1, src2, dst);
		logi("Reflection Reduction::: divide success");
		return dst;
	}

	public static Mat multiply(Mat src1, Mat src2) {
		Mat dst = new Mat();
		Core.multiply(src1, src2, dst);
		return dst;
	}
	
	
	public static Mat subtract(Mat src1, Mat src2) {
		Mat dst = new Mat();
		Core.subtract(src1, src2, dst);
		return dst;
	}
	
	// Invert Mask
	// Ex: 1  1  0       0  0  1
	//	   0  1  0  -->  1  0  1
	//     1  1  1       0  0  0
	public static Mat invertMask(Mat mask) {
		Mat subM1 = new Mat(mask.size(), mask.type(), new Scalar(255));
		return subtract(subM1, mask);

	}
	

	public static Mat Canny(Mat image, double threshold1, double threshold2) {
		Mat edges = new Mat();
		Imgproc.Canny(image, edges, threshold1, threshold2);
	    return edges;
	}
	
	

	
	
	// Convert RGBA to RGB 
		public static Mat rgb2Rgba(Mat src) {
			Mat dst = new Mat();
			Imgproc.cvtColor(src, dst, Imgproc.COLOR_RGB2RGBA, 4);
			return dst;
		}
	
	
	// Convert RGBA to RGB 
	public static Mat rgba2Rgb(Mat src) {
		Mat dst = new Mat();
		Imgproc.cvtColor(src, dst, Imgproc.COLOR_RGBA2RGB, 3);
		return dst;
	}
	
	
	// Convert RGBA to GRAY 
	public static Mat rgb2Gray1C(Mat src) {
		Mat dst = new Mat();
		Imgproc.cvtColor(src, dst, Imgproc.COLOR_RGBA2GRAY, 1);
		return dst;
	}
	
	// Convert RGB to HSV 
		public static Mat rgb2hsv(Mat src) {
			Mat dst = new Mat();
			Imgproc.cvtColor(src, dst, Imgproc.COLOR_RGB2Luv, src.channels());
			return dst;
		}
	
		
	public static Mat getMaskFromCountour(Mat src) {
		List<Mat> mv = new ArrayList<Mat>();
		Core.split(src, mv);
		return mv.get(0);
	}
		
	
	// Bitmap to Mat
	public static Mat bitmapToMat(Bitmap bmp, boolean unPremultiplyAlpha) {
		Mat mat = new Mat();
		Utils.bitmapToMat(bmp, mat, unPremultiplyAlpha);
		return mat;
	}
	
	// Byte to Mat
	public static Mat byte2Mat(byte[] src) {
		logi("Convert Byte to Mat");
		Mat result = new Mat();
		Utils.bitmapToMat(Bytes2Bimap(src), result);
		logi("Convert Byte to Mat successfully");
		return result;
	}
	
	//Byte[] to Bitmap
  	public static Bitmap Bytes2Bimap(byte[] b) {  
          if(b.length!=0){  
              return BitmapFactory.decodeByteArray(b, 0, b.length);  
          }  
          else {  
              return null;  
          }  
  	}
	
  	//Bitmap to Byte[]
  	public static byte[] Bitmap2Bytes(Bitmap bm) {  
  	    ByteArrayOutputStream baos = new ByteArrayOutputStream();    
  	    bm.compress(Bitmap.CompressFormat.PNG, 100, baos);    
  	    return baos.toByteArray();  
  	}
 
  	
  	// Mat to Bitmap
  	public static Bitmap mat2Bitmap(Mat inputMat) {
  		Bitmap outBitmap = Bitmap.createBitmap(inputMat.cols(), inputMat.rows(), Bitmap.Config.ARGB_8888);
		Utils.matToBitmap(inputMat, outBitmap);
  		return outBitmap;
  	}
  	
  	// Return true if Homography transformation matrix is good
  	// Return false if Homography transformation matrix is bad
  	public static boolean goodHomography(Mat m) {
  		double det = m.get(0, 0)[0] * m.get(1, 1)[0] - m.get(1, 0)[0] * m.get(0, 1)[0];
  		if (det < 0) {
  			return false;
  		}
  		
  		double N1 = Math.sqrt(m.get(0, 0)[0] * m.get(0, 0)[0] - m.get(1, 0)[0] * m.get(1, 0)[0]);
  		if (N1 > 4 || N1 < 0.1) {
  			return false;
  		}
  		
  		double N2 = Math.sqrt(m.get(0, 1)[0] * m.get(0, 1)[0] - m.get(1, 1)[0] * m.get(1, 1)[0]);
  		if (N2 > 4 || N2 < 0.1) {
  			return false;
  		}
  		
  		double N3 = Math.sqrt(m.get(2, 0)[0] * m.get(2, 0)[0] - m.get(2, 1)[0] * m.get(2, 1)[0]);
  		if (N3 > 0.002) {
  			return false;
  		}
  		return true;
  	}
  	
	// count brightness value from RGB values
	public static synchronized double rgb2Gray(double[] data){
    	return data[0]*0.299 + data[1]*0.587 + data[2]*0.114;
    }
	
	/*
	 * Transform the given Mat to bitmap and show it in the given ImageView
	 * log the info
	 */
	private static void showImage(Mat inputMat, ImageView view, String info) {
		// convert result to bitmap
		logi("HomoTrans::: " + info + " :: get bitmap from matrix");
		Bitmap outBitmap = Bitmap.createBitmap(inputMat.cols(), inputMat.rows(), Bitmap.Config.ARGB_8888);
		Utils.matToBitmap(inputMat, outBitmap);
		logi("HomoTrans::: " + info + " :: get bitmap from matrix success");
				
				
		logi("HomoTrans::: " + info + " :: show image in " + view.getId() + " ImageView");
		view.setImageBitmap(outBitmap);
		logi("HomoTrans::: " + info + " :: show image success");
		////////////	
	}
	
	private static void logi(String str) {
		Log.i(TAG, "%%%%%%%%%%%%%%%" + str);
	}

}




//logi("ReflReduction::: subsitution:: contourInfoMap ");
//logi(contourInfoMap.toString());

//// no segmentation available
//if (contourInfoMap.size() == 0) {
//	logi("no segmentation available");
//	reducResultMat = getResultbyDamp(queryMat, homoResultMat, morDiffMap);
//
////	Mat[] result = reflReduction(queryMat, homoResultMat);
////	reducResultMat= result[0];
//
////	showImage(morDiffMap, tran2ImageView, "show morphological differnce map");
//}else {
//	boolean allRefOutOfObject = true;
//	queryMat.copyTo(reducResultMat);
//	for (int k : contourInfoMap.keySet()) {
//		
//		// occupied proportion
//		double pass = (double) contourInfoMap.get(k)[0] / contourInfoMap.get(k)[1]; 
//		logi("k = " + k + ", ref pixel= " + contourInfoMap.get(k)[0] + ", total area= " + contourInfoMap.get(k)[1] + ", occupied proportion= " + pass);
//
//		if (pass > pptlThreshold) {
//			logi("object replacement, taken k:" + k);
//			subMask = getMaskFromCountour(contourMap.get(k));
//			finalMask = max(contourMap.get(k), finalMask);///////////
//			homoResultMat.copyTo(reducResultMat, subMask);
//			allRefOutOfObject = false;
//		// reduce such area by diffMap only
//		}else if(contourInfoMap.get(k)[0] > 0){
//			logi("in object pixel replacement, taken k:" + k);
//			subMask = getMaskFromCountour(contourMap.get(k));
//			finalMask = max(contourMap.get(k), finalMask);///////////
//			Mat minMask = min(subMask, morDiffMap);
//			minQuery.copyTo(reducResultMat, minMask);
//			allRefOutOfObject = false;
//		}
//	}
////	showImage(finalMask, tran2ImageView, "show final mask");	
//
//	
//}
//logi("ReflReduction::: subsitution success");
