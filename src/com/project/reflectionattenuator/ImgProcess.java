 package com.project.reflectionattenuator;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.SeekBar;
import android.widget.SeekBar.OnSeekBarChangeListener;
import android.widget.TextView;
import android.widget.Toast;



public class ImgProcess extends Activity implements View.OnClickListener, OnSeekBarChangeListener{
	
	private Button stereoTransButton;
	private Button reflectionRednButton;
	private Button saveImgButton;
	private Button resetButton;
	private Button testButton;
	private TextView seekBarValue;  
	private SeekBar seekBar;
	private ImageView ori1ImageView;
    private ImageView ori2ImageView;
    private ImageView tran1ImageView;
    private ImageView tran2ImageView;
    private ImageView resultImageView;
    private RadioGroup checkRadioGroup;
    private RadioButton crossCheckRadio;
    private RadioButton distanceCheckRadio;
    private RadioButton localCheckRadio;
    
    private List<byte[]> byteArrayList;
    private static final String TAG = "ImgProcActivity";
    private int imgHeight;
    private int imgWidth;
    private int transCount;
    private int sizeSE;
    private Mat trainMat;
	private Mat queryMat;
	private Mat homoTrainMat;
	private Mat relfReducResult;
	private Mat saveDifMap;
	private Mat savefinalMask;
    private FeatureDetector featureDetector;
    private DescriptorExtractor descriptorExtractor;
    private DescriptorMatcher descriptorMatcher;
    private Bitmap reducRefResult;
    private boolean crossCheck;
    private boolean distanceCheck;
    private boolean localCheck;
    private boolean isTransformed;
   
    private static final int SEEKBARMAX = 50;
    private static final int SEEKBARINITIALVALUE = 10;
    private static final int KNNNUMBER = 2;
    private static final int NUMOFCHESSBOARD = 24; // local check blocks
    public static final int MEDIA_TYPE_IMAGE = 1;
    
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.img_process);
        initialization();
        findViews();
        showResult();
        setListensers();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.img_process, menu);
        return true;
    }
    
    private void initialization() {
    	logi("initialization");
    	byteArrayList = new ArrayList<byte[]>();
        featureDetector = FeatureDetector.create(FeatureDetector.ORB);
        descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.BRIEF);
        descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_L1);//// Flann-based descriptor
        transCount = 0;
        sizeSE = SEEKBARINITIALVALUE;
        trainMat = new Mat();
    	queryMat = new Mat();
        crossCheck = false;
        distanceCheck = false;
        localCheck = false;
        isTransformed = false;
        logi("initialization success");
    }
    
    private void findViews() {
    	logi("findViews");
    	stereoTransButton = (Button) findViewById(R.id.button_trans);
    	reflectionRednButton = (Button) findViewById(R.id.button_redn);
    	saveImgButton = (Button) findViewById(R.id.button_saveImg);
    	resetButton = (Button) findViewById(R.id.button_reset);
    	testButton = (Button) findViewById(R.id.button_testing);
    	seekBarValue = (TextView)findViewById(R.id.seekBarValueText);
    	ori1ImageView = (ImageView) findViewById(R.id.imageView_ori1);
    	ori2ImageView = (ImageView) findViewById(R.id.imageView_ori2);
    	tran1ImageView = (ImageView) findViewById(R.id.imageView_tran1);
    	tran2ImageView = (ImageView) findViewById(R.id.imageView_tran2);
    	resultImageView = (ImageView) findViewById(R.id.imageView_result);
    	seekBar = (SeekBar) findViewById(R.id.seekBar);
    	checkRadioGroup = (RadioGroup) findViewById(R.id.RadioGroup_matchesCheck);
    	crossCheckRadio = (RadioButton) findViewById(R.id.radio_crossCheck);
    	distanceCheckRadio = (RadioButton) findViewById(R.id.radio_distanceCheck); 
    	localCheckRadio = (RadioButton) findViewById(R.id.radio_localCheck); 
    	seekBar.setMax(SEEKBARMAX);
    	seekBar.setProgress(SEEKBARINITIALVALUE);
    	logi("findViews success");
    }
    
    
    private void getBundledImage() {
    	logi("Read bundle");    	
    	ImagePackage mPackage = (ImagePackage) this.getIntent().getSerializableExtra(MainActivity.SER_KEY); 
    	trainMat = mPackage.getImage1Mat();
    	queryMat = mPackage.getImage2Mat();
    	
//    	trainMat = CVTool.byte2Mat(mPackage.getImage1Byte());
//    	queryMat = CVTool.byte2Mat(mPackage.getImage2Byte());
    	
    	logi("Read bundle success"); 
    	imgHeight = trainMat.height();
    	imgWidth = trainMat.width();
    }
    
    
    private void oldGetBundledImage() {
    	logi("Read bundle");
    	Bundle bunde = this.getIntent().getExtras();
    	// get trainMat and queryMat
    	trainMat = CVTool.byte2Mat(bunde.getByteArray("imageData1"));
    	queryMat = CVTool.byte2Mat(bunde.getByteArray("imageData2"));
 
    	logi("Read bundle success"); 
    	//preview reference
    	imgHeight = trainMat.height();
    	imgWidth = trainMat.width();
    }
    
    // Show image in ImageView and initialize trainMat and queryMat
    private void showResult() {
    	// get image from bundle
    	getBundledImage();
    	// preview images
    	
//    	logi("show::: test");
//    	logi("show::: test::: channel" + queryMat.channels());
    	logi("show::: test::: r" + trainMat.size().toString());
//    	
//    	logi("show::: test::: r" + queryMat.get(0, 0)[0]);
//    	logi("show::: test::: g" + queryMat.get(0, 0)[1]);
//    	logi("show::: test::: b" + queryMat.get(0, 0)[2]);
//    	logi("show::: test::: a" + queryMat.get(0, 0)[3]);
//    	
    	
    	Size size = new Size(324, 243);
    	
//    	trainMat = CVTool.pyrDown(trainMat, size);
    	
    	logi("show::: test::: r" + trainMat.size().toString());
    	
    	showImage(trainMat, ori1ImageView, "Show Train Image");
    	showImage(queryMat, ori2ImageView, "Show Train Image");
    	logi("Read bundle succesffully");
    }
    
    private void setListensers() {
    	logi("setListensers");
    	stereoTransButton.setOnClickListener(this);
    	reflectionRednButton.setOnClickListener(this);
    	saveImgButton.setOnClickListener(this);
    	resetButton.setOnClickListener(this);
    	testButton.setOnClickListener(this);
    	seekBar.setOnSeekBarChangeListener(this);
    	checkRadioGroup.setOnCheckedChangeListener(changeRadio);
    	logi("setListensers success");
    }
        
    // Button of homographic transformation
    private void btn_trans(){
    	if (transCount == 0) {
    		logi("count=0");
    		homoTrainMat = homoTransFormation(queryMat, trainMat);
    	}else {
    		logi("count!=0");
    		homoTrainMat = homoTransFormation(queryMat, homoTrainMat);
    	}
    	transCount++;
    }
    
    // Button of reflection reduction
    private void btn_reduction() {
    	relfReducResult = reflcReduction(homoTrainMat, queryMat);
    }    
    
    /**
     * 
     * @param queryMat: the Mat used to be transformation reference 
     * @param trainMat: the Mat required to be transformed
     * @return homographic trainMat
     */
    		
    private Mat homoTransFormation(Mat queryMat, Mat trainMat) {
    	logi("Homographic transformation starts");
    	
    	// Equalization
    	logi("HomoTrans::: Equlization");    	
    	Mat queryMatEq = CVTool.equalizeMat(queryMat);
    	Mat trainMatEq = CVTool.equalizeMat(trainMat);
//    	showImage(queryMat, tran2ImageView, "show eqQueryMat");
    	logi("HomoTrans::: Equlization success");
    	
    	
//    	logi("HomoTrans::: Calculate histogram");   
//    	Mat histoQuery = CVTool.calcHist(CVTool.GussianBlur(CVTool.pyrDown(queryMat, new Size(324, 243)), new Size(9, 9), 2, 2), false);
//    	Mat histoTrain = CVTool.calcHist(CVTool.GussianBlur(CVTool.pyrDown(trainMat, new Size(324, 243)), new Size(9, 9), 2, 2), false);
//    	logi("HomoTrans::: Calculate histogram success");   
//    	
//    	logi("HomoTrans::: Calculate acumulate histogram");   
//    	Mat acumHistoQuery = CVTool.calcAcumHist(histoQuery);
//    	Mat acumHistoTrain = CVTool.calcAcumHist(histoTrain);
//    	logi("HomoTrans::: Calculate acumulate histogram success"); 
//    	
//    	logi("HomoTrans::: Calcualte transform values"); 
//    	Mat transMat = CVTool.getTransformValue(acumHistoTrain, acumHistoQuery);
//    	logi("HomoTrans::: Calcualte transform values success"); 
//    	
//    	
//    	logi("HomoTrans::: Histogram Specification");
//    	Mat newTrainMat = CVTool.histogramSpecificaiton(trainMat, queryMat, transMat);
////    	trainMat = newTrainMat;
////    	trainMatEq = newTrainMat;
////    	queryMatEq = queryMat;
//    	showImage(newTrainMat, tran2ImageView, "show training image");
//    	logi("HomoTrans::: Histogram Specification success");   
    	
    	
    	// Find List of Mat of key points
    	logi("HomoTrans::: find keypoints");
    	MatOfKeyPoint trainMatOfKeyPoint = CVTool.findKeyPoints(featureDetector, trainMatEq);
    	MatOfKeyPoint queryMatOfKeyPoint = CVTool.findKeyPoints(featureDetector, queryMatEq);
    	logi("HomoTrans::: find keypoints success");
    	logi("===================");
    	
    	// Find descriptor
    	logi("HomoTrans::: find descriptor");
    	Mat trainDescriptors = CVTool.computeDescriptors(trainMatEq, trainMatOfKeyPoint, descriptorExtractor);
    	Mat queryDescriptors = CVTool.computeDescriptors(queryMatEq, queryMatOfKeyPoint, descriptorExtractor);
    	logi("HomoTrans::: find descriptor success");
    	logi("===================");
    	
    	// Find match
    	logi("HomoTrans::: find match :: query to train");
    	MatOfDMatch matches12 = CVTool.getMatchingCorrespondences(queryDescriptors, trainDescriptors, descriptorMatcher);
    	logi("HomoTrans::: find match :: train to query");
    	MatOfDMatch matches21 = CVTool.getMatchingCorrespondences(trainDescriptors, queryDescriptors, descriptorMatcher);
    	logi("HomoTrans::: find match success");
    	logi("===================");
    	
 
    	// Select matches approaches 
    	MatOfDMatch finalMatches;
    	if (crossCheck == true && distanceCheck == false && localCheck == false) {
    		// Get MatOfDMatch by Cross Check
        	logi("HomoTrans::: CrossCheck");
        	finalMatches = CVTool.getCrossMatches(matches12, matches21, trainMatOfKeyPoint, queryMatOfKeyPoint);
        	logi("HomoTrans::: CrossCheck success");
        	logi("===================");
    		
    		toasti("ORB detection + cross check!");
    	}else if (crossCheck == false && distanceCheck == false && localCheck == true) {
    		// Get Local maximum matches
        	logi("HomoTrans::: Local Maximum Matches");
        	MatOfDMatch crossMatches = CVTool.getCrossMatches(matches12, matches21, trainMatOfKeyPoint, queryMatOfKeyPoint);
        	finalMatches = CVTool.getLocalMatches(crossMatches, trainMatOfKeyPoint, queryMatOfKeyPoint, NUMOFCHESSBOARD, imgHeight, imgWidth);
        	logi("HomoTrans::: Local Maximum Matches success");
        	logi("===================");
    		toasti("ORB detection + cross check + chessboard sampling, chessboard: " + NUMOFCHESSBOARD + " x " + NUMOFCHESSBOARD);
    	}else if (crossCheck == false && distanceCheck == true && localCheck == false) {
 
    		// Find knnMatch   
    		logi("HomoTrans::: knnMatch");
        	List<MatOfDMatch> knnMatchesList = CVTool.getKnnMatchList(queryDescriptors, trainDescriptors, KNNNUMBER, descriptorMatcher);
        	logi("HomoTrans::: knnMatch success");
        	logi("===================");
        	
        	// Get MatOfDMatch by Distance Check
        	logi("HomoTrans::: DistanceCheck");
        	int disThreshold = 80;
        	finalMatches = CVTool.getDistanceMatches(knnMatchesList, trainMatOfKeyPoint, queryMatOfKeyPoint, KNNNUMBER, disThreshold);
        	logi("HomoTrans::: DistanceCheck success");
        	logi("===================");
    		toasti("ORB detection + distant-different check, threshold= " + disThreshold);
    		
    	}else {
    		finalMatches = matches12;
    		toasti("ORB detection only!");
    	}
    	    	
    	// Find match point array
    	logi("HomoTrans::: find match point array");
    	MatOfPoint2f[] MatPt2fArray = CVTool.getCorrespondences(finalMatches, trainMatOfKeyPoint, queryMatOfKeyPoint);
    	logi("HomoTrans::: find match point array success");
    	logi("===================");
		
    	// Get Mat of Point
    	logi("HomoTrans::: get match point");
		MatOfPoint2f trainMatPt = MatPt2fArray[0];
		MatOfPoint2f queryMatPt = MatPt2fArray[1];
		logi("HomoTrans::: get match point success");
		logi("===================");
		
		// Get fundamental matrix
//		logi("HomoTrans::: get fundamental matrix");
//		Mat fundametalMatrix = CVTool.findFundamentalMat(trainMatPt, queryMatPt, Calib3d.FM_7POINT, 3, 0.99);
//		logi("HomoTrans::: get fundamental matrix success");
//		logi("===================");
		
		// Find Homographic Transformation Matrix  m
		logi("HomoTrans::: get Homography matrix");
		logi("HomoTrans::: get Homography matrix:: trainMatpt size: " + trainMatPt.size().toString());
		logi("HomoTrans::: get Homography matrix:: queryMatPt size: " + queryMatPt.size().toString());
		// Calib3d.CV_LMEDS 
		Mat m = CVTool.findHomography(trainMatPt, queryMatPt, Calib3d.RANSAC, 8);
		logi("HomoTrans::: get Homography matrix channel: " + m.channels());
		logi("HomoTrans::: get Homography matrix size: " + m.size() + "success");
		logi("HomoTrans::: Good Homography? " + CVTool.goodHomography(m));
		
		logi("===================");
		
		
		
		
		
		// Do Homographic Transformation
		logi("HomoTrans::: Do Homographic transfrom");
		boolean invert = false;
		Mat homoResultMat = CVTool.getWarpedImage(trainMat, m, false);
		logi("HomoTrans::: Do Homographic transfrom success");
		logi("===================");
		
		
		
		
		
		// Show homographic training image in image view
		showImage(homoResultMat, tran1ImageView, "show training image");
		logi("===================");
		
		
		
		
		
		// Get Mat of Match line 
		logi("HomoTrans::: find match line mat");
		Mat matchLineMat = CVTool.getMatchingDraw(trainMat, trainMatOfKeyPoint , queryMat, queryMatOfKeyPoint, finalMatches);
		// Draw Match Lines on Images
		Mat matchLineMatResult = CVTool.getMatchingDrawOnImage(trainMat, queryMat, matchLineMat);
		logi("HomoTrans::: find match line mat success");
		logi("===================");

		// Show matchLine image in imageView
		showImage(matchLineMatResult, resultImageView, "show matchlines");
		logi("HomoTrans::: draw match line mat success");
		logi("===================");
		///////////////////
		
		return homoResultMat;
    }
    
    


	/**
 	 * 
 	 * @param homoTrainMat: the Mat used to reduce reflection
 	 * @param queryMat: the Mat with reflection need to be operated reflection reduction
 	 * @return reflection reduced Mat
 	 */
    private Mat reflcReduction(Mat homoTrainMat, Mat queryMat) {
    	logi("ReflReduction::: Do reflection Reduction");
    	Mat[] reflReducResult = CVTool.reflReduction(queryMat, homoTrainMat);
    	Mat finalResult = reflReducResult[0];
    	logi("ReflReduction::: Do reflection Reduction Success");

//    	logi("ReflReduction::: Mean Shift");
//    	Mat queryMatRef = reflReducResult[1];
//    	Mat trainHomoMatRef = reflReducResult[2];
//    	Mat queryMatMeanShift = CVTool.getMeanShift(queryMatRef, 10, 20);
//    	Mat trainMatMeanShift = CVTool.getMeanShift(trainHomoMatRef, 10, 20);
//    	showImage(queryMatRef, tran2ImageView, "show query reflection img");
//		logi("===================");
//    	logi("ReflReduction::: Mean Shift success");
//    	logi("===================");
    	
    	
    	// Show Result
    	// show result to bitmap
//    	logi("ReflReduction::: get bitmap from matrix");
//    	reducRefResult = Bitmap.createBitmap(finalResult.cols(), finalResult.rows(), Bitmap.Config.ARGB_8888);
//    	Utils.matToBitmap(finalResult, reducRefResult);
//    	logi("ReflReduction::: get bitmap from matrix success");
//
//
//    	logi("ReflReduction::: show result");
//    	resultImageView.setImageBitmap(reducRefResult);
//    	logi("ReflReduction::: show result success");
//    	logi("ReflReduction success!!");
    	
    	showImage(finalResult, resultImageView, "show final image");
    	
    	return finalResult;
    	///////////////////
    }
    
	
	@Override
	public void onClick(View v) {
		// TODO Auto-generated method stub
		logi("onClick!");
		// TODO Auto-generated method stub
		switch (v.getId()) {
		case R.id.button_trans:
			logi("Homographic Transform");
			btn_trans();
			isTransformed = true;
			break;

		case R.id.button_redn:
			logi("Color testing");
			if (isTransformed) {
				btn_reduction();
			}else {
				toasti("Homographic transformation is required first!!");
			}
			break;
		case R.id.button_saveImg:
			if (relfReducResult != null && queryMat != null && homoTrainMat != null) {
				logi("Save result");
				toasti("image saved!");
				saveImg(CVTool.mat2Bitmap(relfReducResult), "reducResult");
				saveImg(CVTool.mat2Bitmap(trainMat), "trainImg");
				saveImg(CVTool.mat2Bitmap(queryMat), "queryImg");
				saveImg(CVTool.mat2Bitmap(homoTrainMat), "otherHomoImg");
				if (saveDifMap != null && savefinalMask != null) {
					saveImg(CVTool.mat2Bitmap(saveDifMap), "DifMap");
					saveImg(CVTool.mat2Bitmap(savefinalMask), "finalMask");
				}
			}else {
				logi("null image!");
				toasti("null image!");
			}
			
			break;
			
		case R.id.button_reset:
			logi("Reset transformation count");
			transCount = 0;
			break;
		case R.id.button_testing:
			logi("Test algorithm");
			if (isTransformed) {
				btn_testing();
			}else {
				toasti("Homographic transformation is required first!!");
			}
			break;
		}
	}
	
	
	@Override
	public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
		// TODO Auto-generated method stub
		sizeSE = progress + 1;
//		logi("SeekBar value= " + (String.valueOf(progress) + 1));
		seekBarValue.setText("SeekBar value= " + String.valueOf(progress + 1)); 
	}

	@Override
	public void onStartTrackingTouch(SeekBar seekBar) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void onStopTrackingTouch(SeekBar seekBar) {
		// TODO Auto-generated method stub
		
	}
	
	// Radio Button for check options
	private RadioGroup.OnCheckedChangeListener changeRadio = new RadioGroup.OnCheckedChangeListener(){
		@Override
        public void onCheckedChanged(RadioGroup group, int checkedId) {
        	switch (checkedId) {
      		case R.id.radio_crossCheck:
      			logi("Apply cross check");
      			crossCheck = true;
      	        distanceCheck = false;
      	        localCheck = false;
      			break;
      			
      		case R.id.radio_distanceCheck:
      			logi("Apply distance check");
      			crossCheck = false;
      	        distanceCheck = true;
      	        localCheck = false;
      			break;
      			
      		case R.id.radio_localCheck:
      			logi("Apply local check");
      			crossCheck = false;
      	        distanceCheck = false;
      	        localCheck = true;
      			break;
      		
      		}	
         }
      };
	 
	
	// save image
	private void saveImg(Bitmap Img, String fileTag) {
		if (Img != null) {
			File pictureFile = getOutputMediaFile(MEDIA_TYPE_IMAGE, fileTag);
        	if (pictureFile == null){
            	Log.d(TAG, "Error creating media file, check storage permissions: ");
            	return;
        	}
        	try {
        		logi("Save Image start");
        		BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(pictureFile));
        		/* compress the image */
            	Img.compress(Bitmap.CompressFormat.JPEG, 90, bos);
            
            	/* release the BufferStream */
            	bos.flush();
            
            	/* close OutputStream */
            	bos.close();
            	logi("Save Image success");
            	toasti("Image has been saved");
        	} catch (FileNotFoundException e) {
            	Log.d(TAG, "File not found: " + e.getMessage());
        	} catch (IOException e) {
            	Log.d(TAG, "Error accessing file: " + e.getMessage());
        	}
		} else {
			toasti("No result yet");
		}
	}
	
	
	/** Create a File for saving an image or video */
	private static File getOutputMediaFile(int type, String fileTag){
	    // To be safe, you should check that the SDCard is mounted
	    // using Environment.getExternalStorageState() before doing this.
		
	    File mediaStorageDir = new File(Environment.getExternalStoragePublicDirectory(
	              Environment.DIRECTORY_PICTURES), "ReflectionAttenuator");
	    
	    if (! mediaStorageDir.exists()){
	        if (! mediaStorageDir.mkdirs()){
	            Log.d("ReflectionAttenuator", "failed to create directory");
	            return null;
	        }
	    }

	    // Create a media file name
	    String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
	    String fileName = timeStamp + "_" + fileTag;
	    File mediaFile;
	    if (type == MEDIA_TYPE_IMAGE){
	        mediaFile = new File(mediaStorageDir.getPath() + File.separator +
	        "IMG_"+ fileName + ".jpg");
	    } else {
	        return null;
	    }

	    return mediaFile;
	}
	
	/*
	 * Transform the given Mat to bitmap and show it in the given ImageView
	 * log the info
	 */
	private void showImage(Mat inputMat, ImageView view, String info) {
		// show result to bitmap
		logi("HomoTrans::: " + info + " :: get bitmap from matrix");
		Bitmap outBitmap = Bitmap.createBitmap(inputMat.cols(), inputMat.rows(), Bitmap.Config.ARGB_8888);
		Utils.matToBitmap(inputMat, outBitmap);
		logi("HomoTrans::: " + info + " :: get bitmap from matrix success");
				
				
		logi("HomoTrans::: " + info + " :: show image in " + view.getId() + " ImageView");
		view.setImageBitmap(outBitmap);
		logi("HomoTrans::: " + info + " :: show image success");
		////////////	
	}
	
	
	private void logi(String str) {
		Log.i(TAG, "%%%%%%%%%%%%%%%" + str);
	}
	
	private void toasti(String str) {
		Toast.makeText(ImgProcess.this, str, Toast.LENGTH_SHORT).show();
	}
		
	//Bitmap to Byte[]
  	private byte[] Bitmap2Bytes(Bitmap bm) {  
  	    ByteArrayOutputStream baos = new ByteArrayOutputStream();    
  	    bm.compress(Bitmap.CompressFormat.PNG, 100, baos);    
  	    return baos.toByteArray();  
  	}
  	
  	//Byte[] to Bitmap
  	private Bitmap Bytes2Bimap(byte[] b) {  
          if(b.length!=0){  
              return BitmapFactory.decodeByteArray(b, 0, b.length);  
          }  
          else {  
              return null;  
          }  
  	}
  	
  	// Mat to Bitmap
  	private Bitmap mat2Bitmap(Mat inputMat) {
  		Bitmap outBitmap = Bitmap.createBitmap(inputMat.cols(), inputMat.rows(), Bitmap.Config.ARGB_8888);
		Utils.matToBitmap(inputMat, outBitmap);
  		return outBitmap;
  	}
  	
  	// Byte to Mat
  	private Mat byte2Mat(byte[] src) {
     	logi("Convert Byte to Mat");
     	Mat result = new Mat();
     	Utils.bitmapToMat(Bytes2Bimap(src), result);
     	logi("Convert Byte to Mat successfully");
     	return result;
    }

  	
  	private void btn_testing() {
//    	Mat testResult = reflcReductionTesting(homoTrainMat, queryMat);
  		relfReducResult = reflcReductionTesting(homoTrainMat, queryMat);
    }
    
  	
    private Mat reflcReductionTesting(Mat homoTrainMat, Mat queryMat) {
    	
    	Size size = new Size(9, 9);
    	double sigmaX = 2;
    	double sigmaY = sigmaX;
    	
    
    	logi("ReflReduction::: find different map");
    	Mat[] diffMapArray = CVTool.getDifferenceMap(queryMat, homoTrainMat, size, sizeSE, sigmaX, sigmaY, 1.1, 4);
    	logi("ReflReduction::: find different map success");
    	
//    	showImage(diffMapArray[1], tran1ImageView, "show final image");
//    	showImage(diffMapArray[0], tran2ImageView, "show final image");
    
    	
    	logi("ReflReduction::: Do segmentation");
    	long startTime1 = System.currentTimeMillis();
//    	Mat segMat = CVTool.getSegmentation(homoTrainMat, queryMat, 5, 15, 1, tran1ImageView, tran2ImageView, resultImageView);
    	Mat segMat = CVTool.getSegmentation(homoTrainMat, queryMat, 5, 5, 5, tran1ImageView, tran2ImageView, resultImageView);
    														//  iterate = 5, SE size to remove little segment, padding
    	
    	long endTime1 = System.currentTimeMillis();
		logi("segMat TIME1 = " + (endTime1 - startTime1));
    	logi("ReflReduction::: Do segmentation success");
    
    	
    	logi("ReflReduction::: Do reflection Reduction");
    	Mat[] reflReducResult = CVTool.reflReductionTest(queryMat, homoTrainMat, segMat, diffMapArray, 0.15, true);
    	
    	Mat finalMask = reflReducResult[2];
    	Mat morDiffMap = reflReducResult[1];
    	Mat finalResult = reflReducResult[0];
    	logi("ReflReduction::: Do reflection Reduction Success");


    	
    	showImage(morDiffMap, tran1ImageView, "show final image");
    	showImage(finalMask, tran2ImageView, "show final image");
//    	showImage(finalMask, resultImageView, "show final image");
    	showImage(finalResult, resultImageView, "show final image");
    	
    	saveDifMap = morDiffMap;
    	savefinalMask = finalMask;

    	return finalResult;
    	///////////////////
//    	return null;
    }
  	
    
//    /**
//     * 
//     * @param homoTrainMat
//     * @param queryMat
//     * @param iterate, numbers of looping to find contours with different contrast. Defaut = 3
//     * @param sizeSE, structuring element size for opening morphology
//     * @return
//     */
//    public synchronized Mat getSegmentation (Mat homoTrainMat, Mat queryMat, int iterate, int sizeSE) {
//
//    	Mat finalContourResult = Mat.zeros(queryMat.rows(), queryMat.cols(), CvType.CV_8UC3);
//    	Mat queryContourResult = Mat.zeros(queryMat.rows(), queryMat.cols(), CvType.CV_8UC3);
//    	Mat trainContourResult = Mat.zeros(homoTrainMat.rows(), homoTrainMat.cols(), CvType.CV_8UC3);
//    	Mat queryConMat = Mat.zeros(queryMat.size(), queryMat.type());
//    	Mat trainConMat = Mat.zeros(homoTrainMat.size(), homoTrainMat.type());
//    	
//    	// find contours and draw contours of query and train img
//    	for (int j = 0; j < iterate; j++) {
//
//    		queryMat.convertTo(queryConMat, -1, 1.6 + 0.1*j, 10);
//    		homoTrainMat.convertTo(trainConMat, -1, 1.6 + 0.1*j, 10);
//
//
//    		Mat cannyQuery = CVTool.Canny(queryConMat, 250 - 30 * j, 350 - 30 * j);
//    		Mat cannyTrain = CVTool.Canny(trainConMat, 250 - 30 * j, 350 - 30 * j);
//
//    		// RETR_CCOMP
//    		// CV_RETR_EXTERNAL
//    		// CV_RETR_TREE
//    		Mat hierarchy = new Mat();
//    		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
//    		Imgproc.findContours(cannyQuery, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
//
//    		for (int i = 0; i <  hierarchy.size().width; i++) {
//    			if (hierarchy.get(0, i)[0] == -1 && hierarchy.get(0, i)[1] == -1) {
//    				Mat countourResult = CVTool.drawContours(cannyQuery, contours, i, new Scalar(255, 0, 0), Core.FILLED, Core.LINE_8, hierarchy, 1, new Point(0,0));
//    				queryContourResult = CVTool.max(queryContourResult, countourResult);
//    			}
//    		}
//
//    		
//    		hierarchy = new Mat();
//    		contours = new ArrayList<MatOfPoint>();
//    		Imgproc.findContours(cannyTrain, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
//
//    		for (int i = 0; i <  hierarchy.size().width; i++) {
//    			if (hierarchy.get(0, i)[0] == -1 && hierarchy.get(0, i)[1] == -1) {
//    				Mat countourResult = CVTool.drawContours(cannyTrain, contours, i, new Scalar(0, 255, 0), Core.FILLED, Core.LINE_8, hierarchy, 1, new Point(0,0));
//    				trainContourResult = CVTool.max(trainContourResult, countourResult);
//    			}
//    		}
//    	}
//		
//    	
//    	Mat SE = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(sizeSE, sizeSE), new Point(sizeSE / 2, sizeSE / 2));
//    	Mat morTrainContourResult = CVTool.morphologyEx(trainContourResult, Imgproc.MORPH_OPEN, SE, new Point(sizeSE / 2, sizeSE / 2), 1);
//    	Mat morQueryContourResult = CVTool.morphologyEx(queryContourResult, Imgproc.MORPH_OPEN, SE, new Point(sizeSE / 2, sizeSE / 2), 1);
//
//    	logi("Segmentation::: combine");
//    	finalContourResult = CVTool.combineRG21C(morTrainContourResult, morQueryContourResult);
//    	logi("Segmentation::: combine success");
//
//    	// make buffer at boulders 
//    	SE = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
//    	finalContourResult = CVTool.dilation(finalContourResult, SE, new Point(0, 0), 1);
//    	
//    	
//    	logi("Segmentation::: drawContour");
////    	int maxLevel = 100;
////    	Mat testResult = CVTool.drawContours(queryMat, contours, -1, Core.FILLED, Core.LINE_8, hierarchy, maxLevel, new Point(0,0));
////    	Mat testResult = CVTool.drawContours(queryMat, contours, -1, Core.FILLED);
//    	logi("Segmentation::: drawContour success");
//    	logi("/////////////////////");
//     	
//    	return finalContourResult;
//    }
    
    
    
//    /**
//     * 
//     * @param queryMat
//     * @param homoResultMat
//     * @param ksize, size of Gaussian blur matrix
//     * @param sigmaX, sigmaX for Gaussian blur matrix
//     * @param sigmaY, sigmaY for Gaussian blur matrix
//     * @param segMat, segment map
//     * @param threshold, threshold to define difference map
//     * @return Mat[0] = difference map without doing morphology
//     * @return Mat[1] = difference map with doing morphology
//     */
//    public synchronized Mat[] getDifferenceMap(Mat queryMat, Mat homoResultMat, Size ksize, double sigmaX, double sigmaY, double threshold) {
//    	logi("ReflReduction::: Creat reflection layer");
//	
//		Mat queryGaussianGrayMat =  CVTool.rgb2Gray1C(CVTool.GussianBlur(queryMat, ksize, sigmaX, sigmaY));
//		Mat homoResulGaussianGrayMat = CVTool.rgb2Gray1C(CVTool.GussianBlur(homoResultMat, ksize, sigmaX, sigmaY));
//
//
//		Mat diffMap = CVTool.divide(queryGaussianGrayMat, homoResulGaussianGrayMat);
//		logi("diffMap channels:" + diffMap.channels());
//		
//		Mat diffThresMap = CVTool.threshold(diffMap, threshold, 255);
//
//		
//
////		logi("test Reduce value = " + CVTool.getMatByType(diffMap, Core.REDUCE_MAX));
//
//
//		Mat SE = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(sizeSE, sizeSE), new Point(sizeSE / 2, sizeSE / 2));
//		Mat morDiffMap;
////		morDiffMap = CVTool.morphologyEx(diffThresMap, Imgproc.MORPH_OPEN, SE, new Point(sizeSE / 2, sizeSE / 2), 1);
//		morDiffMap = CVTool.morphologyEx(diffThresMap, Imgproc.MORPH_CLOSE, SE, new Point(sizeSE / 2, sizeSE / 2), 1);
////		morDiffMap = CVTool.erosion(diffThresMap, SE, new Point(10, 10), 1);
//		
//		showImage(morDiffMap, tran2ImageView, "show gaussian query gray image");
////		SE = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(20, 20), new Point(10, 10));
////		morDiffMap = CVTool.dilation(morDiffMap, SE, new Point(0, 0), 10);
//		morDiffMap = CVTool.morphologyEx(morDiffMap, Imgproc.MORPH_OPEN, SE, new Point(sizeSE / 2, sizeSE / 2), 1);
//    	
//    	Mat[] result = {diffMap, morDiffMap};
//    	return result;
//    }
    
    
    
//    public synchronized Mat[] reflReductionTest (Mat queryMat, Mat homoResultMat, Mat segMat, Mat[] diffMapArray) {
//
//		
//		Mat diffThresMap = diffMapArray[0];
//		Mat morDiffMap = diffMapArray[1];
//		Mat finalMask = Mat.zeros(queryMat.size(), CvType.CV_8UC3);
//		
//
//		logi("ReflReduction::: Re define segMat");
//		Queue<Coordinate> qPoint = new LinkedList<Coordinate>();			
//		
//		for (int i = 0; i < segMat.rows(); i++) {
//			for (int j = 0; j < segMat.cols(); j++) {
//				if (segMat.get(i, j)[0] > 0 && CVTool.isBlack(homoResultMat.get(i, j))) {
//					segMat.put(i, j, 0);
//				}
//				if (morDiffMap.get(i, j)[0] > 1 && segMat.get(i, j)[0] > 1) {
//					qPoint.add(new Coordinate(i,j));
//				}
//			}
//		}
//		logi("ReflReduction::: Re define segMat success");
//		
//		
//		logi("ReflReduction::: find countour");
//		Mat hierarchy = new Mat();
//		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
//		Imgproc.findContours(segMat, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
//		logi("ReflReduction::: find countour success");
//		
//		
//		
//		long startTime1 = System.currentTimeMillis();
//
//		logi("ReflReduction::: define finalMask");
//		Map<Integer, Integer[]> contourInfoMap= new HashMap<Integer, Integer[]>();
//		Map<Integer, Mat> contourMap= new HashMap<Integer, Mat>();
//		for (int k = 0; k < hierarchy.size().width; k++) {
//			
//			int maxLevel = 1;
//			Mat countourResult = CVTool.drawContours(segMat, contours, k, new Scalar(255, 0, 0), Core.FILLED, Core.LINE_8, hierarchy, maxLevel, new Point(0,0));
//			contourMap.put(k, countourResult);
//			
//			// {pNumber, area}
//			Integer[] temp = {0, CVTool.countourArea(contours.get(k))};
//			finalMask = CVTool.max(countourResult, finalMask);/////////
//			
//			int end = qPoint.size();
//			for (int i = 0; i < end; i++) {
//				Coordinate p = qPoint.remove();
//
//				if (countourResult.get(p.row, p.col)[0] > 0) {
//					temp[0] = temp[0] + 1;
//					double[] d = {0,0,255};
//					finalMask.put(p.row, p.col, d);/////////
//				}else {
//					qPoint.add(p);
//				}	
//			}
//			contourInfoMap.put(k, temp);
//		}
//		logi("ReflReduction::: define finalMask success");
//		
//		
//		long endTime1 = System.currentTimeMillis();
//		logi("TIME1 = " + (endTime1 - startTime1));
//		
//		
//		// reduce reflection by min
//		logi("ReflReduction::: subsitution");
//		Mat minQuery = CVTool.min(queryMat, homoResultMat);
//		//draw contour and substitute pixels
//		Mat subMask = new Mat();
//		Mat reducResultMat = new Mat();
//		
//		if (contourInfoMap.size() == 0) {
//			logi("no segmentation available");
//			reducResultMat = getResultbyDamp(queryMat, homoResultMat, morDiffMap);
//		
////			Mat[] result = CVTool.reflReduction(queryMat, homoResultMat);
////			reducResultMat= result[0];
//	
////			showImage(morDiffMap, tran2ImageView, "show morphological differnce map");
//		}else {
//			queryMat.copyTo(reducResultMat);
//			for (int k : contourInfoMap.keySet()) {
//				double pass = (double) contourInfoMap.get(k)[0] / contourInfoMap.get(k)[1]; 
//
//				logi("k:" + k);
//				logi("number:" + contourInfoMap.get(k)[0]);
//				logi("area:" + contourInfoMap.get(k)[1]);
//				logi("propostion:" + pass);
//
//
//				if (pass > 0.5) {
//					logi("taken k:" + k);
//					subMask = CVTool.getMaskFromCountour(contourMap.get(k));
//					finalMask = CVTool.max(contourMap.get(k), finalMask);///////////
//					homoResultMat.copyTo(reducResultMat, subMask);
//				// reduce such area by diffMap only
//				}else if(contourInfoMap.get(k)[0] > 0){
//					subMask = CVTool.getMaskFromCountour(contourMap.get(k));
//					finalMask = CVTool.max(contourMap.get(k), finalMask);///////////
//					Mat minMask = CVTool.min(subMask, morDiffMap);
//					minQuery.copyTo(reducResultMat, minMask);
//				}
//			}
////			showImage(finalMask, tran2ImageView, "show final mask");	
//		}
//		logi("ReflReduction::: subsitution success");
//		
//		
////		showImage(homoResultMat, tran1ImageView, "show gaussian query gray image");	
////		showImage(reducResultMat, resultImageView, "show final image");
//		
//		Mat[] result = {reducResultMat, morDiffMap, finalMask};	
//		return result;
//	}
  	
    
//    public synchronized Mat getResultbyDamp(Mat queryMat, Mat homoResultMat, Mat diffMap) {
//    	
//		Mat reducResultMat = new Mat();
//		queryMat.copyTo(reducResultMat);
//		for (int i = 0; i < diffMap.rows(); i++) {
//			for (int j = 0; j < diffMap.cols(); j++) {
//				if(diffMap.get(i, j)[0] > 1 && !CVTool.isBlack(homoResultMat.get(i,j))) {
//					reducResultMat.put(i, j, homoResultMat.get(i,j));
//				}
//			}
//		}
//    	return reducResultMat;
//    }
    
//    private class Coordinate {
//    	public int col;
//    	public int row;
//		public Coordinate(int row, int col) {
//			this.col = col;
//			this.row = row;
//		}
//		
//		public String toString() {
//			return "(" + row + ", " + col + ")";
//		}
//    }
}


//Imgproc.findContours(segMat, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);

//subMask = new Mat();
//reducResultMat = new Mat();
//queryMat.copyTo(reducResultMat);
//
//for (int k = 0; k < hierarchy.size().width; k++) {
//
//	logi("contours quantity:" + hierarchy.size());
//	//
//	int maxLevel = 1;
//	Mat countourResult = CVTool.drawContours(segMat, contours, k, new Scalar(255, 0, 0), Core.FILLED, Core.LINE_8, hierarchy, maxLevel, new Point(0,0));
//
//	outerloop:
//	for (int i = 0; i < countourResult.rows(); i++) {
//		for (int j = 0; j < countourResult.cols(); j++) {
//			if(morDiffMap.get(i, j)[0] > 1 && countourResult.get(i, j)[0] > 1) {
//				logi("Reflection Reduction::: getReplace Area, i=" + i + ", j=" + j + ", k=" + k);
//
//				finalMask = CVTool.max(countourResult, finalMask);
//				
//				
//				subMask = CVTool.getMaskFromCountour(countourResult);
//				homoResultMat.copyTo(reducResultMat, subMask);
//				
//				break outerloop;
//			}
//		}
//	} 
//
//
//
//}





/////////////////////////
//long startTime2 = System.currentTimeMillis();
//logi("test Map solution2");
//contourInfoMap= new HashMap<Integer, Integer[]>();
//contourMap= new HashMap<Integer, Mat>();
//Mat.zeros(queryMat.size(), CvType.CV_8UC3);
//
//for (int k = 0; k < hierarchy.size().width; k++) {
//int maxLevel = 1;
//Mat countourResult = CVTool.drawContours(segMat, contours, k, new Scalar(255, 0, 0), Core.FILLED, Core.LINE_8, hierarchy, maxLevel, new Point(0,0));
//contourMap.put(k, countourResult);
//Integer[] temp = {0, CVTool.countourArea(contours.get(k))};
//contourInfoMap.put(k, temp);
//}
//
//int end = qPoint2.size();
//for (int i = 0; i < end; i++) {
//Coordinate p = qPoint2.remove();
//for (int k : contourMap.keySet()) {
//Integer[] temp = contourInfoMap.get(k);
//if (contourMap.get(k).get(p.row, p.col)[0] > 0) {
//temp[0] = temp[0] + 1;	
//contourInfoMap.put(k, temp);
////logi("count here");
//}
//
//}
//}
//
//logi("solution2");
//for (int k : contourInfoMap.keySet()) {
//double pass = (double) contourInfoMap.get(k)[0] / contourInfoMap.get(k)[1]; 
//logi("k:" + k);
//logi("number:" + contourInfoMap.get(k)[0]);
//logi("area:" + contourInfoMap.get(k)[1]);
//logi("propostion:" + pass);
//}
//
//long endTime2 = System.currentTimeMillis();