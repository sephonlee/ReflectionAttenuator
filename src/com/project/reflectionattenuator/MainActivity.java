package com.project.reflectionattenuator;



import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.opencv.R.id;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ContentResolver;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Camera;
import android.hardware.Camera.PictureCallback;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends Activity implements View.OnClickListener{

	private CameraPreview mPreview;
	private Button captureButton;
	private Button cameraNumberButton;
	private Button browseButton1;
	private Button browseButton2;
	private Button processButton;
	private Button cameraOpenCloseButton;
    private TextView cameraNumberText;
    private ImageView imageView01;
    private ImageView imageView02;
    private FrameLayout previewRight;
    private FrameLayout previewLeft;
    
    
    private int targetImageView;
    private Bitmap[] bitmapList;
    private List<byte[]> byteArrayList;
    private Mat[] MArray;
    private Boolean cameraOpen;
    
    public Camera mCamera;
    private int previewWin;
    
    private static final int OPENCV = 1;
    public final static String SER_KEY = "ReflectionaAttenuation";
	private static final String TAG = "ReflectionaAttenuationActivity";
	public static final int MEDIA_TYPE_IMAGE = 1;
	public static final int MEDIA_TYPE_VIDEO = 2;
	private static final int SCALEFACTOR = 4; //downsamping ratio
	
	protected static final int MENU_RGB = Menu.FIRST;
	protected static final int MENU_GRAY = Menu.FIRST + 1;
	protected static final int MENU_CANNY = Menu.FIRST + 2;

	////////////////////////////////////////////
//	private static final String TAG = "Sample::Activity";

    private MenuItem            mItemPreviewRGBA;
    private MenuItem            mItemPreviewGray;
    private MenuItem            mItemPreviewCanny;
    private Sample1View         mView;
    
    // call when load
    private BaseLoaderCallback  mOpenCVCallBack = new BaseLoaderCallback(this) {
    	@Override
    	public void onManagerConnected(int status) {
    		switch (status) {
				case LoaderCallbackInterface.SUCCESS:
				{
					Log.i(TAG, "OpenCV loaded successfully");
					// Create and set View
					mView = new Sample1View(mAppContext);
//					setContentView(mView);
					setContentView(R.layout.main);
					initializeation();
					findViews();
			        previewRight.addView(mView);        
			   
			        setListensers();
			        // not open camera here
//			        mCamera = mView.getCamera();
//			        mCamera.stopPreview();
			        
//					// Check native OpenCV camera
//					if( !mView.openCamera() ) {
//						AlertDialog ad = new AlertDialog.Builder(mAppContext).create();
//						ad.setCancelable(false); // This blocks the 'BACK' button
//						ad.setMessage("Fatal error: can't open camera!");
//						ad.setButton("OK", new DialogInterface.OnClickListener() {
//						    public void onClick(DialogInterface dialog, int which) {
//							dialog.dismiss();
//							finish();
//						    }
//						});
//						ad.show();
//					}
				} break;
				default:
				{
					super.onManagerConnected(status);
				} break;
			}
    	}
	};
    
	// initialize parameter
	private void initializeation() {
		bitmapList = new Bitmap[2];
		MArray = new Mat[2];
		byteArrayList = new ArrayList<byte[]>();
		previewWin = 1;
		cameraOpen = false;
	}
	
	// find views in the user interface
	private void findViews() {
		cameraNumberButton = (Button) findViewById(R.id.button_number);
	    processButton = (Button) findViewById(R.id.button_process);
		cameraOpenCloseButton = (Button) findViewById(R.id.button_cameraOpenClose);
	    captureButton = (Button) findViewById(R.id.button_capture);
	    browseButton1 = (Button) findViewById(R.id.button_browse1);
	    browseButton2 = (Button) findViewById(R.id.button_browse2);
	    cameraNumberText = (TextView) findViewById(R.id.textView_number);
		previewRight = (FrameLayout) findViewById(R.id.camera_preview1);
		imageView01 = (ImageView) findViewById(R.id.imageView1);
		imageView02 = (ImageView) findViewById(R.id.imageView2);
		
	}
	
    // Listen for button clicks
    private void setListensers() {
    	cameraNumberButton.setOnClickListener(this);
    	captureButton.setOnClickListener(this);
    	browseButton1.setOnClickListener(this);
    	browseButton2.setOnClickListener(this);
    	processButton.setOnClickListener(this);
    	cameraOpenCloseButton.setOnClickListener(this);
    }
	
    
    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "onCreate");
        super.onCreate(savedInstanceState);
//        requestWindowFeature(Window.FEATURE_NO_TITLE);

        Log.i(TAG, "Trying to load OpenCV library");
        if (!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_2, this, mOpenCVCallBack))
        {
        	Log.e(TAG, "Cannot connect to OpenCV Manager");
        }
    }
    
    @Override
	protected void onPause() {
        Log.i(TAG, "onPause");
		super.onPause();
		if (null != mView)
			mView.releaseCamera();
	}

	@Override
	protected void onResume() {
        Log.i(TAG, "onResume");
		super.onResume();
//		if( (null != mView) && !mView.openCamera() ) {
//			AlertDialog ad = new AlertDialog.Builder(this).create();  
//			ad.setCancelable(false); // This blocks the 'BACK' button  
//			ad.setMessage("Fatal error: can't open camera!");  
//			ad.setButton("OK", new DialogInterface.OnClickListener() {  
//			    public void onClick(DialogInterface dialog, int which) {  
//				dialog.dismiss();
//				finish();
//			    }  
//			});  
//			ad.show();
//		}
	}
	
	///////////////////////////
	
	private PictureCallback mPicture = new PictureCallback() {

	    @Override
	    public void onPictureTaken(byte[] data, Camera camera) {

	    	/* data is the binary data of the pictureon */
	        Bitmap bm = BitmapFactory.decodeByteArray(data, 0, data.length); 
	        
	        File pictureFile = getOutputMediaFile(MEDIA_TYPE_IMAGE);
//	        File pictureFile = new File("mnt/sdcard/asd.bmp");
	        if (pictureFile == null){
	            Log.d(TAG, "Error creating media file, check storage permissions: ");
	            return;
	        }

	        try {
	        	BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(pictureFile));
	        	/* compress the data */
	            bm.compress(Bitmap.CompressFormat.JPEG, 80, bos);
	            
	            
	            logi("Width: " + bm.getWidth() + " ,height: " + bm.getHeight());
	            
	            /*update the BufferStream */
	            bos.flush();
	            
	            /* close OutputStream */
	            bos.close();
	            
	            
//	            Bitmap bitmap = Bitmap.createScaledBitmap(bm, bm.getWidth() / SCALEFACTOR, bm.getHeight() / SCALEFACTOR, false);
	            
	            if (previewWin == 1) {
	            	logi("preview1 start");
//	            	bitmapList[0] = null;
	            	MArray[0] = null;
	            	imageView01.setImageBitmap(null);
	            	logi("get bitmap");
	            	Bitmap bitmap = Bitmap.createScaledBitmap(bm, bm.getWidth() / SCALEFACTOR, bm.getHeight() / SCALEFACTOR, false);
	            	logi("set bitmap to view");
	            	imageView01.setImageBitmap(bitmap);
	            	
	            	
	            	logi("put bitmap into arrayList[0]");
//	            	bitmapList[0] = bitmap; 
	            	MArray[0] = CVTool.bitmapToMat(bitmap, true);
	            	
	            	logi("set next shut to preview2");
	            	previewWin = 0;
	            } else if (previewWin == 0) {
	            	logi("preview2 start");
//	            	bitmapList[1] = null;
	            	MArray[1] = null;
	            	
	            	imageView02.setImageBitmap(null);
	            	logi("get bitmap");
	            	Bitmap bitmap = Bitmap.createScaledBitmap(bm, bm.getWidth() / SCALEFACTOR, bm.getHeight() / SCALEFACTOR, false);
	            	logi("set bitmap to view");
	            	imageView02.setImageBitmap(bitmap);
	            	
	            	logi("put bitmap into arrayList[1]");
//	            	bitmapList[1] = bitmap;
	            	MArray[1] = CVTool.bitmapToMat(bitmap, true);
	            	
	            	logi("set next shut to preview1");
	            	previewWin = 1;
	            }
	            
//	            FileOutputStream fos = new FileOutputStream(pictureFile);
//	            fos.write(data);
//	            fos.close();
	            
//	            imageView01.setImageBitmap(bm);
	            
	            
//	    		mView.getCamera().stopPreview();
//	            mView.getCamera().stopPreview();
//	            mView.setupCamera(mView.getFrameWidth(), mView.getFrameHeight());
	    		mView.getCamera().startPreview();
	    		
	            
	        } catch (FileNotFoundException e) {
	            Log.d(TAG, "File not found: " + e.getMessage());
	        } catch (IOException e) {
	            Log.d(TAG, "Error accessing file: " + e.getMessage());
	        }
	    }
	};
	
	
	/** Create a file Uri for saving an image or video */
	private static Uri getOutputMediaFileUri(int type){
	      return Uri.fromFile(getOutputMediaFile(type));
	}

	// check if SD card exists
	private boolean checkSDCard() {
	    if(android.os.Environment.getExternalStorageState().equals(android.os.Environment.MEDIA_MOUNTED)) {
	      return true;
	    } else {
	      return false;
	    }
	}
	
	
	/** Create a File for saving an image or video */
	private static File getOutputMediaFile(int type){
	    // To be safe, you should check that the SDCard is mounted
	    // using Environment.getExternalStorageState() before doing this.

//		if (checkSDCard()) {
		
	    File mediaStorageDir = new File(Environment.getExternalStoragePublicDirectory(
	              Environment.DIRECTORY_PICTURES), "ReflectionAttenuator");
	    // This location works best if you want the created images to be shared
	    // between applications and persist after your app has been uninstalled.

	    // Create the storage directory if it does not exist
	    if (! mediaStorageDir.exists()){
	        if (! mediaStorageDir.mkdirs()){
	            Log.d("ReflectionAttenuator", "failed to create directory");
	            return null;
	        }
	    }

	    // Create a media file name
	    String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
	    File mediaFile;
	    if (type == MEDIA_TYPE_IMAGE){
	        mediaFile = new File(mediaStorageDir.getPath() + File.separator +
	        "IMG_"+ timeStamp + ".jpg");
	    } else if(type == MEDIA_TYPE_VIDEO) {
	        mediaFile = new File(mediaStorageDir.getPath() + File.separator +
	        "VID_"+ timeStamp + ".mp4");
	    } else {
	        return null;
	    }

	    return mediaFile;
	}
	
	
	
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "onCreateOptionsMenu");
        mItemPreviewRGBA = menu.add(0, MENU_RGB, 0 ,"Preview RGBA");
        mItemPreviewGray = menu.add(0, MENU_GRAY, 0 ,"Preview GRAY");
        mItemPreviewCanny = menu.add(0, MENU_CANNY, 0 ,"Canny");
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "Menu Item selected " + item);
        if (item == mItemPreviewRGBA) {
        	mView.setViewMode(Sample1View.VIEW_MODE_RGBA);
        } else if (item == mItemPreviewGray) {
        	mView.setViewMode(Sample1View.VIEW_MODE_GRAY);
        } else if (item == mItemPreviewCanny) {
        	mView.setViewMode(Sample1View.VIEW_MODE_CANNY);
        }
        return super.onOptionsItemSelected(item);
    }
	
	////////////////////////////////////////////////////////////////////
    
    
	@Override
	public void onClick(View v) {
		// TODO Auto-generated method stub
		logi("onClick!");
		// TODO Auto-generated method stub
		switch (v.getId()) {
		case R.id.button_number:
			btn_getCameraQuantity();
			break;
			
		case R.id.button_capture:
			if (cameraOpen) {
				btn_takePicture();
			} else {
				toasti("You have to open the camera first");
				logi("Camera is open:" + cameraOpen);
			}
			break;
		
		case R.id.button_browse1:
			// toImageView = 1, saveIndex = 0
			btn_browseFile(1, 0);
			break;
			
		case R.id.button_browse2:
			// toImageView = 1, saveIndex = 0
			btn_browseFile(2, 1);
			break;
			
		case R.id.button_process:
			btn_toImgProcess();
			break;
			
		case R.id.button_cameraOpenClose:
			btn_openCloseCamera();
			break;
		}
	}
    
	// get number of cameras
	private void btn_getCameraQuantity() {
		Log.i(TAG, "This device has "+ Camera.getNumberOfCameras() );
        DecimalFormat nf = new DecimalFormat("0");
        double numberofcamera = Camera.getNumberOfCameras();
        cameraNumberText.setText("numebrs:" + nf.format(numberofcamera));
	}
	
	// take picture
	private void btn_takePicture() {
		mView.getCamera().takePicture(null, null, mPicture);
//		mView.getCamera().stopPreview();
//		mView.getCamera().startPreview();
	}
	
	// show image in the given 
	private void btn_browseFile(int toImageView, int index) {
		Log.i(TAG, "/////set picture button1.");
		targetImageView = toImageView;
		bitmapList[index] = null; 
		MArray[index] = null;
		Intent intent = new Intent();
	    /* set the intent type */
	    intent.setType("image/*");
	    /* set the intent action */
	    intent.setAction(Intent.ACTION_GET_CONTENT);
	    
//	    Intent intent = new Intent(Intent.ACTION_PICK,
//                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
	    /* start intent */
	    startActivityForResult(intent, 1);
	}
	
	// go to image process page
	private void btn_toImgProcess() {
		try {
			goToImgProcess();
		} catch (Exception obj) {
			toasti("You have to select two pictures");
		}
	}
	
	//open and close camera
	private void btn_openCloseCamera() {
		logi("openCloseCamera::: cameraOn= " + cameraOpen);
			
		if (!cameraOpen) {
			previewRight.removeAllViews();
			previewRight.addView(mView); 
			// open camera
			if( (null != mView) && !mView.openCamera() ) {
				AlertDialog ad = new AlertDialog.Builder(this).create();  
				ad.setCancelable(false); // This blocks the 'BACK' button  
				ad.setMessage("Fatal error: can't open camera!");  
				ad.setButton("OK", new DialogInterface.OnClickListener() {  
					public void onClick(DialogInterface dialog, final int which) {  
						dialog.dismiss();
						finish();
					}  
				});				
				ad.show();
			}
			cameraOpenCloseButton.setText("Close Camera");
			cameraOpen = true;
		} else {
			logi("closeCamera");
			cameraOpenCloseButton.setText("Open Camera");
			if (null != mView) {
				mView.releaseCamera();
				previewRight.removeAllViews();
				cameraOpen = false;
			}
		}
	}
	

	
//	public void oldGoToImgProcess() {
//		Log.i(TAG, "/////////////start package");
//		Intent intent = new Intent();
//		intent.setClass(MainActivity.this, ImgProcess.class);
//		Bundle bundle = new Bundle();
//        
//		bundle.putByteArray("imageData1", Bitmap2Bytes(bitmapList[0]));
//		bitmapList[0].recycle();
//		bundle.putByteArray("imageData2", Bitmap2Bytes(bitmapList[1]));
//		bitmapList[1].recycle();
//		
//		intent.putExtras(bundle);
//		Log.i(TAG, "/////////////process successful");
//		startActivity(intent);
//	}
	
	//
    public void goToImgProcess(){  
    	if (MArray[0] != null && MArray[1] != null) {
    		Log.i(TAG, "/////////////start package");
        	Intent intent = new Intent();  
        	intent.setClass(MainActivity.this, ImgProcess.class);
            Bundle bundle = new Bundle();  
        	
        	ImagePackage mPackage = new ImagePackage(); 
            
            mPackage.putImage1(MArray[0]);
            mPackage.putImage2(MArray[1]);
            
            bundle.putSerializable(SER_KEY, mPackage);  
            
            intent.putExtras(bundle);
            Log.i(TAG, "/////////////process successful");
            startActivity(intent);  
    	}else {
    		logi("MArray[0] or MArray[1] is null");
    		toasti("You must select or take two pictures before go to process page");
    	}
    } 
	
	
	//get image from gallery
	@Override
	protected void onActivityResult(int requestCode, int resultCode, Intent data) {
		if (resultCode == RESULT_OK) {
			Uri uri = data.getData();
		    ContentResolver cr = this.getContentResolver();
		    
		    try {
		    	Log.i(TAG, "/////////////read image from gallery");
		    	Bitmap orig_bitmap = BitmapFactory.decodeStream(cr.openInputStream(uri));
		    	Bitmap bitmap = Bitmap.createScaledBitmap(orig_bitmap, orig_bitmap.getWidth() / SCALEFACTOR, orig_bitmap.getHeight() / SCALEFACTOR, false);
		    	orig_bitmap.recycle();
		    	
		    	
		    	//Bitmap bitmap = ThumbnailUtils.createVideoThumbnail(uri.getPath(), MediaStore.Video.Thumbnails.MINI_KIND);

		        /* show Bitmap in ImageView */
		    	switch(targetImageView) {
		    	case 1:
//		    		bitmapList[0] = bitmap; 	    				
		    		MArray[0] = CVTool.bitmapToMat(bitmap, true);		

		    		
		    		imageView01.setImageBitmap(bitmap);
		    		Log.i(TAG, "/////////////read file 1 path:" + uri.getPath());
		    		break;
		    		
		    	case 2:
//		    		bitmapList[1] = bitmap; 
		    	    MArray[1] = CVTool.bitmapToMat(bitmap, true);
		    		
		    		imageView02.setImageBitmap(bitmap);
		    		Log.i(TAG, "/////////////read file 2 path:" + uri.getPath()); 		
		    		break;
		    	}
		    	
		        Log.i(TAG, "set picture in imageView" + targetImageView + "success.");
		        
		        
		        } catch (FileNotFoundException e) {
		        e.printStackTrace();
		        }
		        
		        
		    }
		 
		super.onActivityResult(requestCode, resultCode, data);
	}		
    
	
	// Byte to Mat
	private Mat byte2Mat(byte[] src) {
		logi("Convert Byte to Mat");
		Mat result = new Mat();
		Utils.bitmapToMat(Bytes2Bimap(src), result);
		logi("Convert Byte to Mat successfully");
		return result;
	}

	
	
	//Bitmap to Byte[]
	private byte[] Bitmap2Bytes(Bitmap bm) {  
	    ByteArrayOutputStream baos = new ByteArrayOutputStream();    
	    bm.compress(Bitmap.CompressFormat.PNG, 100, baos);    
	    return baos.toByteArray();  
	}
	
	//Byte[] to Bitmap
	private Bitmap Bytes2Bimap(byte[] b){  
        if(b.length!=0){  
            return BitmapFactory.decodeByteArray(b, 0, b.length);  
        }  
        else {  
            return null;  
        }  
	}
	
	private void logi(String str) {
		Log.i(TAG, "%%%%%%%%%%%%%%%" + str);
	}
	
	private void toasti(String str) {
		Toast.makeText(MainActivity.this, str, Toast.LENGTH_SHORT).show();
	}
}
	
	
