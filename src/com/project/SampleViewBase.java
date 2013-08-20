package com.project;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.PictureCallback;
import android.hardware.Camera.PreviewCallback;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

public abstract class SampleViewBase extends SurfaceView implements SurfaceHolder.Callback, Runnable {
    private static final String TAG = "Sample::SurfaceView";

    private Camera              mCamera;
    private SurfaceHolder       mHolder;
    private int                 mFrameWidth;
    private int                 mFrameHeight;
    private byte[]              mFrame;
    private boolean             mThreadRun;
    private byte[]              mBuffer;

    public static final int MEDIA_TYPE_IMAGE = 1;
	public static final int MEDIA_TYPE_VIDEO = 2;
    
	// initialize holder
    public SampleViewBase(Context context) {
        super(context);
        mHolder = getHolder();
        mHolder.addCallback(this);
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    public int getFrameWidth() {
        return mFrameWidth;
    }

    public int getFrameHeight() {
        return mFrameHeight;
    }

    public Camera getCamera(){
    	return mCamera;
    }
    
    //set up preview parameters
    public void setPreview() throws IOException {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.HONEYCOMB)
            mCamera.setPreviewTexture( new SurfaceTexture(10) );
        else
        	mCamera.setPreviewDisplay(null);
	}
    
    //start camera 
    public boolean openCamera() {
        Log.i(TAG, "openCamera");
        releaseCamera();
        mCamera = Camera.open();
        if(mCamera == null) {
        	Log.e(TAG, "Can't open camera!");
        	return false;
        }

        mCamera.setPreviewCallbackWithBuffer(new PreviewCallback() {
            public void onPreviewFrame(byte[] data, Camera camera) {
                synchronized (SampleViewBase.this) {
                    System.arraycopy(data, 0, mFrame, 0, data.length);
                    SampleViewBase.this.notify(); 
                }
                camera.addCallbackBuffer(mBuffer);
            }
        });
        return true;
    }
    
    /*  
    1.stop thread
	2.stop preview
	3.release camera
	4.onPreviewStopped() will be defined in Sample1View	*/
    public void releaseCamera() {
        Log.i(TAG, "releaseCamera");
        mThreadRun = false;
        synchronized (this) {
	        if (mCamera != null) {
                mCamera.stopPreview();
                mCamera.setPreviewCallback(null);
                mCamera.release();
                mCamera = null;
            }
        }
        onPreviewStopped();
    }
    
    /*  
    1. get camera parameters
    2. compute the best preview size and set the size to be the previewSize of camera
	3. set camera as continuous focusing
	4. allocate buffer(no idea)
	5. setPreview()
	6. onPreviewStarted, will be defined in Sample1View
	7. stop preview	*/
    public void setupCamera(int width, int height) {
        Log.i(TAG, "setupCamera");
        synchronized (this) {
            if (mCamera != null) {
                Camera.Parameters params = mCamera.getParameters();
                List<Camera.Size> sizes = params.getSupportedPreviewSizes();
                mFrameWidth = width;
                mFrameHeight = height;

                // selecting optimal camera preview size
                {
                    int  minDiff = Integer.MAX_VALUE;
                    for (Camera.Size size : sizes) {
                        if (Math.abs(size.height - height) < minDiff) {
                            mFrameWidth = size.width;
                            mFrameHeight = size.height;
                            minDiff = Math.abs(size.height - height);
                        }
                    }
                }

                params.setPreviewSize(getFrameWidth(), getFrameHeight());
                
                List<String> FocusModes = params.getSupportedFocusModes();
                if (FocusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO))
                {
                	params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
                }            
                
                mCamera.setParameters(params);
                
                /* Now allocate the buffer */
                params = mCamera.getParameters();
                int size = params.getPreviewSize().width * params.getPreviewSize().height;
                size  = size * ImageFormat.getBitsPerPixel(params.getPreviewFormat()) / 8;
                mBuffer = new byte[size];
                /* The buffer where the current frame will be copied */
                mFrame = new byte [size];
                mCamera.addCallbackBuffer(mBuffer);

    			try {
    				setPreview();
    			} catch (IOException e) {
    				Log.e(TAG, "mCamera.setPreviewDisplay/setPreviewTexture fails: " + e);
    			}

                /* Notify that the preview is about to be started and deliver preview size */
                onPreviewStarted(params.getPreviewSize().width, params.getPreviewSize().height);

                /* Now we can start a preview */
                mCamera.startPreview();
            }
        }
    }
    
   //call setupCamera
    public void surfaceChanged(SurfaceHolder _holder, int format, int width, int height) {
        Log.i(TAG, "surfaceChanged");
        setupCamera(width, height);
    }

    
    /*  
    1. new a bitmap bmp = processFrame(mFrame), will be define in Sample1View
	2. lock surfaceholder canvas. Paint the bmp on the canvas
	3.pass the canvas to the surfaceholder, then unlock */
    public void surfaceCreated(SurfaceHolder holder) {
        Log.i(TAG, "surfaceCreated");
        (new Thread(this)).start();
    }

    /*  
    1.stop thread
	2.stop preview
	3.release camera
	4.onPreviewStopped, will be defined in Sample1View		*/
    public void surfaceDestroyed(SurfaceHolder holder) {
        Log.i(TAG, "surfaceDestroyed");
        releaseCamera();
    }

    /* The bitmap returned by this method shall be owned by the child and released in onPreviewStopped() */
    protected abstract Bitmap processFrame(byte[] data);

    /**
     * This method is called when the preview process is being started. It is called before the first frame delivered and processFrame is called
     * It is called with the width and height parameters of the preview process. It can be used to prepare the data needed during the frame processing.
     * @param previewWidth - the width of the preview frames that will be delivered via processFrame
     * @param previewHeight - the height of the preview frames that will be delivered via processFrame
     */
    
    protected abstract void onPreviewStarted(int previewWidtd, int previewHeight);

    /**
     * This method is called when preview is stopped. When this method is called the preview stopped and all the processing of frames already completed.
     * If the Bitmap object returned via processFrame is cached - it is a good time to recycle it.
     * Any other resources used during the preview can be released.
     */
    protected abstract void onPreviewStopped();

    // new a thread to call processFrame to handle mFrame
    // pass the frame to the mHodler to show
    public void run() {
        mThreadRun = true;
        Log.i(TAG, "Starting processing thread");
        while (mThreadRun) {
            Bitmap bmp = null;

            synchronized (this) {
                try {
                    this.wait();
                    bmp = processFrame(mFrame);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            if (bmp != null) {
                Canvas canvas = mHolder.lockCanvas();
                if (canvas != null) {
                    canvas.drawBitmap(bmp, (canvas.getWidth() - getFrameWidth()) / 2, (canvas.getHeight() - getFrameHeight()) / 2, null);
                    mHolder.unlockCanvasAndPost(canvas);
                }
            }
        }
    }
    
///////////
	
//    private PictureCallback mPicture = new PictureCallback() {
//
//    	@Override
//    	public void onPictureTaken(byte[] data, Camera camera) {
//
//    		File pictureFile = getOutputMediaFile(MEDIA_TYPE_IMAGE);
//    		if (pictureFile == null){
//    			Log.d(TAG, "Error creating media file, check storage permissions: ");
//    			return;
//    		}
//
//    		try {
//    			FileOutputStream fos = new FileOutputStream(pictureFile);
//    			fos.write(data);
//    			fos.close();
//    		} catch (FileNotFoundException e) {
//    			Log.d(TAG, "File not found: " + e.getMessage());
//    		} catch (IOException e) {
//    			Log.d(TAG, "Error accessing file: " + e.getMessage());
//    		}
//    	}
//    };


//    /** Create a file Uri for saving an image or video */
//    private static Uri getOutputMediaFileUri(int type){
//    	return Uri.fromFile(getOutputMediaFile(type));
//    }
//
//    
//    private boolean checkSDCard() {
//    	
//    	if(android.os.Environment.getExternalStorageState().equals(android.os.Environment.MEDIA_MOUNTED)) {
//    		return true;
//    	} else {
//    		return false;
//    	}
//    }
//
//
//    /** Create a File for saving an image or video */
//    private static File getOutputMediaFile(int type){
//    	// To be safe, you should check that the SDCard is mounted
//    	// using Environment.getExternalStorageState() before doing this.
//
//    	//	if (checkSDCard()) {
//	
//    	File mediaStorageDir = new File(Environment.getExternalStoragePublicDirectory(
//    			Environment.DIRECTORY_PICTURES), "ReflectionAttenuator");
//    	// This location works best if you want the created images to be shared
//    	// between applications and persist after your app has been uninstalled.
//
//    	// Create the storage directory if it does not exist
//    	if (! mediaStorageDir.exists()){
//    		if (! mediaStorageDir.mkdirs()){
//    			Log.d("ReflectionAttenuator", "failed to create directory");
//    			return null;
//    		}
//    	}
//
//    	// Create a media file name
//    	String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
//    	File mediaFile;
//    	if (type == MEDIA_TYPE_IMAGE){
//    		mediaFile = new File(mediaStorageDir.getPath() + File.separator +
//    				"IMG_"+ timeStamp + ".jpg");
//    	} else if(type == MEDIA_TYPE_VIDEO) {
//    		mediaFile = new File(mediaStorageDir.getPath() + File.separator +
//    				"VID_"+ timeStamp + ".mp4");
//    	} else {
//    		return null;
//    	}
//
//    	return mediaFile;
//    }
    
    
    
}