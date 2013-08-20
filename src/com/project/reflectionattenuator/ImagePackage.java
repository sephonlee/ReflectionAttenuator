package com.project.reflectionattenuator;

import java.io.Serializable;

import org.opencv.core.Mat;

public class ImagePackage implements Serializable{
 
    private Mat trainMat;  
    private Mat queryMat;  
    
    public void putImage1(Mat src) {  
        trainMat = src;
    }  
    public void putImage2(Mat src) {  
        queryMat = src; 
    }  
    
    public Mat getImage1Mat() {  
        return trainMat;  
    }  
    public Mat getImage2Mat() {  
        return queryMat;  
    }  
	
}
