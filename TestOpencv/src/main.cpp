#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include<opencv2/legacy/legacy.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include <opencv2/objdetect/objdetect.hpp>


#include<stdio.h>
#include<cstdlib>
#include<iostream>
#include<string.h>
#include<fstream>
#include<sstream>
#include<dirent.h>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;



Mat sequenceimage , referenceimage ,img_matches , highestmatchimg;

int numbermatch = 0 ,
    highestmatch = 0 ,
	j = 0 ;
string strfilename = "" , highestmatchimage=""; // declare images
const double THRESHOLD = 400; //declare threshold value

ifstream gpsfile ;
string gpsvalue;

void SurfFinMatch();  // calling SurfFinMatch method

int main( int argc, char** argv )
{

  	referenceimage = imread("src/ReferenceImage/b5.jpg",CV_LOAD_IMAGE_GRAYSCALE); // Read reference image
     int width = referenceimage.cols; // Get width of reference image
     struct dirent *entry;
	 DIR *dir;
	 dir = opendir("src/Dataset"); // Path of Dataset

	 namedWindow("Closest Matching Images (CMI)", 1);  // Open Window

	 /*
	  *  Read Dataset Images , Get values of Dataset , Passed the SurfFinMatch() Funcation
	  *  Set value of Highest keypoint match Images
	  *  Show in imshow both reference and sequance Images
	  */
      gpsfile.open("src/GPS_Data/gpsdetail.txt");
      if (gpsfile.fail()){
    	 cout << " Error : File Couldnot Found !!!" ;
      }

	  while (((entry = readdir(dir)) != NULL) && (!gpsfile.eof())) {
	        if(((string)entry->d_name == "..") || ((string)entry->d_name == ".") || ((string)entry->d_name == "4.jpg") ){
	        } else {
	        	strfilename = string(entry->d_name);
	        	sequenceimage = imread("src/Dataset/" + strfilename , CV_LOAD_IMAGE_GRAYSCALE);
	        	numbermatch = 0 ;
	        	SurfFinMatch();
	        	stringstream ss ;
	            ss << numbermatch;

	            /* Reading File */

	            gpsfile >> gpsvalue;
	            int index = gpsvalue.find(",");
	            string longitude = gpsvalue.substr(0,index);
	            string latitude = gpsvalue.substr(index + 1,gpsvalue.length());
                string gpsresult = "longitude :" + longitude + " latitude : " + latitude ;

	            putText(img_matches,"Sequence Image ",Point(20,20),FONT_HERSHEY_SIMPLEX,.8,Scalar(255,0,0),2);
	            putText(img_matches,"Reference Image ",Point(width + 20,20),FONT_HERSHEY_SIMPLEX,.8,Scalar(255,0,0),2);
	        	putText(img_matches,"KeyPoint Nb: " + ss.str(),Point(20,50),FONT_HERSHEY_SIMPLEX,.8,Scalar(0,0,255),2);
	        	putText(img_matches,gpsresult,Point(20,75),FONT_HERSHEY_SIMPLEX,.8,Scalar(0,0,255),2);
	        	imshow( "Closest Matching Images (CMI)", img_matches );

	        	if (highestmatch < numbermatch )
	        	{
	        		highestmatch = numbermatch ;
	        		highestmatchimage = strfilename;
	        		highestmatchimg = sequenceimage;
	        	}
	        	waitKey(5000);
	        }

	        j = j + 1;

	    }
	    closedir(dir);

	    /* Write Result image in Result Folder */

	    imwrite( "src/Output/" + highestmatchimage , highestmatchimg );

	    waitKey(0);
 	    return 0;
}




void SurfFinMatch(){

	  int minHessian = 400;

	  SurfFeatureDetector detector( minHessian );

	  std::vector<KeyPoint> keypoints_1, keypoints_2; //declear keypoints

	  detector.detect( sequenceimage, keypoints_1 );
	  detector.detect( referenceimage, keypoints_2 );

	  //-- Step 2: Calculate descriptors (feature vectors)
	  SurfDescriptorExtractor extractor;

	  Mat descriptors_1, descriptors_2;

	  extractor.compute( sequenceimage, keypoints_1, descriptors_1 );
	  extractor.compute( referenceimage, keypoints_2, descriptors_2 );

	  //-- Step 3: Matching descriptor vectors using FLANN matcher
	  FlannBasedMatcher matcher;
	  std::vector< DMatch > matches;
	  matcher.match( descriptors_1, descriptors_2, matches );

	  double max_dist = 0; double min_dist = 100;

	  //-- Quick calculation of max and min distances between keypoints
	  for( int i = 0; i < descriptors_1.rows; i++ )
	  { double dist = matches[i].distance;
	    if( dist < min_dist ) min_dist = dist;
	    if( dist > max_dist ) max_dist = dist;
	  }

	 // printf("-- Max dist : %f \n", max_dist );
	 // printf("-- Min dist : %f \n", min_dist );

	  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	  //-- small)
	  //-- PS.- radiusMatch can also be used here.
	  std::vector< DMatch > good_matches;

	  for( int i = 0; i < descriptors_1.rows; i++ )
	  { if( matches[i].distance <= max(2*min_dist, 0.02) )
	    { good_matches.push_back( matches[i]); }
	  }

	  //-- Draw only "good" matches

	  drawMatches( sequenceimage, keypoints_1, referenceimage, keypoints_2,
	               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
	               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	  numbermatch = (int)good_matches.size();

}



