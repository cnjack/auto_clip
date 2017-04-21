//
//  main.cpp
//  image
//
//  Created by 王文慧 on 2017/4/21.
//  Copyright © 2017年 王文慧. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "auto_clip.hpp"


int main() {
    cv::Mat image = cv::imread("test.jpg");
    cv::Mat dest_image = auto_clip(image, 300, 300);
    cv::imwrite("dest.jpg", dest_image);
    return 0;
}
