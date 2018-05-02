#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MIT License

Copyright (c) 2018 Volral Francois 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import division
import numpy as np
import cv2 

def getContours(img):
    """
    The MIT License (MIT)

    Copyright (c) 2017 Dhanushka Dangampola
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
    """
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad   = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    
    # using RETR_EXTERNAL instead of RETR_CCOMP
    _, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours

def processContours(contours, img_w, img_h):
    """
    Get the bounding rects of the contours and compute 
    the median ratio of rectangle. 
    """
    ratioWH = []
    area = img_w * img_h 
    reject = 0
    for c  in contours:
        x, y, w, h = cv2.boundingRect(c)
        """
        area of rect should cover at least 0.05% of the image
        and ratio should be interesting enough by its distance
        for a perfect squared ratio of 1:1
        """
        rect_area = w * h 
        ratio = w / h 
        dist_of_square = abs(ratio - 1.0)

        if rect_area > area * 0.0005 and dist_of_square > 0.3:
            ratioWH.append(w/h)
        else :
            reject += 1
            continue

    print reject
    if len(ratioWH) == 0:
        return 0
    
    med_ratio = np.median(np.asarray(ratioWH))

    return med_ratio

def needToBeRotated(img, img_name):
    """
    Detect if an image that mainly contains text is arranged vertically or 
    horizontally, not in which sens it should be rotated though. It's DIY and 
    probably overfitted but seems robust. The idea use the excellent answer of 
    Dhanushka Dangampola that can be found here : 
    https://stackoverflow.com/a/23672571/5075502. 
    I try to detect text box and compute their ratio.
    """

    img_w, img_h, chan = img.shape
    
    #
    #  Image Preprocessing 
    #

    # When too big
    if img_w > 1000 or img_h > 1000:
        img = cv2.pyrDown(img)                      
    
    img = cv2.bilateralFilter(img, 3, 75, 75)   # Remove noise
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
       
    
    # Remove thin line from image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 12))
    morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.add(img, (255-morphed))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
    morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.add(img, (255-morphed))
    
    # Rotate and compare contours 
    rotated_img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)

    c1 = getContours(img)
    med_ratio_a = processContours(c1, img_w, img_h)
    
    c2 = getContours(rotated_img)
    med_ratio_b = processContours(c2, img_w, img_h)
  
    return med_ratio_a < med_ratio_b


test_images = [ 
      #             NAME             ASSERT
      ( "test_images/text_test1.png", False),
      ( "test_images/text_test2.jpg", False),
      ( "test_images/text_test3.png", False),
      ( "test_images/text_test4.png",  True),
      ( "test_images/text_test5.jpg", False),
      ( "test_images/text_test6.png", False),
      ( "test_images/text_test7.png",  True),
      ( "test_images/text_test8.png", False),
      ( "test_images/text_test9.png", False),
      ("test_images/text_test10.jpg", False),
      ("test_images/text_test11.png", False),
      ("test_images/text_test12.png", False),
      ("test_images/text_test13.jpg", False),
      ("test_images/text_test14.jpg", False),
      ("test_images/text_test15.jpg", False),
      ("test_images/text_test16.png",  True),
      ("test_images/text_test17.png", False),
      ("test_images/text_test18.png", False),
      ("test_images/text_test19.png", False),
      ("test_images/text_test20.png", False),
      ("test_images/text_test21.png", False),
      ("test_images/text_test22.jpg", False),
      ("test_images/text_test23.jpg", False),
      ("test_images/text_test24.jpg", False),
      ("test_images/text_test24.jpg", False),
      ("test_images/text_test25.jpg", False),
      ("test_images/text_test26.jpg", False),
      ("test_images/text_test27.jpg", False),
      ("test_images/text_test28.jpg", False),
      ("test_images/text_test29.jpg", False)]


WIN, FAIL = (0,0)

for t in test_images:
    img_name, should_be = t
    img = cv2.imread( img_name )  
    
    print img_name,   
    if needToBeRotated(img, img_name) == should_be:
        WIN  += 1; print "SUCCESS !"
    else:
        FAIL += 1; print "FAILED  !"

print WIN, "/", FAIL, " W/F"









