import os
from os import listdir
from os.path import isfile,join
import sys, traceback
import cv2
import numpy as np
import argparse
import string
from plantcv import plantcv as pcv
from plantcv import learn as pc_learn
from plantcv.plantcv.visualize import histogram
from plantcv.plantcv import outputs
from plantcv import plantcv as pcv
from matplotlib import pyplot as plt


### Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-n", "--nir", help="Input image file.", required=False)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r", "--result", help="result file.", required=False)
    parser.add_argument("-w", "--writeimg", help="write out images.", default=False, action="store_true")
    parser.add_argument("-D", "--debug",
                        help="can be set to 'print' or None (or 'plot' if in jupyter) prints intermediate images.",
                        default=None)
    args = parser.parse_args()
    return args

def rotateImage(image, angle):
    row,col = image.shape
    #center=tuple(np.array([row,col])/2)
    center=(1196, 1648)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def main():
    args=options()

    pcv.params.debug = args.debug
    if (args.writeimg == True):
        filepathname = os.path.basename(args.image)
        pathname = os.path.join(args.outdir, filepathname.rstrip(".png"))
        print(pathname)

        if os.path.exists(pathname) == False:
        	os.mkdir(pathname)

        pcv.params.debug_outdir = pathname
    #  pcv.outputs.clear()

    img, path, filename = pcv.readimage(filename=args.image)
    # In[8]:

    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    # In[9]:

    s_thresh = pcv.threshold.binary(gray_img=s, threshold=85, max_value=255, object_type='light')

    # In[10]:

    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)

    # In[11]:

    s_cnt = pcv.median_blur(gray_img=s_thresh, ksize=5)

    # In[12]:

    b = pcv.rgb2gray_lab(rgb_img=img, channel='b')

    # In[14]:

    b_thresh = pcv.threshold.binary(gray_img=b, threshold=160, max_value=255, object_type='light')
    b_cnt = pcv.threshold.binary(gray_img=b, threshold=160, max_value=255, object_type='light')

    # In[15]:

    bs = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_cnt)

    # In[16]:

    masked = pcv.apply_mask(img=img, mask=bs, mask_color='white')

    # In[17]:

    masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel='a')
    masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel='b')

    # In[18]:

    masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel='a')
    masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel='b')

    # In[19]:


    maskeda_thresh = pcv.threshold.binary(gray_img=masked_a, threshold=115, max_value=255, object_type='dark')
    maskeda_thresh1 = pcv.threshold.binary(gray_img=masked_a, threshold=135, max_value=255, object_type='light')
    maskedb_thresh = pcv.threshold.binary(gray_img=masked_b, threshold=128, max_value=255, object_type='light')

    # In[20]:

    ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
    ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)

    ab_fill = pcv.fill(bin_img=ab, size=200)

    masked2 = pcv.apply_mask(img=masked, mask=ab_fill, mask_color='white')

    # In[23]:

    id_objects, obj_hierarchy = pcv.find_objects(img=masked2, mask=ab_fill)

    # In[25]:

    if "z2000" in filename:
        roi1, roi_hierarchy= pcv.roi.rectangle(img=masked2, x=850, y=400, h=1400, w=800)
    else :
    	roi1, roi_hierarchy= pcv.roi.rectangle(img=masked2, x=850, y=0, h=1500, w=800)

       # Decide which objects to keep
    roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img=img, roi_contour=roi1,
                                                                  roi_hierarchy=roi_hierarchy,
                                                                  object_contour=id_objects,
                                                                  obj_hierarchy=obj_hierarchy,
                                                                  roi_type='partial')

    obj, mask = pcv.object_composition(img, roi_objects, hierarchy3)

    ############### Analysis ################


## below 3 lines were commented out before
#    outfile=False
#    if args.writeimg == True:
#	outfile = args.outdir + "/" + filename

    # Find shape properties, output shape image (optional)
    shape_imgs = pcv.analyze_object(img, obj, mask)

    # Shape properties relative to user boundary line (optional)
    if "z1000" in filename:
        boundary_img1 = pcv.analyze_bound_horizontal(img,obj,mask,1470)
    elif "z2000" in filename:
        boundary_img1 = pcv.analyze_bound_horizontal(img, obj, mask, 1680)
    else:
        boundary_img1 = pcv.analyze_bound_horizontal(img, obj, mask, 1370)

    # Determine color properties: Histograms, Color Slices, output color analyzed histogram (optional)
    color_histogram = pcv.analyze_color(img, kept_mask, 'all')

    # Pseudocolor the grayscale image
    pseudocolored_img = pcv.visualize.pseudocolor(gray_img=s, mask=kept_mask, cmap='jet')

    nir_path = pcv.get_nir(path=path,filename=filename)
    nir_img, path, img_filename = pcv.readimage(filename= nir_path, mode="native")


    # In[161]:

    if "z1000" in filename:
        img_warped, mat = pcv.transform.warp(img=kept_mask,
                refimg=nir_img,
                pts = [(229, 241), (234, 1156), (1050, 1467), (2230, 241),
                                             (2227, 1171), (1411, 1461)],
                refpts = [(28, 72), (30, 332), (262,422), (600, 75),
                                                    (594, 333), (364, 420)],
                method='lmeds')

        img_warped_original, mat = pcv.transform.warp(img=img,
                refimg=nir_img,
                pts = [(229, 241), (234, 1156), (1050, 1467), (2230, 241),
                                             (2227, 1171), (1411, 1461)],
                refpts = [(28, 72), (30, 332), (262,422), (600, 75),
                                                    (594, 333), (364, 420)],
                method='lmeds')

    elif "z2000" in filename:
        img_warped, mat = pcv.transform.warp(img=kept_mask,
            	refimg=nir_img,
            	pts = [(1005, 1660), (1140, 1730), (1140, 1850), (1413, 1677),
                                         (1389, 1734), (1389, 1854)],
            	refpts = [(254, 200), (291, 214), (290, 245), (372, 200),
                                                (362, 214), (361, 247)],
            	method='lmeds')

        img_warped_original, mat = pcv.transform.warp(img=img,
            	refimg=nir_img,
            	pts = [(1005, 1660), (1140, 1730), (1140, 1850), (1413, 1677),
                                         (1389, 1734), (1389, 1854)],
            	refpts = [(254, 200), (291, 214), (290, 245), (372, 200),
                                                (362, 214), (361, 247)],
            	method='lmeds')

    else:
        img_warped, mat = pcv.transform.warp(img=kept_mask,
            	refimg=nir_img,
            	pts = [(495, 435), (500, 1120), (510, 1795), (1990, 440),
                                         (1985, 1125), (1976, 1795)],
            	refpts = [(100, 115), (100, 310), (100, 500), (530, 115),
                                                (530, 310), (530, 500)],
            	method='lmeds')

        img_warped_original, mat = pcv.transform.warp(img=img,
            	refimg=nir_img,
            	pts = [(495, 435), (500, 1120), (510, 1795), (1990, 440),
                                         (1985, 1125), (1976, 1795)],
            	refpts = [(100, 115), (100, 310), (100, 500), (530, 115),
                                                (530, 310), (530, 500)],
            	method='lmeds')


    analysis_nir = pcv.analyze_nir_intensity(nir_img,img_warped, 256, label = "default")

    # In[164]:
    #print(pcv.outputs)

    #hist_figure1, hist_data1 = pcv.visualize.histogram(nir_img, mask=kept_mask, hist_data=True)
    wvs1 = [480.0, 550.0, 670.0]
    wvs2 = [800.0]


    fused_img = pcv.image_fusion(img1=img_warped_original, img2=nir_img, wvs1=wvs1, wvs2=wvs2,
                             array_type="vis-nir_fusion")


    ndvi = pcv.spectral_index.ndvi(fused_img)

    # Pseudocolor the NDVI image
    colmap = pcv.visualize.pseudocolor(gray_img=ndvi.array_data, mask=img_warped, cmap="RdYlGn",
                                       min_value=-0.8, max_value=0.8)

    gdvi = pcv.spectral_index.gdvi(fused_img)

    psri = pcv.spectral_index.psri(fused_img)


    colmap = pcv.visualize.pseudocolor(gray_img=psri.array_data, mask=img_warped, cmap="RdYlGn",
                                       min_value=-0.8, max_value=0.8)


    colmap = pcv.visualize.pseudocolor(gray_img=gdvi.array_data, mask=img_warped, cmap="RdYlGn",
                                       min_value=-0.8, max_value=0.8)

    gdvi_array = np.histogram(gdvi.array_data, bins=256)


    # Calculate histogram
    gdvi_hyper = pcv.hyperspectral.analyze_index(index_array=gdvi, mask=img_warped, bins=100, min_bin=-2, max_bin=2, label="default")
    psri_hyper = pcv.hyperspectral.analyze_index(index_array=psri, mask=img_warped, bins=100, min_bin=np.amin(psri.array_data), max_bin=np.amax(psri.array_data), label="PSRI array data")
    ndvi_hyper = pcv.hyperspectral.analyze_index(index_array=ndvi, mask=img_warped, bins=100, min_bin=-1, max_bin=1, label="NDVI array data")
##    Histogram_for_Hyperspectral(gdvi.array_data, min = -2, max = 2, mask = img_warped)
##
##    Histogram_for_Hyperspectral(psri.array_data, min = np.amin(psri.array_data), max = np.amax(psri.array_data), mask = img_warped, label_addition = "PSRI array data")
##
##    Histogram_for_Hyperspectral(ndvi.array_data, min = -1, max = 1, mask = img_warped, label_addition = "NDVI array data")

    pcv.outputs.save_results(filename=args.result)
  #  pcv.outputs.clear()

if __name__=='__main__':
    main()

