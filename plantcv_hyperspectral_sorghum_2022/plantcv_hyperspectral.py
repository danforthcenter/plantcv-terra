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

def Histogram_for_Hyperspectral(array_data, mask, min=-2, max =2, label_addition = "gdvi array data"):
    fig_hist, hist_data = histogram(array_data, mask=mask, bins=256, lower_bound=min, upper_bound=max, title=None,
                                    hist_data=True)

    masked_array = array_data[np.where(mask > 0)]
    _mean = np.average(masked_array)
    _median = np.median(masked_array)
    _std = np.std(masked_array)

    _mean = np.float64(_mean)
    _median =np.float64(_median)
    _std = np.float64(_std)

    outputs.add_observation(sample='default', variable='Mean_Value ' + str(label_addition), trait='Mean Value of ' + str(label_addition),
                            method='custom', scale='none', datatype=float,
                            value=_mean, label='none')

    outputs.add_observation(sample='default', variable='Median_Value ' + str(label_addition), trait='Median Value of ' + str(label_addition),
                            method='custom', scale='none', datatype=float,
                            value=_median, label='none')
    outputs.add_observation(sample='default', variable='STD_Value ' + str(label_addition), trait='STDev Value of ' + str(label_addition),
                            method='custom', scale='none', datatype=float,
                            value=_std, label='none')
    outputs.add_observation(sample='default', variable=str(label_addition)+'_frequencies', trait=str(label_addition) +  ' frequencies',
                            method='custom', scale='frequency', datatype=list,
                            value=hist_data['hist_count'].tolist(), label=hist_data["pixel intensity"].tolist())


    return hist_data["pixel intensity"].tolist(), hist_data['hist_count'].tolist()


def main():
    args=options()

    pcv.params.debug = args.debug
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

    skeleton = pcv.morphology.skeletonize(mask=kept_mask)

    #pcv.params.line_thickness = 3
    if "z2000" in filename:
        img1,test1,test2 = pcv.morphology.prune(skel_img=skeleton, size=40)
    else :
        img1,test1,test2 = pcv.morphology.prune(skel_img=skeleton, size=82)

    # Identify branch points
    branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=img1, mask=kept_mask)

    # Identify tip points
    tip_pts_mask = pcv.morphology.find_tips(skel_img=img1, mask=None)

    # In[128]:

    pcv.params.line_thickness = 5 #was 8 and changed to 5 which is default

    # Segment a skeleton into pieces
    seg_img, edge_objects = pcv.morphology.segment_skeleton(skel_img=img1, mask=kept_mask)

    # Sort segments into leaf objects and stem objects
    leaf_obj, stem_obj = pcv.morphology.segment_sort(skel_img=img1, objects=edge_objects,
                                                     mask=kept_mask)

    # Identify segments
    segmented_img, labeled_img = pcv.morphology.segment_id(skel_img=img1, objects=leaf_obj,
                                                           mask=kept_mask)

    # In[132]:

    labeled_img2 = pcv.morphology.segment_path_length(segmented_img=segmented_img,
                                                      objects=leaf_obj)

    # In[129]:

    labeled_img3 = pcv.morphology.segment_euclidean_length(segmented_img=segmented_img,
                                                           objects=leaf_obj)

    # In[131]:

    labeled_img4 = pcv.morphology.segment_curvature(segmented_img=segmented_img,
                                                    objects=leaf_obj)


    # In[130]:

    labeled_img5 = pcv.morphology.segment_angle(segmented_img=segmented_img,
                                                                              objects=leaf_obj)
    # In[135]:


    #NOTE: Below sections img6 and img7 were commented out because they were not able to run.

##   labeled_img6 = pcv.morphology.segment_tangent_angle(segmented_img=segmented_img,
##                                                      objects=leaf_obj, size=15)

##     In[ ]

##    labeled_img7 = pcv.morphology.segment_insertion_angle(skel_img=img1,
##                                                          segmented_img=segmented_img,
##                                                          leaf_objects=leaf_obj,
##                                                          stem_objects=stem_obj,
##                                                          size=25)

    # In[156]:

    print(path)
   # print(img)
    print(filename)
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


    analysis_nir = pcv.analyze_nir_intensity(nir_img,img_warped, 256, histplot=True, label = "default")

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

    Histogram_for_Hyperspectral(gdvi.array_data, min = -2, max = 2, mask = img_warped)

    Histogram_for_Hyperspectral(psri.array_data, min = np.amin(psri.array_data), max = np.amax(psri.array_data), mask = img_warped, label_addition = "PSRI array data")

    Histogram_for_Hyperspectral(ndvi.array_data, min = -1, max = 1, mask = img_warped, label_addition = "NDVI array data")

    pcv.outputs.save_results(filename=args.result)
  #  pcv.outputs.clear()

if __name__=='__main__':
    main()

