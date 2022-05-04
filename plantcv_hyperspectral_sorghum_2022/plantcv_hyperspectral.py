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
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r", "--result", help="result file.", required=False)
    parser.add_argument("-w", "--writeimg", help="write out images.", default=False, action="store_true")
    parser.add_argument("-D", "--debug",
                        help="can be set to 'print' or None (or 'plot' if in jupyter) prints intermediate images.",
                        default=None)
    args = parser.parse_args()
    return args

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
    if "TV" in filename:

        maskeda_thresh1 = pcv.threshold.binary(gray_img=masked_a, threshold=150, max_value=255, object_type='light')
        if "z2000" in filename or "z1000" in filename:
            maskedb_thresh = pcv.threshold.binary(gray_img=masked_b, threshold=130, max_value=255, object_type='light')
        else:
            maskedb_thresh = pcv.threshold.binary(gray_img=masked_b, threshold=139, max_value=255, object_type='light')
    else:
        maskeda_thresh1 = pcv.threshold.binary(gray_img=masked_a, threshold=135, max_value=255, object_type='light')
        maskedb_thresh = pcv.threshold.binary(gray_img=masked_b, threshold=128, max_value=255, object_type='light')

    # In[20]:

    ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
    ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)

    ab_fill = pcv.fill(bin_img=ab, size=200)

    masked2 = pcv.apply_mask(img=masked, mask=ab_fill, mask_color='white')


    # In[23]:


    if "TV" in filename:
        ab_fill =  pcv.naive_bayes_classifier(rgb_img=masked2,
                                      pdf_file="TM17_TopView_NaiveBayes.pdf.txt")

        ab_fill = pcv.fill_holes(ab_fill["Leaf"])

        id_objects, obj_hierarchy = pcv.find_objects(img=masked2, mask=ab_fill)
    else:
        id_objects, obj_hierarchy = pcv.find_objects(img=masked2, mask=ab_fill)


    # In[25]:
    if "TV" in filename:
        roi1, roi_hierarchy= pcv.roi.rectangle(img=masked2, x=950, y=800, h=600, w=550)
    else:
        if "z2000" in filename or "z1000" in filename:
            roi1, roi_hierarchy= pcv.roi.rectangle(img=masked2, x=850, y=400, h=1400, w=1600)
        else :
        	roi1, roi_hierarchy= pcv.roi.rectangle(img=masked2, x=850, y=0, h=1500, w=800)

       # Decide which objects to keep
    roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img=img, roi_contour=roi1,
                                                                  roi_hierarchy=roi_hierarchy,
                                                                  object_contour=id_objects,
                                                                  obj_hierarchy=obj_hierarchy,
                                                                  roi_type='partial')

    if obj_area > 0:
        obj, mask = pcv.object_composition(img, roi_objects, hierarchy3)

    ############### Analysis ################
    ## below 3 lines were commented out before
    #    outfile=False
    #    if args.writeimg == True:
    #	outfile = args.outdir + "/" + filename

        # Find shape properties, output shape image (optional)
        shape_imgs = pcv.analyze_object(img, obj, mask)

        # Shape properties relative to user boundary line (optional)
        if "SV" in filename:
            if "z1000" in filename:
                boundary_img1 = pcv.analyze_bound_horizontal(img,obj,mask,1470)
            elif "z2000" in filename:
                boundary_img1 = pcv.analyze_bound_horizontal(img, obj, mask, 1680)
            else:
                boundary_img1 = pcv.analyze_bound_horizontal(img, obj, mask, 1370)

        elif "TV" in filename:
            # Shape properties relative to user boundary line (optional)
            if "z1000" in filename:
                boundary_img1 = pcv.analyze_bound_horizontal(img,obj,mask,1030)
                boundary_img2 = pcv.analyze_bound_vertical(img,obj,mask,1270)
            elif "z2000" in filename:
                boundary_img1 = pcv.analyze_bound_horizontal(img, obj, mask, 1045)
                boundary_img2 = pcv.analyze_bound_vertical(img, obj, mask, 1280)
            else:
                boundary_img1 = pcv.analyze_bound_horizontal(img, obj, mask, 1050)
                boundary_img2 = pcv.analyze_bound_vertical(img, obj, mask, 1260)

        # Determine color properties: Histograms, Color Slices, output color analyzed histogram (optional)
        color_histogram = pcv.analyze_color(img, kept_mask, 'all')

        # Pseudocolor the grayscale image
        pseudocolored_img = pcv.visualize.pseudocolor(gray_img=s, mask=kept_mask, cmap='jet')

        nir_path = pcv.get_nir(path=path,filename=filename)
        nir_img, path, img_filename = pcv.readimage(filename= nir_path, mode="native")


        # In[161]:
        if "SV" in filename:
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
                if "TM024_F_020617" in path:
                    img_warped, mat = pcv.transform.warp(img=kept_mask,
                        	refimg=nir_img,
                        	pts = [(1207, 1589), (1140, 1730), (1140, 1850), (1379, 1535),
                                                     (1389, 1734), (1389, 1854)],
                        	refpts = [(309, 177), (291, 214), (290, 245), (356, 157),
                                                            (362, 214), (361, 247)],
                        	method='lmeds')

                    img_warped_original, mat = pcv.transform.warp(img=img,
                        	refimg=nir_img,
                        	pts = [(1207, 1589), (1140, 1730), (1140, 1850), (1379, 1535),
                                                     (1389, 1734), (1389, 1854)],
                        	refpts = [(309, 177), (291, 214), (290, 245), (356, 157),
                                                            (362, 214), (361, 247)],
                        	method='lmeds')

                elif "TM023_F_010917" in path:
                    img_warped, mat = pcv.transform.warp(img=kept_mask,
                    	refimg=nir_img,
                    	pts = [(1156, 1240), (987, 1672), (938, 1828), (1659, 1130),
                                                 (1289, 1566), (1375, 1683)],
                    	refpts = [(298, 92), (253, 197), (249, 241), (434, 47),
                                                        (331, 172), (368, 195)],
                    	method='lmeds')

                    img_warped_original, mat = pcv.transform.warp(img=img,
                        	refimg=nir_img,
                   	    pts = [(978, 882), (994, 1665), (1050, 1870), (1462, 1458),
                                                 (1432, 1751), (1311, 1865)],
                    	refpts = [(253, 0), (252, 198), (266, 254), (379, 136),
                                                        (371, 239), (337, 255)],
                        	method='lmeds')

                else:
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
                    	pts = [(495, 435), (500, 1126), (512, 1797), (1990, 440),
                                                 (1985, 1125), (1976, 1795)],
                    	refpts = [(105, 119), (105, 313), (104, 499), (525, 119),
                                                        (525, 313), (527, 499)],
                    	method='lmeds')

                img_warped_original, mat = pcv.transform.warp(img=img,
                    	refimg=nir_img,
                    	pts = [(495, 435), (500, 1126), (512, 1797), (1990, 440),
                                                 (1985, 1125), (1976, 1795)],
                    	refpts = [(105, 119), (105, 313), (104, 499), (525, 119),
                                                        (525, 313), (527, 499)],
                    	method='lmeds')


        elif "TV" in filename:
            if "z1000" in filename:
                img_warped, mat = pcv.transform.warp(img=mask,
                    	refimg=nir_img,
                        pts = [(1010, 793), (920, 973), (1018, 1275), (1700, 734),
                                                 (1631, 975), (1460, 1211)],
                    	refpts = [(265, 193), (254, 242), (262, 325), (450, 170),
                                                        (436, 240), (388, 307)],
                    	method='lmeds')
                img_warped_original, mat = pcv.transform.warp(img=img,
                    	refimg=nir_img,
                        pts = [(1010, 793), (920, 973), (1018, 1275), (1700, 734),
                                                 (1631, 975), (1460, 1211)],
                    	refpts = [(265, 193), (254, 242), (262, 325), (450, 170),
                                                        (436, 240), (388, 307)],
                    	method='lmeds')

            elif "z2000" in filename:
               img_warped, mat = pcv.transform.warp(img=mask,
                    	refimg=nir_img,
                     pts = [(955, 720), (973, 1284), (916, 1370), (1531, 707),
                                                 (1620, 793), (1591, 1359)],
                    	refpts = [(251, 174), (254, 320), (238, 350), (407, 172),
                                                        (431, 199), (422, 350)],
                    	method='lmeds')
               img_warped_original, mat = pcv.transform.warp(img=img,
                    	refimg=nir_img,
                     pts = [(955, 720), (973, 1284), (916, 1370), (1531, 707),
                                                 (1620, 793), (1591, 1359)],
                    	refpts = [(251, 174), (254, 320), (238, 350), (407, 172),
                                                        (431, 199), (422, 350)],
                    	method='lmeds')

            elif "TK001" in path:
               if "z1" in filename:
                   nir_img =  cv2.flip(nir_img, 0)
                   img_warped, mat = pcv.transform.warp(img=mask,
                        	refimg=nir_img,
                            pts = [(312, 202), (1093, 862), (305, 1474), (2191, 197),
                                                     (2100, 938), (2202, 1471)],
                        	refpts = [(51, 24), (141, 102), (49, 172), (270, 24),
                                                            (259, 111), (270, 175)],
                        	method='lmeds')
                   img_warped_original, mat = pcv.transform.warp(img=img,
                        	refimg=nir_img,
                            pts = [(312, 202), (1093, 862), (305, 1474), (2191, 197),
                                                     (2100, 938), (2202, 1471)],
                        	refpts = [(51, 24), (141, 102), (49, 172), (270, 24),
                                                            (259, 111), (270, 175)],
                        	method='lmeds')

               elif "z2500" in filename:
                   nir_img =  cv2.flip(nir_img, 0)
                   img_warped, mat = pcv.transform.warp(img=mask,
                        	refimg=nir_img,
                            pts = [(820, 606), (92, 817), (812, 1439), (1653, 619),
                                                     (2407, 996), (1644, 1449)],
                        	refpts = [(117, 66), (30, 95), (117, 162), (214, 67),
                                                            (303, 116), (213, 164)],
                        	method='lmeds')
                   img_warped_original, mat = pcv.transform.warp(img=img,
                        	refimg=nir_img,
                            pts = [(820, 606), (92, 817), (812, 1439), (1653, 619),
                                                     (2407, 996), (1644, 1449)],
                        	refpts = [(117, 66), (30, 95), (117, 162), (214, 67),
                                                            (303, 116), (213, 164)],
                        	method='lmeds')

               elif "z500" in filename:
                   nir_img =  cv2.flip(nir_img, 0)
                   img_warped, mat = pcv.transform.warp(img=mask,
                        	refimg=nir_img,
                            pts = [(192, 95), (802, 975), (185, 1530), (2311, 90),
                                                     (1846, 1026), (2321, 1527)],
                        	refpts = [(39, 10), (107, 112), (35, 179), (284, 11),
                                                            (231, 118), (285, 181)],
                        	method='lmeds')
                   img_warped_original, mat = pcv.transform.warp(img=img,
                        	refimg=nir_img,
                            pts = [(192, 95), (802, 975), (185, 1530), (2311, 90),
                                                     (1846, 1026), (2321, 1527)],
                        	refpts = [(39, 10), (107, 112), (35, 179), (284, 11),
                                                            (231, 118), (285, 181)],
                        	method='lmeds')

            else:
                img_warped, mat = pcv.transform.warp(img=kept_mask,
                    	refimg=nir_img,
                        pts = [(306, 210), (728, 1062), (302, 1500), (2205, 207),
                                                 (1914, 1055), (2217, 1496)],
                    	refpts = [(59, 25), (179, 263), (49, 392), (594, 24),
                                                        (512, 261), (593, 392)],
                    	method='lmeds')
                img_warped_original, mat = pcv.transform.warp(img=img,
                    	refimg=nir_img,
                        pts = [(306, 210), (728, 1062), (302, 1500), (2205, 207),
                                                 (1914, 1055), (2217, 1496)],
                    	refpts = [(59, 25), (179, 263), (49, 392), (594, 24),
                                                        (512, 261), (593, 392)],
                    	method='lmeds')



        analysis_nir = pcv.analyze_nir_intensity(nir_img,img_warped, 256, label = "default")

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

        pcv.outputs.save_results(filename=args.result)

if __name__=='__main__':
    main()