
# coding: utf-8

# In[41]:


import cv2
import numpy as np
from plantcv import plantcv as pcv
from plantcv.plantcv import params
from scipy import ndimage
import argparse
import os

# In[112]:


#argument parser
def options():
    parser = argparse.ArgumentParser(description="Arguments for the Ring of Power.")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-d", "--debug", help="Turn on debug, prints intermediate images.", default=None)
    parser.add_argument("-w","--writeimg", help="write out images.", default=False)
    parser.add_argument("-r","--result", help="result file.", required=False)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-n", "--npz", help="Background Mask for subtraction.", required=True)
    parser.add_argument("-p", "--pdf", help="PDF from Naive-Bayes.", required=True)
    args = parser.parse_args()
    return args

# In[113]:

############################
###        Main          ###
############################


def main():
    args = options() #create options object for argument parsing
    device = 0 #set device
    params.debug = args.debug #set debug


    # In[114]:


    img, path, filename = pcv.readimage(filename=args.image, debug = args.debug) #read in image
    background = pcv.transform.load_matrix(args.npz) #read in background mask image for subtraction


    # In[115]:


    device, mask = pcv.naive_bayes_classifier(img, args.pdf, device, args.debug) #naive bayes on image

    if args.writeimg:
        pcv.print_image(img=mask["94,104,47"], filename=os.path.join(args.outdir, args.image[:-4] + "_nb_mask.png"))


    # In[116]:


    new_mask = pcv.image_subtract(mask["94,104,47"], background) #subtract background noise


    # In[117]:


    #image blurring using scipy median filter
    blurred_img = ndimage.median_filter(new_mask, (5,1))
    blurred_img = ndimage.median_filter(blurred_img, (1,5))
    pcv.plot_image(blurred_img, cmap="gray")
    device, cleaned = pcv.fill(np.copy(blurred_img), np.copy(blurred_img), 50, 0, args.debug) #fill leftover noise


    # In[118]:


    #dilate and erode to repair plant breaks from background subtraction
    device, cleaned_dilated = pcv.dilate(cleaned, 6, 1, 0)
    device, cleaned = pcv.erode(cleaned_dilated, 6, 1, 0, args.debug)


    # In[119]:


    device, objects, obj_hierarchy = pcv.find_objects(img, cleaned, device, debug=args.debug) #find objects using mask
    if "TM015" in args.image:
        h = 1620
    elif "TM016" in args.image:
        h = 1555
    else:
        h = 1350
    roi_contour, roi_hierarchy = pcv.roi.rectangle(x=570, y=0, h=h, w=1900-550, img=img) #grab ROI


    # In[120]:


    #isolate plant objects within ROI
    device, roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi_contour, roi_hierarchy, objects, obj_hierarchy, device, debug=args.debug)

    #Analyze only images with plants present.
    if roi_objects>0:
        # In[121]:


        # Object combine kept objects
        device, plant_contour, plant_mask = pcv.object_composition(img=img, contours=roi_objects,
                                                                   hierarchy=hierarchy, device=device, debug=args.debug)

        if args.writeimg:
            pcv.print_image(img=plant_mask, filename=os.path.join(args.outdir, args.image[:-4] + "_mask.png"))


        # In[122]:

        outfile = False
        if args.writeimg:
            outfile = os.path.join(args.outdir, args.image)

        # Find shape properties, output shape image (optional)
        device, shape_header, shape_data, shape_img = pcv.analyze_object(img=img, imgname=args.image, obj=plant_contour,
                                                                         mask=plant_mask, device=device, debug=args.debug,
                                                                         filename=outfile)


        # In[123]:


        if "TM015" in args.image:
            line_position = 380
        elif "TM016" in args.image:
            line_position = 440
        else:
            line_position = 690

        # Shape properties relative to user boundary line (optional)
        device, boundary_header, boundary_data, boundary_img = pcv.analyze_bound_horizontal(img=img, obj=plant_contour,
                                                                                 mask=plant_mask, line_position=line_position,
                                                                                 device=device, debug=args.debug, filename=outfile)


        # In[124]:


        # Determine color properties: Histograms, Color Slices and Pseudocolored Images,
        # output color analyzed images (optional)
        device, color_header, color_data, color_img = pcv.analyze_color(img=img, imgname=args.image, mask=plant_mask, bins=256,
                                                                        device=device, debug=args.debug, hist_plot_type=None,
                                                                        pseudo_channel="v", pseudo_bkg="img", resolution=300,
                                                                        filename=outfile)


        # In[55]:


        # Output shape and color data
        result = open(args.result, "a")
        result.write('\t'.join(map(str, shape_header)) + "\n")
        result.write('\t'.join(map(str, shape_data)) + "\n")
        for row in shape_img:
            result.write('\t'.join(map(str, row)) + "\n")
        result.write('\t'.join(map(str, color_header)) + "\n")
        result.write('\t'.join(map(str, color_data)) + "\n")
        result.write('\t'.join(map(str, boundary_header)) + "\n")
        result.write('\t'.join(map(str, boundary_data)) + "\n")
        result.write('\t'.join(map(str, boundary_img)) + "\n")
        for row in color_img:
            result.write('\t'.join(map(str, row)) + "\n")
        result.close()

if __name__ == '__main__':
    main()