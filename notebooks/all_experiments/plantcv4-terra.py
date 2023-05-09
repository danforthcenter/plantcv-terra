#!/usr/bin/env python
# coding: utf-8

# In[1]:


from plantcv import plantcv as pcv
from plantcv.parallel import workflow_inputs
import numpy as np
import math
import cv2
from skimage.util import img_as_ubyte
import json
import os


# In[2]:


# This function takes the log transformed image and applies an affine transformation to correct the colors based on the color card
def affine_color_correction(img, source_matrix, target_matrix):
    h,w,c = img.shape
    
    n = source_matrix.shape[0]
    S = np.concatenate((source_matrix[:,1:].copy(),np.ones((n,1))),axis=1)
    T = target_matrix[:,1:].copy()
    
    tr = T[:,0]
    tg = T[:,1]
    tb = T[:,2]
    
    ar = np.matmul(np.linalg.pinv(S), tr)
    ag = np.matmul(np.linalg.pinv(S), tg)
    ab = np.matmul(np.linalg.pinv(S), tb)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pix = np.concatenate((img_rgb.reshape(h*w,c).astype(np.float64)/255, np.ones((h*w,1))), axis=1)
    
    img_r_cc = (255*np.clip(np.matmul(img_pix,ar),0,1)).astype(np.uint8)
    img_g_cc = (255*np.clip(np.matmul(img_pix,ag),0,1)).astype(np.uint8)
    img_b_cc = (255*np.clip(np.matmul(img_pix,ab),0,1)).astype(np.uint8)
    
    img_cc = np.stack((img_b_cc,img_g_cc,img_r_cc), axis=1).reshape(h,w,c)
    
    return img_cc


# In[11]:


def get_source_matrix(filepath, exp_config):
    path_parts = filepath.split(os.sep)
    exp, barcode, date, snapshot, filename = path_parts[-5:]
    exp_id = exp.split("_")[0]
    zoom = filename.split("_")[3]
    with open(exp_config, "r") as fp:
        config = json.load(fp)
    mat_s = np.array(config[exp_id][zoom]["cc_matrix"])
    return mat_s


# In[18]:


# Define workflow inputs
# TM017
args = workflow_inputs(*["config"])
# TM018
# args = WorkflowInputs(images=["./random_samples/TM018_F_100416/Fa095AA034672/2016-10-12/snapshot212586/VIS_SV_0_z1_h1_g0_e65_v500_419003_0.png",
#                              "./random_samples/TM018_F_100416/Fa095AA034672/2016-10-12/snapshot212586/NIR_SV_0_z1_h1_g0_e18000_v500_419006_0.png"],
#                       names="vis,nir",
#                       result="results.json",
#                       outdir=".",
#                       writeimg=True,
#                       debug="plot")
# TM027
# args = WorkflowInputs(images=["./random_samples/TM027_F_091517/Fa159AB049436/2017-10-10/snapshot82852/VIS_SV_0_z1_h1_g0_e65_v500_165272_0.png",
#                              "./random_samples/TM027_F_091517/Fa159AB049436/2017-10-10/snapshot82852/NIR_SV_0_z1_h1_g0_e18000_v500_165275_0.png"],
#                       names="vis,nir",
#                       result="results.json",
#                       outdir=".",
#                       writeimg=True,
#                       debug="plot")



# Global parameters
pcv.params.debug = args.debug


# In[14]:


# Read the VIS image
rgb_img, rgb_path, rgb_filename = pcv.readimage(filename=args.vis)


# In[15]:


# Apply a gamma correction to linearize the white balance
gamma_img = pcv.transform.gamma_correct(img=rgb_img, gamma=0.45, gain=1)


# In[19]:


# Target and source matrices for color correction
mat_t = pcv.transform.std_color_matrix(pos=3)
mat_s = get_source_matrix(args.vis, args.config)


# In[20]:


# Apply an affine color correction
img_cc = affine_color_correction(img=gamma_img, source_matrix=mat_s, target_matrix=mat_t)
# pcv.plot_image(img_cc)


# In[21]:


# Set the region of interest
#roi, roi_str = pcv.roi.rectangle(img=img_cc, x=650, y=100, h=1225, w=1200)
roi = pcv.roi.rectangle(img=img_cc, x=650, y=100, h=1225, w=1200)


# In[22]:


bkgd = pcv.rgb2gray_cmyk(rgb_img=img_cc, channel="k")


# In[23]:


bkgd_mask = pcv.threshold.binary(gray_img=bkgd, threshold=165, max_value=255, object_type="light")


# In[24]:


bkgd_mask = pcv.dilate(gray_img=bkgd_mask, ksize=5, i=1)


# In[25]:


gray_img = pcv.rgb2gray_lab(rgb_img=img_cc, channel="b")


# In[26]:


bin_img = pcv.threshold.mean(gray_img=gray_img, block_size=100, offset=-5, object_type="light")


# In[27]:


bkgd_overlap = pcv.logical_and(bin_img1=bkgd_mask, bin_img2=bin_img)


# In[28]:


remove_overlap = bin_img - bkgd_overlap
#pcv.plot_image(remove_overlap)


# In[29]:


edges = pcv.canny_edge_detect(img=img_cc)


# In[30]:


edges_overlap = pcv.logical_and(bin_img1=bkgd_mask, bin_img2=edges)


# In[31]:


remove_overlap_edges = edges - edges_overlap
#pcv.plot_image(remove_overlap_edges)


# In[32]:


combine = pcv.logical_or(bin_img1=remove_overlap, bin_img2=remove_overlap_edges)


# In[33]:


fill_gaps = pcv.closing(gray_img=combine)

bot_img = fill_gaps[1750:, :]
bot_blur = pcv.median_blur(gray_img=bot_img, ksize=(5, 1))
bot_blur = pcv.median_blur(gray_img=bot_blur, ksize=(1, 5))
blur_img = fill_gaps.copy()
blur_img[1750:, :] = bot_blur

# In[34]:


clean_bin = pcv.fill(bin_img=blur_img, size=10)


# In[35]:


objs = pcv.find_objects(img=img_cc, mask=clean_bin)


# In[36]:


flt_objs, flt_mask, flt_area = pcv.roi_objects(img=img_cc, roi=roi, obj=objs)


# In[37]:

if flt_area > 0:
    plant, plant_mask = pcv.object_composition(img=img_cc, contours=flt_objs.contours[0], hierarchy=flt_objs.hierarchy[0])


    # In[38]:


    shape_img = pcv.analyze_object(img=img_cc, obj=plant, mask=plant_mask)


    # In[39]:


    hline_img = pcv.analyze_bound_horizontal(img=img_cc, obj=plant, mask=plant_mask, line_position=1380)


    # In[40]:


    hist = pcv.analyze_color(rgb_img=img_cc, mask=plant_mask, colorspaces="hsv")


    # In[41]:


    nir, nir_path, nir_filename = pcv.readimage(filename=args.nir)


    # In[42]:


    warped_mask, warp_mat = pcv.transform.warp(img=plant_mask, refimg=nir,
                                               pts=[(498,430), (500,1126), (512,1797), (1990,440), (1985,1125), (1976,1795)],
                                               refpts=[(104,113), (106,307), (104,499), (529,116), (525,313), (527,499)],
                                               method='lmeds')


    # In[43]:


    nir_hist = pcv.analyze_nir_intensity(gray_img=nir, mask=warped_mask, bins=100)


    # In[44]:


    warp_img = pcv.transform.warp_align(img=img_cc, refimg=nir, mat=warp_mat)


    # In[45]:


    msi = pcv.image_fusion(img1=warp_img, img2=nir, wvs1=[480, 520, 700], wvs2=[800])


    # In[46]:


    ndvi = pcv.spectral_index.ndvi(hsi=msi)


    # In[47]:


    ndvi_hist = pcv.hyperspectral.analyze_index(index_array=ndvi, mask=warped_mask, min_bin=-1, max_bin=1)

    pcv.print_image(shape_img, os.path.join(args.outdir, rgb_filename[:-4]) + "_shape.png")
    pcv.print_image(hline_img, os.path.join(args.outdir, rgb_filename[:-4]) + "_hline.png")

pcv.outputs.save_results(args.result)




