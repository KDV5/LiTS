# from __future__ import print_function, division
# import SimpleITK as sitk
# import numpy as np
# import cv2
# import os
#
# trainImage = r'./train_label'
# trainLiverMask = r'./train_mask'
# testImage = r'./test_label'
# testLiverMask = r'./test_mask'
# # trainTumorMask = r'./train'
# #
#
# def getRangImageDepth(image):
#     """
#     :param image:
#     :return:rangofimage depth
#     """
#     fistflag = True
#     startposition = 0
#     endposition = 0
#     for z in range(image.shape[0]):
#         notzeroflag = np.max(image[z])
#         if notzeroflag and fistflag:
#             startposition = z
#             fistflag = False
#         if notzeroflag:
#             endposition = z
#
#     return startposition, endposition
#
#
# def subimage_generator(image, mask, patch_block_size, numberxy, numberz):
#     """
#     generate the sub images and masks with patch_block_size
#     :param image:
#     :param patch_block_size:
#     :param stride:
#     :return:
#     """
#     width = np.shape(image)[1]
#     height = np.shape(image)[2]
#     imagez = np.shape(image)[0]
#     block_width = np.array(patch_block_size)[1]
#     block_height = np.array(patch_block_size)[2]
#     blockz = np.array(patch_block_size)[0]
#     stridewidth = (width - block_width) // numberxy
#     strideheight = (height - block_height) // numberxy
#     stridez = (imagez - blockz) // numberz
#     # step 1:if stridez is bigger 1,return  numberxy * numberxy * numberz samples
#     if stridez >= 1 and stridewidth >= 1 and strideheight >= 1:
#         step_width = width - (stridewidth * numberxy + block_width)
#         step_width = step_width // 2
#         step_height = height - (strideheight * numberxy + block_height)
#         step_height = step_height // 2
#         step_z = imagez - (stridez * numberz + blockz)
#         step_z = step_z // 2
#         hr_samples_list = []
#         hr_mask_samples_list = []
#         for z in range(step_z, numberz * (stridez + 1) + step_z, numberz):
#             for x in range(step_width, numberxy * (stridewidth + 1) + step_width, numberxy):
#                 for y in range(step_height, numberxy * (strideheight + 1) + step_height, numberxy):
#                     if np.max(mask[z:z + blockz, x:x + block_width, y:y + block_height]) != 0:
#                         hr_samples_list.append(image[z:z + blockz, x:x + block_width, y:y + block_height])
#                         hr_mask_samples_list.append(mask[z:z + blockz, x:x + block_width, y:y + block_height])
#         hr_samples = np.array(hr_samples_list).reshape((len(hr_samples_list), blockz, block_width, block_height))
#         hr_mask_samples = np.array(hr_mask_samples_list).reshape(
#             (len(hr_mask_samples_list), blockz, block_width, block_height))
#         return hr_samples, hr_mask_samples
#     # step 2: other sutitation,return one samples
#     else:
#         nb_sub_images = 1 * 1 * 1
#         hr_samples = np.zeros(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float)
#         hr_mask_samples = np.zeros(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float)
#         rangz = min(imagez, blockz)
#         rangwidth = min(width, block_width)
#         rangheight = min(height, block_height)
#         hr_samples[0, 0:rangz, 0:rangwidth, 0:rangheight] = image[0:rangz, 0:rangwidth, 0:rangheight]
#         hr_mask_samples[0, 0:rangz, 0:rangwidth, 0:rangheight] = mask[0:rangz, 0:rangwidth, 0:rangheight]
#         return hr_samples, hr_mask_samples
#
#
# def make_patch(image,mask, patch_block_size, numberxy, numberz, startpostion, endpostion):
#     """
#     make number patch
#     :param image:[depth,512,512]
#     :param patch_block: such as[64,128,128]
#     :return:[samples,64,128,128]
#     expand the dimension z range the subimage:[startpostion-blockz//2:endpostion+blockz//2,:,:]
#     """
#     blockz = np.array(patch_block_size)[0]
#     imagezsrc = np.shape(image)[0]
#     subimage_startpostion = startpostion - blockz // 2
#     subimage_endpostion = endpostion + blockz // 2
#     if subimage_startpostion < 0:
#         subimage_startpostion = 0
#     if subimage_endpostion > imagezsrc:
#         subimage_endpostion = imagezsrc
#     if (subimage_endpostion - subimage_startpostion) < blockz:
#         subimage_startpostion = 0
#         subimage_endpostion = imagezsrc
#     imageroi = image[subimage_startpostion:subimage_endpostion, :, :]
#     maskroi = mask[subimage_startpostion:subimage_endpostion, :, :]
#     image_subsample, mask_subsample = subimage_generator(image=imageroi, mask=maskroi, patch_block_size=patch_block_size,
#                                                          numberxy=numberxy, numberz=numberz)
#     return image_subsample, mask_subsample
#
#
# '''
# This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
# read_Image_mask fucntion get image and mask
# '''
#
#
# def load_itk(filename):
#     """
#     load mhd files and normalization 0-255
#     :param filename:
#     :return:
#     """
#     rescalFilt = sitk.RescaleIntensityImageFilter()
#     rescalFilt.SetOutputMaximum(255)
#     rescalFilt.SetOutputMinimum(0)
#     # Reads the image using SimpleITK
#     itkimage = rescalFilt.Execute(sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32))
#     return itkimage
#
#
# def gen_image_mask(srcimg, seg_image, index, shape, numberxy, numberz, image_path, mask_path):
#     # step 1 get mask effective range(startpostion:endpostion)
#     startpostion, endpostion = getRangImageDepth(seg_image)
#     # step 2 get subimages (numberxy*numberxy*numberz,16, 256, 256)
#     sub_srcimages, sub_liverimages = make_patch(srcimg, seg_image, patch_block_size=shape, numberxy=numberxy,
#                                                numberz=numberz, startpostion=startpostion, endpostion=endpostion)
#     # step 3 only save subimages (numberxy*numberxy*numberz,16, 256, 256)
#     samples, imagez = np.shape(sub_srcimages)[0], np.shape(sub_srcimages)[1]
#     sub_masks = sub_liverimages.astype(np.float32)
#     sub_masks = np.clip(sub_masks, 0, 255).astype('uint8')
#     sub_image = sub_srcimages.astype(np.float32)
#     sub_image = np.clip(sub_image, 0, 255).astype('uint8')
#     for j in range(samples):
#         if np.max(sub_masks[j, :, :, :]) == 255:
#             image_save_path = image_path + "/" + str(index) + "_" + str(j) + "/"
#             mask_save_path = mask_path + "/" + str(index) + "_" + str(j) + "/"
#             if not os.path.exists(image_save_path) and not os.path.exists(mask_save_path):
#                 os.makedirs(image_save_path)
#                 os.makedirs(mask_save_path)
#                 print(image_save_path+' created!')
#             for z in range(imagez):
#                 cv2.imwrite(image_save_path + str(z) + ".bmp", sub_image[j, z, :, :])
#                 cv2.imwrite(mask_save_path + str(z) + ".bmp", sub_masks[j, z, :, :])
#
#
# def get_img_mask_data(img_path, index):
#     seg = sitk.ReadImage(img_path + index + "/mask.nii", sitk.sitkUInt8)
#     segimg = sitk.GetArrayFromImage(seg)
#     src = load_itk(img_path + index + "/data.nii")
#     srcimg = sitk.GetArrayFromImage(src)
#     return segimg, srcimg
#
#
# def preparetraindata():
#     TrainSeqs = ['47', '21', '79', '110', '52', '61', '100', '13', '97', '127', '116', '27', '44', '56', '83', '28',
#                  '26', '124', '104', '31', '126', '65', '6', '92', '63', '7', '32', '0', '93', '29', '109', '37', '4',
#                  '72', '22', '70', '129', '123', '46', '67', '38', '18', '82', '8', '66', '19', '106', '122', '105',
#                  '11', '90', '55', '68', '98', '53', '49', '75', '107', '9', '102', '117', '42', '108', '89', '85',
#                  '103', '17', '25', '62', '54', '69', '5', '43', '35', '2', '23', '3', '88', '20', '118', '95', '84',
#                  '58', '74', '77', '60', '76', '81', '39', '59', '24', '10', '78', '48', '86', '94', '45', '57', '125',
#                  '33', '121', '40', '14', '91']
#     TestSeqs = ['51', '12', '96', '112', '73', '120', '15', '128', '114', '119', '130', '71', '99', '64', '80', '115',
#                 '1', '111', '30', '34', '113', '16', '101', '36', '50', '41', '87']
#     for i in range(len(TrainSeqs)):
#         print('start generate '+TrainSeqs[i]+'!')
#         segimg, srcimg = get_img_mask_data(r'/mnt/02520c27-ec8e-4661-b88f-05aa2011ffa7/lhk/LiTS/ori/',TrainSeqs[i])
#         print('get data finished!')
#         seg_liverimage = segimg.copy()
#         seg_liverimage[segimg > 0] = 255
#         gen_image_mask(srcimg, seg_liverimage, i, shape=(16, 256, 256), numberxy=5, numberz=10,
#                        image_path=trainImage, mask_path=trainLiverMask)
#         print('generate '+TrainSeqs[i]+' finished!')
#     for i in range(len(TestSeqs)):
#         print('start generate '+TestSeqs[i]+'!')
#         segimg, srcimg = get_img_mask_data(r'/mnt/02520c27-ec8e-4661-b88f-05aa2011ffa7/lhk/LiTS/ori/', TestSeqs[i])
#         print('get data finished!')
#         seg_liverimage = segimg.copy()
#         seg_liverimage[segimg > 0] = 255
#         gen_image_mask(srcimg, seg_liverimage, i, shape=(16, 256, 256), numberxy=5, numberz=10,
#                        image_path=testImage, mask_path=testLiverMask)
#         print('generate '+TestSeqs[i]+' finished!')
#
#         # seg_tumorimage = segimg.copy()
#         # seg_tumorimage[segimg == 1] = 0
#         # seg_tumorimage[segimg == 2] = 255
#         # gen_image_mask(srcimg, seg_tumorimage, i, shape=(16, 256, 256), numberxy=5, numberz=10)
#
#
# preparetraindata()
#
#
# def get_denoise_data(path, name):
#     # step 1: get raw_list(denoise/train/name/HDCT/0_HD_1.raw...)
#     HD_list = os.listdir(path+name+'/HDCT/')
#     length = len(HD_list)
#     HD_images = np.zeros((length, 512, 512), dtype=np.float32)
#     for i in range(length):
#         slice = HD_list[i].split('_')[2].split('.')[0]-1
#         img = np.fromfile(HD_list[i], dtype='float32')
#         img = img.reshape((512, 512))
#         HD_images[slice, :, :] = img
#     LD_list = os.listdir(path+name+'/LDCT/')
#     length = len(LD_list)
#     LD_images = np.zeros((length, 512, 512), dtype=np.float32)
#     for i in range(length):
#         slice = LD_list[i].split('_')[2].split('.')[0]-1
#         img = np.fromfile(HD_list[i], dtype='float32')
#         img = img.reshape((512, 512))
#         LD_images[slice, :, :] = img
#     return HD_images, LD_images
#
#
# def prepare_denoise_data():
#     TrainSeqs = ['47', '21', '79', '110', '52', '61', '100', '13', '97', '127', '116', '27', '44', '56', '83', '28',
#                  '26', '124', '104', '31', '126', '65', '6', '92', '63', '7', '32', '0', '93', '29', '109', '37', '4',
#                  '72', '22', '70', '129', '123', '46', '67', '38', '18', '82', '8', '66', '19', '106', '122', '105',
#                  '11', '90', '55', '68', '98', '53', '49', '75', '107', '9', '102', '117', '42', '108', '89', '85',
#                  '103', '17', '25', '62', '54', '69', '5', '43', '35', '2', '23', '3', '88', '20', '118', '95', '84',
#                  '58', '74', '77', '60', '76', '81', '39', '59', '24', '10', '78', '48', '86', '94', '45', '57', '125',
#                  '33', '121', '40', '14', '91']
#     TestSeqs = ['51', '12', '96', '112', '73', '120', '15', '128', '114', '119', '130', '71', '99', '64', '80', '115',
#                 '1', '111', '30', '34', '113', '16', '101', '36', '50', '41', '87']
#     for i in range(len(TrainSeqs)):
#         print('start generate '+TrainSeqs[i]+'!')
#         segimg, srcimg = get_img_mask_data(r'/mnt/02520c27-ec8e-4661-b88f-05aa2011ffa7/lhk/LiTS/ori/', TrainSeqs[i])
#         print('get data finished!')
#         seg_liverimage = segimg.copy()
#         seg_liverimage[segimg > 0] = 255
#         gen_image_mask(srcimg, seg_liverimage, i, shape=(16, 256, 256), numberxy=5, numberz=10,
#                        image_path=trainImage, mask_path=trainLiverMask)
#         print('generate '+TrainSeqs[i]+' finished!')
#     for i in range(len(TestSeqs)):
#         print('start generate '+TestSeqs[i]+'!')
#         segimg, srcimg = get_img_mask_data(r'/mnt/02520c27-ec8e-4661-b88f-05aa2011ffa7/lhk/LiTS/ori/', TestSeqs[i])
#         print('get data finished!')
#         seg_liverimage = segimg.copy()
#         seg_liverimage[segimg > 0] = 255
#         gen_image_mask(srcimg, seg_liverimage, i, shape=(16, 256, 256), numberxy=5, numberz=10,
#                        image_path=testImage, mask_path=testLiverMask)
#         print('generate '+TestSeqs[i]+' finished!')
