# Adversarial Examples Using Style Transfer

An little exercise project for paper "Adversarial Camouflage: Hiding Physical World Attacks with Natural Styles" (CVPR 2020), with PyTorch implementation.

- Step 1: Open the  imagenet directory and store all the content, style (along with their corresponding masks) images into the corresponding folders. To match the content image with its style image and masks, please use the same file name (For example, use 1.png for the same group of content, style and masks).

- Step 2: Run `main.py`. You can modify the following arguments:
  - `target_label`: specify which label you want the adversarial sample to be classified to
  - `num_steps`: the iteration steps of the training process
  - `content_weight`: weight for content loss, to reserve the original content
  - `style_weight`:  weight for style loss, to transfer the original style to the target style image
  - `adv_weight`:  weight for adversarial loss, to misclassify the image

- Step 3: A result directory will be generated. Check the directory for the generated adversarial examples.