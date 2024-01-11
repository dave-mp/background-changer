import pixellib
from pixellib.tune_bg import alter_bg
from tensorflow.keras.layers import BatchNormalization

change_bg = alter_bg()
change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
change_bg.change_bg_img(f_image_path = "sample.jpeg",b_image_path = "background.png", output_image_name="new_img.jpg")