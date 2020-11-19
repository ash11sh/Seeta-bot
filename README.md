# Seeta-bot


You can place your "Bot-token" in `bot.py` and deploy directly to heroku.


#### requirements:
* python-telegram-bot
* tensorflow


-----

ML Models used :

[The White-box CartoonGAN model](https://github.com/SystemErrorWang/White-box-Cartoonization) for Cartoon like images.
    --used tflite model [refer link](https://github.com/margaretmz/Cartoonizer-with-TFLite/)
    
[Fast arbitrary image style transfer](https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1) for artistic styles.

[BiseNetv2-Tensorflow](https://github.com/MaybeShewill-CV/bisenetv2-tensorflow) for hair segmentation and coloring.
    --used tflite model Quantized by [Katsuya Hyodo] (https://github.com/PINTO0309/PINTO_model_zoo/tree/master/057_BiSeNetV2)
