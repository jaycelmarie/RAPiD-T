from api import Detector

# Initialize detector
detector = Detector(model_name='rapid',
                    weights_path='./weights/RAPiD.ckpt',
                    use_cuda=False)

# A simple example to run on a single image and plt.imshow() it
detector.detect_one(img_path='./examples/warehouse_short/warehouse_000451.png',
                    input_size=1024, conf_thres=0.3,
                    visualize=True)
