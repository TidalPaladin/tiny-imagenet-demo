

levels = [4, 3, 6, 2]
num_classes = 100
width = 32
model = Resnet(num_classes, width, levels)

# tf.keras.InputLayer in TF 2.0
inputs = tf.keras.layers.InputLayer(
        input_shape=(512, 512, 3),
        batch_size=32,
        dtype=tf.float32
)
outputs = model(inputs.input)


model.summary()


FMT = "%-22s : %15s -> %-15s"
for layer in model.layers:
    name = type(layer).__name__
    inp= layer.input_shape
    out= layer.output_shape
    msg = FMT % (name, inp, out)
    print(msg)

