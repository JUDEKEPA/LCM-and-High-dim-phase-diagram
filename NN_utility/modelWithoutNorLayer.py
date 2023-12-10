from tensorflow.keras.models import Model, load_model


def save_without_normalize_layer(model, path):
    layers_before_normalize = model.layers[:-1]

    truncated_model = Model(inputs=layers_before_normalize[0].input, outputs=layers_before_normalize[-1].output)

    for orig_layer, trunc_layer in zip(model.layers, truncated_model.layers):
        trunc_layer.set_weights(orig_layer.get_weights())

    truncated_model.save(path)


def load_model_without_normalize_layer(path):
    loaded_model = load_model(path)

    return loaded_model