from keras.models import load_model


class calc_model:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.decoder = None

    def import_model(self, model_path):
        self.model = load_model(model_path)

    def import_encoder(self, encoder_path):
        self.encoder = load_model(encoder_path)

    def import_decoder(self, decoder_path):
        self.decoder = load_model(decoder_path)




def calc_phases_vol_metrics(composition_metrics, model):
    return model.predict(composition_metrics)


def output_normalize(composition_metrics):
    for i in range(composition_metrics.shape[0]):
        row_sum = sum(composition_metrics[i, :])
        for j in range(composition_metrics.shape[1]):
            composition_metrics[i, j] = composition_metrics[i, j] / row_sum

    return composition_metrics