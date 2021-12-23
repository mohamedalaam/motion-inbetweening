
class Model:
    def __init__(self,weights_dir_path=None):
        self.load_components()
        if weights_dir_path:
            self.load_pre_trained(weights_dir_path=weights_dir_path)
        pass
    def load_components(self):
        pass

    def load_pre_trained(self,weights_dir_path):
        pass

    def predict(self):
        pass
    def train(self):
        pass




