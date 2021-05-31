import torch


class DeviceProvider:

    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            self.device_info = 'cuda:0'
            self.device = torch.device('cuda:0')
        else:
            self.device_info = 'cpu'
            self.device = torch.device('cpu')

    def provide(self):
        return self.device

    @property
    def device_info(self):
        return self.__device_info

    @device_info.setter
    def device_info(self, info):
        self.__device_info = info
