from config import config
from data.dataset import load_data
from models.cnn_model import CNNModel
from navigation.slam import Localization

def main():
    # Ініціалізація та запуск основних компонентів
    data = load_data(config.DATA_PATH)
    model = CNNModel()
    localization = Localization()

    # Приклад використання
    model.train(data)
    localization.start()

if __name__ == "__main__":
    main()
