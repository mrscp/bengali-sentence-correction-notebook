from processors.data import RawData


class Main:
    def __init__(self):
        raw_data = RawData()
        raw_data.get_data()


if __name__ == '__main__':
    Main()
