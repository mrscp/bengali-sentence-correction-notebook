from processors.data import RawData


class Main:
    def __init__(self):
        raw_data = RawData()
        lines, vocabulary = raw_data.get_data()
        print(lines[:5])


if __name__ == '__main__':
    Main()
