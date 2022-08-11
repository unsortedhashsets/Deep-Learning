class LR:

    infoMsg = ""

    def __init__(self, infoMsg):
        self.infoMsg = infoMsg

    def info(self):
        print(self.infoMsg)

if __name__ == "__main__":
    lr = LR("Here I will try to make python program with the same features as Jupiter notebook has");
    lr.info()