class Outliers:
    def __init__(self):
        # ad count > 10000
        self.train_userid_outliers = [839368]
        self.test_userid_outliers = [3548147, 3522917, 3206914, 3093561, 3834944, 3648518]

        self.train_idx_outliers = [x-1 for x in self.train_userid_outliers]
        self.test_idx_outliers = [x - 3000001 for x in self.test_userid_outliers]

        self.outlier_age = 6
        self.outlier_gender = 1


if __name__ == '__main__':
    outliers = Outliers()
    print(outliers.train_userid_outliers)
    print(outliers.test_userid_outliers)
