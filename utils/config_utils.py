import scipy.io as scio
import os

"""
Default Config in this Project
"""


class Config:
    def __init__(self):
        self.seed = 2020
        self.data_path = '/home/tione/notebook'
        self.gpu = 3
        self.model = 'CIN'

        if os.path.exists(os.path.join(self.data_path, 'excluded_id_over20.mat')):
            ids = scio.loadmat(os.path.join(self.data_path, 'excluded_id.mat'))
            self.excluded_creative_id = ids['excluded_creative_id'][0]
            self.excluded_ad_id = ids['excluded_ad_id'][0]
            self.excluded_advertiser_id = ids['excluded_advertiser_id'][0]
            self.excluded_product_id = ids['excluded_product_id'][0]
            self.excluded_industry = ids['excluded_industry'][0]
