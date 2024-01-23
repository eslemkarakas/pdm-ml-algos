# import standard packages
import numpy as np
import matplotlib.pyplot as plt

class Sanity:
    def __init__(self, y_train, y_val):
        self.min_yt_value = y_train.min()
        self.min_vt_value = y_val.min()
        self.max_yt_value = y_train.max()
        self.max_vt_value = y_val.max()
        self.mean_yt_value = y_train.mean()
        self.mean_vt_value = y_val.mean()
        
    @property
    def _get_yt_vals(self):
        return self.min_yt_value, self.max_yt_value, self.mean_yt_value
    
    @property
    def _get_vt_vals(self):
        return self.min_vt_value, self.max_vt_value, self.mean_vt_value
    
    def check_min_val(self):
        assert (self.y_val < self.min_yt_value).sum() == 0
    
    def check_max_val(self):
        assert (self.y_val > self.max_yt_value).sum() == 0
    
    def check_mean_diff_val(self):
        assert np.abs(self.mean_yt_value - self.mean_vt_value) < 0.01
    
    def visualize_histograms(self):
        plt.figure(figsize=(12,6))
        plt.hist(self.y_val, bins=100, alpha=0.5, density=True, label='y_val')
        plt.hist(self.y_train, bins=100, alpha=0.5, density=True, label='y_train')
        plt.legend()
        plt.title('Check Sanity')
        plt.show()
        
    def check_sanity(self):
        self.check_min_val()
        self.check_max_val()
        self.check_mean_diff_val()
        self.visualize_histograms()
        
def call_sanity_check(y_train, y_val):
    try:
        sanity = Sanity(y_train, y_val)
    except Exception as e:
        print(f'{str(e)}')