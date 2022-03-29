# moved class to separate file to stop pickle error
class ReviewUnique(): # stores unique points at intervals for a review
    def __init__(self, network_epoch, review_num, review_length, num_points, interval_tuple):
        self.epoch = network_epoch
        self.rev_num = review_num
        self.rev_len = review_length
        self.num_points = num_points
        self.intervals = interval_tuple
        self.data = {}

    def add_data(self, end_point, num_unique):
        if end_point not in self.intervals:
            return False
        self.data[end_point] = num_unique
        return True

    def get_data(self, end_point):
        return self.data.get(end_point)
