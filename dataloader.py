class PCDataLoader(Dataset):
    def __init__(self, points, gt,labels, n_points=2048):
        self.n_points = n_points
        self.points = points
        self.gt = gt
        self.labels = labels

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):

        pc = self.points[index][:,0:3]
        gt = self.gt[index][:,0:3]
        labels = self.labels[index]
        return pc, gt, labels