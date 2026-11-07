import torch


class JointLoss(torch.nn.Module):

    def __init__(self, loss_fns, weights=None):
        self.loss_fns = loss_fns
        if weights is None:
            self.weights = [1/len(self.loss_fns) for _ in range(len(self.loss_fns))]
        else:
            self.weights = weights
        super(JointLoss, self).__init__()

    def forward(self, *kwargs):
        """
        Report weighted average of each underlying loss function.
        """

        loss = [self.weights[i] * self.loss_fns[i](*kwargs) for i in range(len(self.loss_fns))]
        return sum(loss)
