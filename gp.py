import torch
import gpytorch
import os
import argparse
import pickle
import numpy as np
from matplotlib import pyplot as plt


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def main():
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment.')
    parser.add_argument('--data_path', type=str, help='Path to pkl file with x, y data.')
    parser.add_argument('--output_dir', type=str, help='Path to directory where output results are written.')
    parser.add_argument('--plot_dir', type=str, help='Path to directory where plots are written.')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    print(args)

    # load the data
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    
    train_x = torch.from_numpy(np.array(data['x_train']))
    train_y = torch.from_numpy(np.array(data['y_train']))
    test_x = torch.from_numpy(np.array(data['x_test']))
    test_y = torch.from_numpy(np.array(data['y_test']))

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    training_iter = 250

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    data['gen'] = observed_pred.mean.numpy()
    data['mse'] = gpytorch.metrics.mean_squared_error(observed_pred, test_y, squared=True)
    data['mae']= gpytorch.metrics.mean_absolute_error(observed_pred, test_y).numpy()
    data['nll'] = gpytorch.metrics.negative_log_predictive_density(observed_pred, test_y).numpy()
    lower, upper = observed_pred.confidence_region()
    data['gen_lower'] = lower.numpy()
    data['gen_upper'] = upper.numpy()
    os.makedirs(args.output_dir, exist_ok=True)  # create the output directory
    with open(os.path.join(args.output_dir, args.experiment_name + '.pkl'), "wb") as f:
        pickle.dump(data, f)

    print('MAE = {:.5f} NLL = {:.5f}'.format(data['mae'], data['nll']))
    
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(7, 4))

        # Plot test data as purple
        # ax.scatter(test_x.numpy(), test_y.numpy(), label='Test points', c='tab:purple')
        # Plot training data as green
        ax.scatter(train_x.numpy(), train_y.numpy(), label='Training points', c='black', marker='v')
        # Plot true function in blue
        if 'x_true' in data:
            ax.plot(data['x_true'],data['y_true'], label='True Function', c='tab:cyan')

        # Plot predictive means as orange line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), color='tab:orange', label='Mean')
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), color='tab:orange', alpha=0.4, label='Confidence')

        ax.set_title(args.experiment_name + ' MAE = {:.5f} NLL = {:.5f}'.format(data['mae'], data['nll']))
        ax.grid()
        ax.legend()
        plt.savefig(os.path.join(args.plot_dir, args.experiment_name + '.pdf'), bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    main()
