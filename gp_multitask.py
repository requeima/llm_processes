import torch
import gpytorch
import os
import argparse
import pickle
import numpy as np
from matplotlib import pyplot as plt


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)



def main():
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment.')
    parser.add_argument('--data_path', type=str, help='Path to pkl file with x, y data.')
    parser.add_argument('--output_dir', type=str, help='Path to directory where output results are written.')
    parser.add_argument('--plot_dir', type=str, help='Path to directory where plots are written.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_tasks', type=int, default=1)
    args = parser.parse_args()
    print(args)

    # load the data
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    
    train_x = torch.from_numpy(np.array(data['x_train'])).type(torch.float32)
    train_y = torch.from_numpy(np.array(data['y_train'])).type(torch.float32)
    test_x = torch.from_numpy(np.array(data['x_test'])).type(torch.float32)
    test_y = torch.from_numpy(np.array(data['y_test'])).type(torch.float32)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=args.num_tasks)
    model = MultitaskGPModel(train_x, train_y, likelihood, args.num_tasks)

    training_iter = 500

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
        print('Iter %d/%d - Loss: %.3f' % (
            i + 1, training_iter, loss.item()
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

    print('MAE = {} NLL = {}'.format(data['mae'], data['nll']))

    fig, axes = plt.subplots(args.num_tasks, 1, figsize=(7, args.num_tasks * 4), sharex=True)
    for i in range(args.num_tasks):
        # plot training points
        axes[i].scatter(train_x.numpy(), (train_y.numpy())[:, i], label='Training points', c='black', marker='v')
        # plot true function
        if 'x_true' in data:
            axes[i].plot(data['x_true'], data['y_true'][:, i], label='True Function', c='black')
        # Plot predictive means as orange line
        axes[i].plot(test_x.numpy(), (observed_pred.mean.numpy())[:, i], color='tab:orange', label='Mean')
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Shade between the lower and upper confidence bounds
        axes[i].fill_between(test_x.numpy(), (lower.numpy())[:, i], (upper.numpy())[:, i], color='tab:orange', alpha=0.4, label='Confidence')

        axes[i].set_title(args.experiment_name + ' MAE = {:.3f} NLL = {:.3f}'.format(data['mae'][i], data['nll']))
        axes[i].grid()
        axes[i].legend()
        plt.savefig(os.path.join(args.plot_dir, args.experiment_name + '.pdf'), bbox_inches='tight')


if __name__ == '__main__':
    main()
