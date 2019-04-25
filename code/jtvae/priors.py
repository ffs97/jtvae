import numpy as np

import torch
import torch.nn as nn

from torch.autograd import Variable


def sample_gumbel(shape, eps=1e-20):
    U = np.random.uniform(0, 1, shape)
    U = - np.log(eps - np.log(U + eps))

    return U


def sample_bernoulli(probs):
    shape = probs.shape

    return torch.where(
        torch.rand(shape) - probs.cpu() < 0,
        torch.zeros(shape), torch.ones(shape)
    ).cuda()


class Normal(nn.Module):
    def __init__(self, hidden_layer_size, dim):
        super(Normal, self).__init__()

        self.dim = dim

        self.mean = nn.Linear(hidden_layer_size, dim)
        self.log_var = nn.Linear(hidden_layer_size, dim)

    def sample_reparametrization_variable(self, n):
        return torch.randn(n, self.dim)

    def sample_feed(self, n=None, vecs=None, gamma=1.0):
        if vecs is not None:
            mean = self.mean(vecs)
            # log_var = -torch.abs(self.log_var(vecs))

            samples = torch.randn_like(mean)
            # samples = mean + samples * torch.exp(log_var) / gamma
            samples = mean + samples / gamma

        elif n is not None:
            samples = torch.randn(n, self.dim)

        else:
            raise AttributeError

        return samples

    def inverse_reparametrize(self, epsilon, mean, log_var):
        return mean + torch.exp(log_var / 2) * epsilon

    def kl_from_prior(self, mean, log_var, eps=1e-20):
        kl = torch.exp(log_var) + mean ** 2 - 1. - log_var
        kl = torch.mean(0.5 * torch.sum(kl, dim=1))

        return kl

    def __call__(self, vecs, sample=True):
        mean = self.mean(vecs)

        if not sample:
            return mean

        log_var = -torch.abs(self.log_var(vecs))

        epsilon = self.sample_reparametrization_variable(mean.shape[0]).cuda()
        z_vecs = self.inverse_reparametrize(epsilon, mean, log_var)

        kl = self.kl_from_prior(mean, log_var)

        return z_vecs, kl


class Discrete(nn.Module):
    def __init__(self, hidden_layer_size, dim, n_classes):
        super(Discrete, self).__init__()

        self.dim = dim

        self.n_classes = n_classes

        self.logits = nn.Linear(hidden_layer_size, dim * n_classes)

    def sample_reparametrization_variable(self, n):
        return torch.tensor(
            sample_gumbel((n, self.dim, self.n_classes)),
            dtype=torch.float
        )

    def sample_feed(self, n=None, logits=None):
        # TODO: Change logits to x_vecs

        if logits is not None:
            shape = logits.shape
            samples = sample_gumbel(shape) + logits

        elif n is not None:
            shape = (n, self.dim, self.n_classes)
            samples = sample_gumbel(shape)

        else:
            raise AttributeError

        samples = np.reshape(samples, (-1, self.n_classes))
        samples = np.argmax(samples, axis=1)

        samples = torch.from_numpy(np.eye(self.n_classes)[samples])

        samples = np.reshape(samples, shape)

        return samples

    def inverse_reparametrize(self, epsilon, logits, temperature):
        logits = logits.view(-1, self.n_classes)

        res = epsilon.view(-1, self.n_classes)

        res = (logits + res) / temperature
        res = nn.functional.softmax(res, dim=1)

        if self.n_classes == 2:
            res = res[:, 0]
            res = res.view(-1, self.dim)
        else:
            res = res.view(-1, self.dim, self.n_classes)

        return res

    def kl_from_prior(self, logits, eps=1e-20):
        logits = logits.view(-1, self.n_classes)
        q_z = nn.functional.softmax(logits, dim=1)

        kl = q_z * (torch.log(q_z + eps) - np.log(1.0 / self.n_classes))
        kl = kl.view(-1, self.dim * self.n_classes)
        kl = torch.mean(torch.sum(kl, dim=1))

        return kl

    def __call__(self, vecs):
        # TODO: Add sample=False
        logits = self.logits(vecs).view(-1, self.dim, self.n_classes)

        epsilon = self.sample_reparametrization_variable(
            logits.shape[0]
        ).cuda()
        z_vecs = self.inverse_reparametrize(epsilon, logits, 0.2)

        kl = self.kl_from_prior(logits)

        return z_vecs, kl


class RBM(nn.Module):
    def __init__(self, hidden_layer_size, visible_dim, hidden_dim, beta=1.0,
                 requires_grad=False, init_gibbs_iters=1000, kl_gibbs_samples=10, gen_feed_gibbs_gap=100):
        super(RBM, self).__init__()

        self.beta = beta
        self.hidden_dim = hidden_dim
        self.visible_dim = visible_dim
        self.dim = self.hidden_dim + self.visible_dim

        self.init_gibbs_iters = init_gibbs_iters
        self.kl_gibbs_samples = kl_gibbs_samples
        self.gen_feed_gibbs_gap = gen_feed_gibbs_gap

        self.bv = Variable(torch.zeros(1, visible_dim),
                           requires_grad=requires_grad).cuda()
        self.bh = Variable(torch.zeros(1, hidden_dim),
                           requires_grad=requires_grad).cuda()
        self.w = Variable(torch.randn(visible_dim, hidden_dim),
                          requires_grad=requires_grad).cuda()

        self.b = torch.cat([self.bv, self.bh], 1)
        self.logits = nn.Linear(2 * hidden_layer_size, self.dim)

    def free_energy(self, samples):
        samples_visible = samples[:, :self.visible_dim]
        samples_hidden = samples[:, self.visible_dim:]

        energy = torch.sum(samples * self.b, dim=1) + torch.sum(
            torch.matmul(
                samples_visible, self.w
            ) * samples_hidden, dim=1
        )

        return energy

    def _propup(self, samples_visible):
        pre_sigmoid = self.bh + torch.matmul(
            samples_visible, self.w
        )
        return pre_sigmoid, torch.sigmoid(pre_sigmoid)

    def _sample_hgv(self, samples_visible):
        _, probs = self._propup(samples_visible)
        return sample_bernoulli(probs)

    def _propdown(self, samples_hidden):
        pre_sigmoid = self.bv + torch.matmul(
            samples_hidden, torch.transpose(self.w, 0, 1)
        )
        return pre_sigmoid, torch.sigmoid(pre_sigmoid)

    def _sample_vgh(self, samples_hidden):
        _, probs = self._propdown(samples_hidden)
        return sample_bernoulli(probs)

    def _gibbs_vhv(self, samples_visible):
        samples_hidden = self._sample_hgv(samples_visible)
        samples_visible = self._sample_vgh(samples_hidden)

        return samples_visible, samples_hidden

    def _gibbs_vhv_k(self, samples_visible, k):
        for _ in range(k - 1):
            samples_visible, _ = self._gibbs_vhv(samples_visible)

        return self._gibbs_vhv(samples_visible)

    def generate_gibbs_samples(self, k, g=1):
        clip = 0
        if not hasattr(self, "samples_visible"):
            clip = self.init_gibbs_iters // g
            k += clip

            probs = torch.ones((1, self.visible_dim)).cuda() / 2.0

            samples_visible = sample_bernoulli(probs)
            self.samples_visible = self._gibbs_vhv_k(
                samples_visible, self.init_gibbs_iters
            )[0]

        def sample(visible_samples):
            samples = torch.cat(self._gibbs_vhv_k(
                visible_samples, g
            ), dim=1)

            return samples

        samples = []
        for _ in range(k):
            samples_ = sample(self.samples_visible)
            self.samples_visible = samples_[:, :self.visible_dim]

            samples.append(samples_)

        samples = samples[clip:]
        samples = torch.cat(samples, dim=0)

        return samples

    def sample_reparametrization_variable(self, n):
        U = np.random.uniform(0, 1, (n, self.dim))
        U = np.log(U / (1 - U))

        return torch.tensor(U, dtype=torch.float)

    def sample_feed(self, n):
        # TODO: Add option to pass logits

        z_vecs = self.generate_gibbs_samples(n, g=self.gen_feed_gibbs_gap)
        z_vecs = torch.split(
            z_vecs, (self.visible_dim, self.hidden_dim), dim=1
        )

        return z_vecs

    def log_partition(self, samples):
        return - torch.mean(self.free_energy(samples))

    def inverse_reparametrize(self, epsilon, log_ratios, temperature):
        res = (log_ratios + epsilon) / temperature
        res = torch.sigmoid(res)

        return res

    def kl_from_prior(self, log_ratios, samples, zeta, eps=1e-20):
        probs = torch.sigmoid(log_ratios)

        log_posterior = torch.sum(
            torch.log(zeta * probs + (1 - zeta) * (1 - probs)),
            dim=-1
        )
        log_prior_un = - self.free_energy(zeta)

        kl = torch.mean(
            log_posterior - log_prior_un
        )
        kl += self.log_partition(samples)

        return self.beta * kl

    def __call__(self, vecs, sample=True):
        logits = self.logits(vecs)

        if sample:
            epsilon = self.sample_reparametrization_variable(
                logits.shape[0]
            ).cuda()
            z_vecs = self.inverse_reparametrize(epsilon, logits, 0.2)

            samples = self.generate_gibbs_samples(self.kl_gibbs_samples, g=1)
            kl = self.kl_from_prior(logits, samples, z_vecs)

        else:
            shape = logits.shape

            z_vecs = torch.where(
                logits.cpu() < 0,
                torch.zeros(shape), torch.ones(shape)
            ).cuda()

        z_vecs = torch.split(
            z_vecs, (self.visible_dim, self.hidden_dim), dim=1
        )

        if sample:
            return z_vecs, kl
        else:
            return z_vecs
