from typing import Optional

import numpy as np
from scipy.optimize import minimize


class CRMP:
    """
    Implementation of Capacity-Resistance model for oil production forecasting
    The initial code is obtained from https://github.com/deepthisen/CapacitanceResistanceModel
    """

    def __init__(self, tau: np.array, gain_mat: np.asarray, qp0: np.array,
                 pressure_j: Optional[np.array] = None):
        self.tau = tau
        self.gain_mat = gain_mat
        self.n_inj = self.gain_mat.shape[0]
        self.n_prd = self.gain_mat.shape[1]
        self.qp0 = qp0
        self.params = [self.tau, self.gain_mat, self.qp0]

        self.q = None
        self.grad_tau = None
        self.grad_q0 = None
        self.grad_lambda = None
        self.grad_j = None
        self.j0 = None

        self.j = pressure_j
        self.include_press = pressure_j is not None

    def prim_prod(self):
        tau = self.tau
        qp0 = self.qp0
        del_t = self.del_t

        q_prev = self.q
        q_prime = q_prev * np.exp(-1 * del_t / tau)  # primary production
        self.q = q_prime

        return q_prime

    def inject_term(self):  # qi is array
        tau = self.tau
        del_t = self.del_t
        gain_mat = self.gain_mat
        qi_t = self.qi_t
        q_inj = (qi_t.reshape([1, -1]) @ gain_mat) * (1 - np.exp(-del_t / tau))

        self.q = self.q + q_inj
        return q_inj

    def bhp_term(self):
        tau = self.tau
        del_t = self.del_t
        del_bhp_t = self.del_bhp_t
        J = self.j
        q_bhp = -(J * tau * del_bhp_t / del_t) * (1 - np.exp(-del_t / tau))
        self.q = self.q + q_bhp

        return q_bhp

    def compute_grad_tau(self):
        del_t = self.del_t
        q_prev = self.q_prev
        q = self.q
        qi_t = self.qi_t
        tau = self.tau
        gain_mat = self.gain_mat

        a = np.exp(-del_t / tau) * (del_t / tau ** 2)

        grad_tau = (q_prev * a) - (qi_t.reshape([1, -1]) @ gain_mat) * a
        if self.include_press:
            J = self.j
            del_bhp_t = self.del_bhp_t
            b = del_bhp_t * J * np.exp(-del_t / tau) * ((1 / tau) + (1 / del_t))

            grad_tau -= b
        self.grad_tau = grad_tau

    def compute_grad_lambda(self):
        del_t = self.del_t
        qi_t = self.qi_t
        tau = self.tau

        b = 1.0 - np.exp(-del_t / tau)

        grad_lambda = qi_t.reshape(-1, 1) * b.reshape(1, -1)

        self.grad_lambda = grad_lambda

    def compute_grad_q0(self):
        t = self.t
        t0 = self.t0
        tau = self.tau

        self.grad_q0 = np.exp(-(t - t0) / tau)

    def compute_grad_j(self):
        del_t = self.del_t
        del_bhp_t = self.del_bhp_t
        tau = self.tau
        self.grad_j = tau * del_bhp_t / del_t

    def prod_pred(self, input_series, train=False):
        t_arr = input_series[0]
        qi_arr = input_series[1]
        bhp_arr = 0
        if self.include_press:
            bhp_arr = input_series[2]
            grad_js = []

        grad_taus = []
        grad_lambdas = []
        grad_q0s = [np.ones(self.n_prd)]
        if train:
            self.q = self.qp0.copy()
            self.q_prev = self.q.copy()
            qp0 = self.qp0
        else:
            qp0 = self.q_prev

        qp_arr = [qp0]

        self.t0 = t_arr[0]
        for i in range(1, len(t_arr)):
            self.del_t = t_arr[i] - t_arr[i - 1]
            self.t = t_arr[i]
            self.qi_t = qi_arr[i, :]
            qp_prim = self.prim_prod()
            qp_inj = self.inject_term()
            if self.include_press:
                self.del_bhp_t = bhp_arr[i, :] - bhp_arr[i - 1, :]
                qp_bhp = self.bhp_term()
            qp_arr.append(self.q)
            # compute gradients here
            if train:
                self.compute_grad_tau()
                self.compute_grad_lambda()
                self.compute_grad_q0()
                grad_taus.append(self.grad_tau)
                grad_lambdas.append(self.grad_lambda)
                grad_q0s.append(self.grad_q0)
                if self.include_press:
                    self.compute_grad_j()
                    grad_js.append(self.grad_j)
            self.q_prev = self.q.copy()
        self.q_pred = np.vstack(qp_arr)
        if train:
            self.grad_tau = np.vstack(grad_taus)
            self.grad_lambda = np.array(grad_lambdas)
            self.grad_q0 = np.vstack(grad_q0s)
            if self.include_press:
                self.grad_j = np.array(grad_js)
        return np.vstack(qp_arr)

    def compute_grads(self, q_obs):

        dmse_dtau = 2 * np.sum((q_obs[1:, :] - self.q_pred[1:, :]) * -self.grad_tau, axis=0) / np.max(q_obs,
                                                                                                      axis=0) ** 2

        dmse_dlambda = []
        for i in range(self.n_inj):
            dmse_dlambda.append(
                2 * np.sum((q_obs[1:, :] - self.q_pred[1:, :]) * -self.grad_lambda[:, i, :], axis=0) / np.max(q_obs,
                                                                                                              axis=0) ** 2)

        dmse_dlambda = np.vstack(dmse_dlambda)
        dmse_dq0 = 2 * np.sum((q_obs - self.q_pred) * -self.grad_q0, axis=0) / np.max(q_obs, axis=0) ** 2

        grads = np.concatenate([dmse_dtau.reshape(-1), dmse_dlambda.reshape(-1), dmse_dq0.reshape(-1)])
        if self.include_press:
            dmse_dj = 2 * np.sum((q_obs[1:, :] - self.q_pred[1:, :]) * -self.grad_j, axis=0) / np.max(q_obs,
                                                                                                      axis=0) ** 2
            grads = np.concatenate([grads, dmse_dj.reshape(-1)])

        return grads

    def compute_loss(self, q_obs):
        q_pred = self.q_pred
        mse = np.sum((((q_obs - q_pred) ** 2)), axis=0) / np.max(q_obs, axis=0) ** 2
        return mse

    def _preproc_matrices(self, x, input_series, train=True):
        n_prd = self.n_prd
        n_inj = self.n_inj
        j = None
        self.tau = x[:n_prd]
        self.gain_mat = x[n_prd:n_prd + (n_prd * n_inj)].reshape(n_inj, n_prd)
        self.q0 = x[n_prd + (n_prd * n_inj):(2 * n_prd) + (n_prd * n_inj)]
        if self.include_press:
            j = x[(2 * n_prd) + (n_prd * n_inj):]

        q_pred = self.prod_pred(input_series, train=True)

        return q_pred, j

    def obj_func_fit(self, x, input_series, q_obs):
        _, self.j0 = self._preproc_matrices(x, input_series, train=True)
        obj = np.sum(self.compute_loss(q_obs))
        return obj

    def jac_func_fit(self, x, input_series, q_obs):
        _, self.j = self._preproc_matrices(x, input_series, train=True)

        grads = self.compute_grads(q_obs)
        return grads

    def fit_model(self, input_series, q_obs, tau_0, gain_mat_0, q0_0, j0=None):

        n_inj = self.n_inj
        n_prd = self.n_prd

        x0 = np.concatenate([tau_0.reshape(-1), gain_mat_0.reshape(-1), q0_0.reshape(-1)])
        bnds = []
        # tau_bounds
        for i in range(n_prd):
            bnds.append((0.0001, 10))
        # gain_mat bounds
        for i in range(n_prd * n_inj):
            bnds.append((0.0, 1.0))
        # q0 bounds
        for i in range(n_prd):
            bnds.append((0.0, None))
        if self.include_press:
            x0 = np.concatenate([x0, j0.reshape(-1)])
            for i in range(n_prd):
                bnds.append((0.0, None))
        bnds = tuple(bnds)
        sum_constraints = ({'type': 'ineq', "fun": self.apply_gain_mat_constraint})

        res = minimize(self.obj_func_fit, x0, args=(input_series, q_obs),
                       jac=self.jac_func_fit, method='SLSQP', bounds=bnds,
                       constraints=sum_constraints, options={'disp': None,
                                                             'ftol': 1e-2, 'maxiter': 500})

        fit_param = res.x
        tau_fit = fit_param[:n_prd]
        gain_mat_fit = fit_param[n_prd:n_prd + (n_prd * n_inj)].reshape(n_inj, n_prd)
        q0_fit = fit_param[n_prd + (n_prd * n_inj):(2 * n_prd) + (n_prd * n_inj)]
        param_fits = [tau_fit, gain_mat_fit, q0_fit]
        if self.include_press:
            j_fit = fit_param[(2 * n_prd) + (n_prd * n_inj):]
            param_fits.append(j_fit)

        return param_fits

    def apply_gain_mat_constraint(self, x):
        gain_mat = x[self.n_prd:self.n_prd + (self.n_prd * self.n_inj)].reshape(self.n_inj, self.n_prd)
        residual = 1.0 - np.sum(gain_mat, axis=1)
        return residual
