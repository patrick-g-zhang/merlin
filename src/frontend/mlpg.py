import numpy as np
from numpy import dot
import logging
from numpy import float64
import pdb
import sys
# Adding this before the bandmat import lets us import .pyx files without running bandmat's setup.py:
#import pyximport; pyximport.install()
sys.path.append("..")
import bandmat as bm
import bandmat.linalg as bla
from utils.linalg import cholesky_inv_banded
from nnmnkwii.paramgen.mlpg_helper import full_window_mat
class MLParameterGenerationFast(object):
    def __init__(self, delta_win = [-0.5, 0.0, 0.5], acc_win = [1.0, -2.0, 1.0]):
        self.delta_win = delta_win
        self.acc_win   = acc_win
        ###assume the delta and acc windows have the same length
        self.win_length = int(len(delta_win)/2)

    def build_win_mats(self, windows, frames):
        win_mats = []
        for l, u, win_coeff in windows:
            assert l >= 0 and u >= 0
            assert len(win_coeff) == l + u + 1
            win_coeffs = np.tile(np.reshape(win_coeff, (l + u + 1, 1)), frames)
            win_mat = bm.band_c_bm(u, l, win_coeffs).T
            win_mats.append(win_mat)

        return win_mats

    def build_poe(self, b_frames, tau_frames, win_mats, sdw=None):
        """
            Computes natural parameters for a Gaussian product-of-experts model.
            The natural parameters (b-value vector and precision matrix) are returned.
            The returned precision matrix is stored as a BandMat.
                Mathematically the b-value vector is given as:
            b = \sum_d \transpose{W_d} \tilde{b}_d
                and the precision matrix is given as:
            P = \sum_d \transpose{W_d} \text{diag}(\tilde{tau}_d) W_d
                where $W_d$ is the window matrix for window $d$ as specified by an element
             of `win_mats`, $\tilde{b}_d$ is the sequence over time of b-value
                parameters for window $d$ as given by a column of `b_frames`, and
                $\tilde{\tau}_d$ is the sequence over time of precision parameters for
                window $d$ as given by a column of `tau_frames`.
        """
        if sdw is None:
            sdw = max([ win_mat.l + win_mat.u for win_mat in win_mats ])
        num_windows = len(win_mats)
        frames = len(b_frames)
        assert np.shape(b_frames) == (frames, num_windows)
        assert np.shape(tau_frames) == (frames, num_windows)
        assert all([ win_mat.l + win_mat.u <= sdw for win_mat in win_mats ])

        b = np.zeros((frames,))
        prec = bm.zeros(sdw, sdw, frames)

        for win_index, win_mat in enumerate(win_mats):
            bm.dot_mv_plus_equals(win_mat.T, b_frames[:, win_index], target=b)
            bm.dot_mm_plus_equals(win_mat.T, win_mat, target_bm=prec,
                                  diag=float64(tau_frames[:, win_index]))

        return b, prec

    def generation(self, features, covariance, static_dimension):

        windows = [
            (0, 0, np.array([1.0])),
            (1, 1, np.array([-0.5, 0.0, 0.5])),
            (1, 1, np.array([1.0, -2.0, 1.0])),
        ]
        num_windows = len(windows)

        frame_number = features.shape[0]

        logger = logging.getLogger('param_generation')
        logger.debug('starting MLParameterGeneration.generation')

        gen_parameter = np.zeros((frame_number, static_dimension))

        win_mats = self.build_win_mats(windows, frame_number)
        mu_frames = np.zeros((frame_number, 3))
        var_frames = np.zeros((frame_number, 3))
        # pdb.set_trace()
        for d in range(static_dimension):
            try:
                var_frames[:, 0] = covariance[:, d]
                var_frames[:, 1] = covariance[:, static_dimension+d]
                var_frames[:, 2] = covariance[:, static_dimension*2+d]
                mu_frames[:, 0] = features[:, d]
                mu_frames[:, 1] = features[:, static_dimension+d]
                mu_frames[:, 2] = features[:, static_dimension*2+d]
                var_frames[0, 1] = 100000000000
                var_frames[0, 2] = 100000000000
                var_frames[frame_number-1, 1] = 100000000000
                var_frames[frame_number-1, 2] = 100000000000
                b_frames = mu_frames / var_frames
                tau_frames = 1.0 / var_frames

                b, prec = self.build_poe(b_frames, tau_frames, win_mats)
                mean_traj = bla.solveh(prec, b)
                gen_parameter[0:frame_number, d] = mean_traj
            except IndexError:
                pdb.set_trace()
        return  gen_parameter

    def get_static_features(inputs, num_windows, stream_sizes=[180, 3, 1, 3],
                            has_dynamic_features=[True, True, False, True],
                            streams=[True, True, True, True]):
        """Get static features from static+dynamic features.
        """
        _, _, D = inputs.size()
        if stream_sizes is None or (len(stream_sizes) == 1 and has_dynamic_features[0]):
            return inputs[:, :, :D // num_windows]
        if len(stream_sizes) == 1 and not has_dynamic_features[0]:
            return inputs

        # Multi stream case
        ret = []
        start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
        for start_idx, size, v, enabled in zip(
                start_indices, stream_sizes, has_dynamic_features, streams):
            if not enabled:
                continue
            if v:
                static_features = inputs[:, :, start_idx:start_idx + size // num_windows]
            else:
                static_features = inputs[:, :, start_idx:start_idx + size]
            ret.append(static_features)
        return np.stack(ret, axis=-1)



    def unit_variance_mlpg_matrix(self, windows, T):
        """
            Compute MLPG matrix assuming input is normalized to have unit-variances.
            Let :math:`\mu` is the input mean sequence (``num_windows*T x static_dim``),
            :math:`W` is a window matrix ``(T x num_windows*T)``, assuming input is
            normalized to have unit-variances, MLPG can be written as follows:
            .. math::
                y = R \mu
            where
            .. math::
                R = (W^{T} W)^{-1} W^{T}
            Here we call :math:`R` as the MLPG matrix.
            Args:
                windows: (list): List of windows.
                T (int): Number of frames.
            Returns:
                numpy.ndarray: MLPG matrix (``T x num_windows*T``).
            See also:
                :func:`nnmnkwii.autograd.UnitVarianceMLPG`,
                :func:`nnmnkwii.paramgen.mlpg`.
            Examples:
                >>> from nnmnkwii import paramgen as G
                >>> import numpy as np
                >>> windows = [
                         (0, 0, np.array([1.0])),
                         (1, 1, np.array([-0.5, 0.0, 0.5])),
                         (1, 1, np.array([1.0, -2.0, 1.0])),
                     ]
                >>> G.unit_variance_mlpg_matrix(windows, 3)
                array([[  2.73835927e-01,   1.95121944e-01,   9.20177400e-02,
                          9.75609720e-02,  -9.09090936e-02,  -9.75609720e-02,
                         -3.52549881e-01,  -2.43902430e-02,   1.10864742e-02],
                       [  1.95121944e-01,   3.41463417e-01,   1.95121944e-01,
                          1.70731708e-01,  -5.55111512e-17,  -1.70731708e-01,
                         -4.87804860e-02,  -2.92682916e-01,  -4.87804860e-02],
                       [  9.20177400e-02,   1.95121944e-01,   2.73835927e-01,
                          9.75609720e-02,   9.09090936e-02,  -9.75609720e-02,
                          1.10864742e-02,  -2.43902430e-02,  -3.52549881e-01]], dtype=float32)
        """
        win_mats = self.build_win_mats(windows, T)
        sdw = np.max([win_mat.l + win_mat.u for win_mat in win_mats])
        P = bm.zeros(sdw, sdw, T)
        for win_index, win_mat in enumerate(win_mats):
            bm.dot_mm_plus_equals(win_mat.T, win_mat, target_bm=P)
        chol_bm = bla.cholesky(P, lower=True)
        Pinv = cholesky_inv_banded(chol_bm.full(), width=chol_bm.l + chol_bm.u + 1)
        cocatenated_window = full_window_mat(win_mats, T)
        return Pinv.dot(cocatenated_window.T).astype(np.float32)
